"""Per-meeting artifact generators — speaker data + timeline.

Both functions read ``meeting_dir/journal.jsonl``, rebuild
``detected_speakers.json`` / ``speaker_lanes.json`` /
``timeline.json`` from the deduplicated event stream, and rewrite
the journal with stable ``Speaker N`` seq_index numbering. Used
by the meeting finalize / stop / reprocess paths so the four
on-disk artifacts (journal, detected_speakers, speaker_lanes,
timeline) stay in sync.

Pulled out of ``server.py`` so the lifecycle module can import
them at top level instead of via lazy ``meeting_scribe.server``
imports.
"""

from __future__ import annotations

import logging

from meeting_scribe.runtime import state

logger = logging.getLogger(__name__)


def _generate_speaker_data(
    meeting_dir,
    journal_path,
    _json,
    expected_speakers: int | None = None,
) -> None:
    """Generate state.detected_speakers.json and speaker_lanes.json from journal events.

    Applies speaker_correction journal entries (the same way get_meeting does)
    so reprocessed speaker stats include user-assigned names, not just raw
    cluster IDs. Without this, reprocessing wipes speaker identities.

    ``expected_speakers``: when set, after the size-based ghost dissolve
    runs, additionally force-collapse the smallest surviving clusters into
    their nearest neighbor (by time) until exactly N remain. This is the
    "user told us the count, deliver exactly that" lever — it catches
    leftover stale-cluster events that the size threshold missed.
    """
    # First pass: collect corrections (segment_id → speaker_name)
    corrections: dict[str, str] = {}
    best: dict[str, dict] = {}
    for line in journal_path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            e = _json.loads(line)
        except Exception:
            continue
        if e.get("type") == "speaker_correction":
            corrections[e.get("segment_id", "")] = e.get("speaker_name", "")
            continue
        if not e.get("is_final") or not e.get("text"):
            continue
        sid = e.get("segment_id")
        rev = e.get("revision", 0)
        if sid not in best or rev > best[sid].get("revision", 0):
            best[sid] = e

    # Apply corrections to event speakers
    for sid, e in best.items():
        if sid in corrections:
            name = corrections[sid]
            speakers = e.get("speakers", [])
            if speakers:
                speakers[0]["identity"] = name
                speakers[0]["display_name"] = name
            else:
                e["speakers"] = [{"identity": name, "display_name": name, "cluster_id": 0}]

    events = sorted(best.values(), key=lambda e: e.get("start_ms", 0))

    # Orphan speaker reassignment — no more "Speaker Unknown".
    # Events that landed before diarize caught up can have empty speakers[]
    # or cluster_id == 0. Strategy:
    #   1. First pass: reassign orphans to nearest real speaker WITHIN 5s
    #      (high confidence — same conversational beat).
    #   2. Second pass: any orphan that still has no neighbor falls back
    #      to the nearest-in-time real speaker with NO distance limit.
    #      Meetings often open with one host talking for a minute before
    #      anyone else joins; the 5s window can never reach them, but
    #      attributing those segments to the only nearby real voice is
    #      far better than dropping a whole minute of audio under cluster
    #      id 0 — which (a) leaks into the seq_index allocation and
    #      poisons "Speaker 1", and (b) leaves a gaping unattributed
    #      block in the timeline.
    #   3. If the meeting has NO real speakers at all, leave orphans
    #      alone — that's a genuinely speakerless recording.
    _ORPHAN_WINDOW_MS = 5000
    real_events = [
        e for e in events if e.get("speakers") and (e["speakers"][0].get("cluster_id") or 0) > 0
    ]

    def _attribute_to(orphan_e, donor_e) -> None:
        donor_sp = donor_e["speakers"][0]
        orphan_e["speakers"] = [
            {
                "cluster_id": donor_sp.get("cluster_id"),
                "display_name": donor_sp.get("identity") or donor_sp.get("display_name"),
                "identity": donor_sp.get("identity") or donor_sp.get("display_name"),
                "source": "orphan_reassigned",
            }
        ]

    if real_events:
        for e in events:
            sp = e.get("speakers") or []
            cid = sp[0].get("cluster_id") if sp else None
            if sp and cid and cid > 0:
                continue
            mid_ms = (e.get("start_ms", 0) + e.get("end_ms", e.get("start_ms", 0))) / 2
            nearest = min(
                real_events,
                key=lambda r: abs(((r.get("start_ms", 0) + r.get("end_ms", 0)) / 2) - mid_ms),
            )
            # Always attribute — distance is informational only. Better to
            # guess "the only voice in the room right now" than to surface
            # a black hole as cluster 0.
            _attribute_to(e, nearest)

    # ── Tiny-cluster dissolution + expected-count collapse ─────
    # The full-audio diarize pass + its consolidation work on diarize
    # output, but events the new pass DIDN'T temporally overlap keep
    # their stale live-diarize cluster_id. Those leftovers create ghost
    # clusters in the final speakers list (the "Speaker 2 with 2 segs /
    # 7s" symptom). We do TWO passes:
    #
    #  1. Size-based dissolve: any cluster below the ghost floor (<15s
    #     OR <5 segments) gets its events reassigned to the nearest
    #     real cluster by time.
    #  2. Expected-count collapse: when the caller passed
    #     ``expected_speakers=N``, additionally fold the smallest
    #     surviving clusters into their nearest neighbor until exactly
    #     N remain. This catches stale leftovers that survive the size
    #     check but the user knows aren't real.
    #
    # All reassignments happen on the in-memory ``events`` list, which
    # is the single source of truth for state.detected_speakers.json,
    # speaker_lanes.json, and the journal rewrite below — keeping all
    # four artifacts aligned.
    _DISSOLVE_MAX_DURATION_MS = 15_000
    _DISSOLVE_MAX_SEGMENTS = 5

    def _cluster_sizes() -> tuple[dict[int, int], dict[int, int]]:
        durations: dict[int, int] = {}
        counts: dict[int, int] = {}
        for ev in events:
            sp = ev.get("speakers") or []
            cid = sp[0].get("cluster_id") if sp else 0
            if not cid or cid <= 0:
                continue
            durations[cid] = durations.get(cid, 0) + max(
                0, ev.get("end_ms", 0) - ev.get("start_ms", 0)
            )
            counts[cid] = counts.get(cid, 0) + 1
        return durations, counts

    def _reassign_events_from(targets: set[int], reason: str) -> int:
        if not targets:
            return 0
        keepers = [
            e
            for e in events
            if (e.get("speakers") or [{}])[0].get("cluster_id", 0) > 0
            and (e.get("speakers") or [{}])[0].get("cluster_id") not in targets
        ]
        if not keepers:
            return 0
        for ev in events:
            sp = ev.get("speakers") or []
            cid = sp[0].get("cluster_id") if sp else 0
            if not cid or cid not in targets:
                continue
            mid_ms = (ev.get("start_ms", 0) + ev.get("end_ms", ev.get("start_ms", 0))) / 2
            nearest = min(
                keepers,
                key=lambda r: abs(((r.get("start_ms", 0) + r.get("end_ms", 0)) / 2) - mid_ms),
            )
            _attribute_to(ev, nearest)
        logger.info(
            "%s: reassigned events from %d cluster(s) %s to nearest real speakers",
            reason,
            len(targets),
            sorted(targets),
        )
        return len(targets)

    # Pass 1: size-based dissolve
    durations, counts = _cluster_sizes()
    ghost_ids = {
        cid
        for cid in durations
        if durations[cid] < _DISSOLVE_MAX_DURATION_MS
        and counts.get(cid, 0) <= _DISSOLVE_MAX_SEGMENTS
    }
    _reassign_events_from(ghost_ids, "Ghost dissolve")

    # Pass 2: expected-count collapse
    if expected_speakers is not None and expected_speakers > 0:
        durations, _ = _cluster_sizes()
        live = sorted(durations.keys(), key=lambda c: durations[c], reverse=True)
        if len(live) > expected_speakers:
            collapse_ids = set(live[expected_speakers:])
            _reassign_events_from(
                collapse_ids,
                f"Expected-count collapse (target={expected_speakers})",
            )

    # Build speaker data from corrected events.
    #
    # The cluster's participant-list display_name is chosen by MAJORITY
    # VOTE over the identities present on its events — not just the
    # first one encountered. This matters when a cluster carries a mix
    # of identities (e.g. pyannote clustered Joel + Nikul + Danny
    # together on f38d5807): the sidebar shows the dominant voice name,
    # while individual transcript rows keep their own `identity` field
    # set by upstream corrections.
    from collections import Counter as _Counter

    speaker_stats: dict[int, dict] = {}
    speaker_lanes: dict[str, list] = {}
    cluster_identity_votes: dict[int, _Counter] = {}

    for e in events:
        sp = e.get("speakers", [])
        cluster_id = 0 if not sp else sp[0].get("cluster_id", 0)

        # Skip the orphan sentinel entirely. Reassignment above already
        # tried to attribute every event to a real speaker — anything
        # still tagged cluster_id=0 is unattributable (e.g. a meeting
        # with literally zero recognized voices). Letting cluster_id=0
        # fall through here would (a) eat seq_index=1 from the real
        # speakers, and (b) emit a phantom "Speaker 1" lane in the UI.
        if cluster_id == 0:
            continue

        cid_str = str(cluster_id)
        start_ms = e.get("start_ms", 0)
        end_ms = e.get("end_ms", start_ms + 1500)

        # Build lanes
        if cid_str not in speaker_lanes:
            speaker_lanes[cid_str] = []
        speaker_lanes[cid_str].append(
            {
                "start_ms": start_ms,
                "end_ms": end_ms,
                "segment_id": e.get("segment_id", ""),
            }
        )

        # Build stats — prefer corrected identity over raw cluster name.
        # Store sequential "Speaker N" based on first-seen order (not raw
        # cluster_id which may be 100, 101 etc. from the pseudo-cluster
        # fallback). This keeps the UI consistent with the client-side
        # registry that uses sequential numbering.
        # Vote for this cluster's display name using this event's
        # identity (if the user explicitly labeled it). Generic
        # "Speaker N" labels don't vote — they'd dilute real names.
        if sp:
            candidate = sp[0].get("identity") or sp[0].get("display_name")
            if candidate:
                import re as _re

                if not _re.match(r"^Speaker\s+\d+$", candidate.strip()):
                    cluster_identity_votes.setdefault(cluster_id, _Counter())[candidate] += 1

        if cluster_id not in speaker_stats:
            seq_index = len(speaker_stats) + 1
            speaker_stats[cluster_id] = {
                "display_name": f"Speaker {seq_index}",  # placeholder, resolved below
                "cluster_id": cluster_id,
                "seq_index": seq_index,
                "segment_count": 0,
                "total_speaking_ms": 0,
                "first_seen_ms": start_ms,
                "last_seen_ms": end_ms,
            }
        speaker_stats[cluster_id]["segment_count"] += 1
        speaker_stats[cluster_id]["total_speaking_ms"] += max(0, end_ms - start_ms)
        speaker_stats[cluster_id]["last_seen_ms"] = max(
            speaker_stats[cluster_id]["last_seen_ms"], end_ms
        )

    # After counting, assign each cluster its dominant identity.
    for cid, votes in cluster_identity_votes.items():
        if cid in speaker_stats and votes:
            dominant_name = votes.most_common(1)[0][0]
            speaker_stats[cid]["display_name"] = dominant_name

    # Save.
    # Important: remap the raw cluster_ids we see in-flight (which can
    # number in the dozens after a long meeting with lots of diarize
    # fragmentation) to the stable "Speaker N" seq_index we just assigned.
    # Post-finalize consumers (timeline playback, detected-speakers list,
    # speaker_lanes) ALL key on this seq_index so the participant list and
    # the raw transcript view agree on "how many speakers" and "which
    # speaker". Raw cluster_id stays on each entry for traceability.
    # Drop cluster_id=0 (the orphan sentinel) — after the reassignment
    # pass above it should only hold events we truly couldn't attribute,
    # and surfacing "Unknown" is worse than silently omitting a few
    # sub-second fragments from the participant list.
    speakers_list = sorted(
        [s for s in speaker_stats.values() if s["cluster_id"] > 0],
        key=lambda s: -s["segment_count"],
    )
    # Build the cluster_id → seq_index mapping now that every speaker_stats
    # entry has its seq_index populated.
    cluster_to_seq: dict[int, int] = {s["cluster_id"]: s["seq_index"] for s in speakers_list}

    # Rewrite the lanes dict with seq_index as the key (stringified for JSON).
    # Skip any cluster_id that didn't survive into the speakers list
    # (orphans, stale ids) so the UI never gets a phantom lane.
    remapped_lanes: dict[str, list] = {}
    for cid_str, entries in speaker_lanes.items():
        try:
            cid = int(cid_str)
        except ValueError:
            continue
        if cid <= 0:
            continue
        seq = cluster_to_seq.get(cid)
        if seq is None:
            continue
        remapped_lanes.setdefault(str(seq), []).extend(entries)

    speakers_path = meeting_dir / "detected_speakers.json"
    speakers_path.write_text(_json.dumps(speakers_list, indent=2))

    lanes_path = meeting_dir / "speaker_lanes.json"
    lanes_path.write_text(_json.dumps(remapped_lanes, indent=2))

    # Rewrite journal.jsonl so post-finalize views of the raw transcript
    # show "Speaker N" consistent with the participants list AND so
    # orphan-reassigned events carry their newly-attributed speaker
    # (otherwise the first minute of a meeting where diarization hadn't
    # caught up renders with no speaker highlighting).
    #
    # We rewrite from the in-memory ``events`` list — which already has
    # corrections applied AND orphan reassignment applied — rather than
    # re-reading the journal from disk, so all three downstream files
    # (state.detected_speakers, speaker_lanes, journal) agree second-by-second.
    journal_path = meeting_dir / "journal.jsonl"
    if journal_path.exists():
        # Rewrite the journal from the IN-MEMORY best/events list,
        # NOT by re-iterating the on-disk journal. The on-disk journal
        # is append-only — every re-finalize call stacks another
        # revision on top, and iterating it produces duplicates. We
        # emit exactly one line per segment_id (the highest-revision
        # event with the fully-resolved speakers) + all
        # speaker_correction sentinel rows verbatim.
        #
        # This is the fix for the 2026-04-14 journal-doubling bug:
        # running /api/meetings/<id>/finalize twice used to produce
        # 2× → 4× → 8× lines because each run appended rev=N+1 and
        # then the rewrite loop processed EVERY on-disk line again.
        resolved_events: dict[str, dict] = {}  # sid → full event dict
        for e in events:
            sid = e.get("segment_id")
            if not sid:
                continue
            sp_list = e.get("speakers") or []
            new_sp_list: list[dict] = []
            for sp_entry in sp_list:
                raw = sp_entry.get("cluster_id")
                if raw is None or raw <= 0:
                    continue
                seq = cluster_to_seq.get(raw)
                if seq is None:
                    continue
                # Stamp the seq_index over cluster_id so the UI's join
                # against state.detected_speakers.json works. Keep the raw
                # cluster_id under _raw_cluster_id for diarize debugging.
                merged = dict(sp_entry)
                merged["_raw_cluster_id"] = raw
                merged["cluster_id"] = seq
                new_sp_list.append(merged)
            # Write back the resolved speakers onto the event dict,
            # then record the full event for journal output.
            e_out = dict(e)
            e_out["speakers"] = new_sp_list
            resolved_events[sid] = e_out

        # Collect speaker_correction rows from the current on-disk
        # journal so they survive the rewrite. These are sentinels
        # the UI writes when a user renames a speaker.
        correction_lines: list[str] = []
        for line in journal_path.open("r", encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                evt = _json.loads(line)
            except Exception:
                correction_lines.append(line)  # preserve unparseable
                continue
            if evt.get("type") == "speaker_correction":
                correction_lines.append(line)

        # Build the new journal: one final event per segment_id
        # (sorted by start_ms for reproducibility), then the
        # speaker_corrections.
        out_lines: list[str] = []
        for e in sorted(resolved_events.values(), key=lambda x: x.get("start_ms", 0)):
            out_lines.append(_json.dumps(e, ensure_ascii=False))
        out_lines.extend(correction_lines)
        journal_path.write_text("\n".join(out_lines) + "\n")

    logger.info(
        "Generated speaker data: %d speakers, %d lane entries (journal remapped)",
        len(speakers_list),
        sum(len(v) for v in remapped_lanes.values()),
    )


def _generate_timeline(meeting_id: str, meeting_dir=None) -> None:
    """Generate timeline.json from journal for podcast player.

    ``meeting_dir`` is optional — callers that already have the path
    (e.g. reprocess.py which runs before the server's storage global
    is initialized) can pass it directly to avoid touching the module
    global.
    """
    import json as _json

    if meeting_dir is None:
        meeting_dir = state.storage._meeting_dir(meeting_id)
    journal_path = meeting_dir / "journal.jsonl"
    timeline_path = meeting_dir / "timeline.json"

    # Deduplicate by segment_id, keeping highest revision
    best: dict[str, dict] = {}
    if journal_path.exists():
        for line in journal_path.open():
            line = line.strip()
            if not line:
                continue
            try:
                e = _json.loads(line)
                if e.get("text") and e.get("is_final"):
                    sid = e.get("segment_id")
                    rev = e.get("revision", 0)
                    if sid not in best or rev > best[sid].get("revision", 0):
                        best[sid] = e
            except Exception:
                continue

    # Build segments sorted by start_ms
    segments = sorted(
        [
            {
                "segment_id": e.get("segment_id"),
                "start_ms": e.get("start_ms", 0),
                "end_ms": e.get("end_ms", 0),
                "language": e.get("language", "unknown"),
                "speaker_id": (e.get("speakers") or [{}])[0].get("cluster_id")
                if e.get("speakers")
                else None,
                "text": e.get("text", "")[:100],
            }
            for e in best.values()
        ],
        key=lambda s: s["start_ms"],
    )

    # Audio duration via storage if available; fall back to reading the
    # recording.pcm directly if the module-global storage hasn't been
    # initialized (reprocess.py invocation path — no lifespan yet).
    audio_ms = 0
    try:
        audio_ms = state.storage.audio_duration_ms(meeting_id)
    except Exception:
        pcm = meeting_dir / "audio" / "recording.pcm"
        if pcm.exists():
            audio_ms = int(pcm.stat().st_size // 2 / 16000 * 1000)
    # Timeline duration = audio file duration. This is the canonical
    # length users see in the scrubber / speaker lanes. If events only
    # cover part of the audio (e.g. a crashed session or a meeting that
    # was never fully ASR'd), the proper fix is to re-run full-reprocess
    # so the journal gets ASR events for the whole recording — NOT to
    # shrink the timeline and hide untranscribed audio.
    duration_ms = audio_ms if audio_ms > 0 else max((s["end_ms"] for s in segments), default=0)
    for s in segments:
        if s["end_ms"] > duration_ms:
            s["end_ms"] = duration_ms
        if s["start_ms"] > duration_ms:
            s["start_ms"] = duration_ms
    timeline: dict = {"duration_ms": duration_ms, "segments": segments}

    # Optional: include the frame-level exclusive (community-1) speaker
    # timeline when the sidecar artifact exists.  Additive — older
    # consumers walk the ``segments`` array; new consumers can render a
    # cleaner single-speaker-per-frame timeline.  Empty / missing
    # sidecar is silently skipped.  See plans/2026-Q3-followups.md C2.
    exclusive_path = meeting_dir / "speaker_lanes_exclusive.json"
    if exclusive_path.exists():
        try:
            exclusive_data = _json.loads(exclusive_path.read_text())
            ex_segments = exclusive_data.get("segments") or []
        except Exception:
            ex_segments = []
        if ex_segments:
            # Clamp to duration_ms, same as the main segments.
            for s in ex_segments:
                if s.get("end_ms", 0) > duration_ms:
                    s["end_ms"] = duration_ms
                if s.get("start_ms", 0) > duration_ms:
                    s["start_ms"] = duration_ms
            timeline["exclusive_segments"] = ex_segments

    timeline_path.write_text(_json.dumps(timeline, indent=2))
    logger.info(
        "Generated timeline.json: %d segments%s, %dms",
        len(segments),
        f" + {len(timeline.get('exclusive_segments') or [])} exclusive"
        if "exclusive_segments" in timeline
        else "",
        duration_ms,
    )


# Audio-out wire protocol constants live in
# ``meeting_scribe.ws.audio_output``. ``_buffer_pending_audio`` is
# the only consumer here.
