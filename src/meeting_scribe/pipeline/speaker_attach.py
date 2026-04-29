"""Overlap-aware speaker attribution shared by reprocess + finalize.

Given a list of ASR-final events plus the standard + exclusive pyannote
segmentations, mutate each event's ``speakers`` list in place.

**Source of truth for primary speaker**: the ``exclusive_segments``
array (community-1 single-speaker-per-frame timeline, ≥ 1 cluster id
assigned to every speaking frame).  Track C of the 2026-Q2 bench
measured this resolves 100 % of the previously-overlapping seconds
to a single-speaker assignment on a real production meeting.

**Cross-talk detection**: the standard ``segments`` array.  Whenever
two distinct cluster ids have non-trivial standard overlap on the
same event window, the secondary is recorded with
``source="diarization_overlap"`` and ``event["overlapping_speakers"] = True``.

Both the batch ``reprocess_meeting`` flow and the live "stop meeting"
path in ``routes.meeting_lifecycle`` use this — the catchup loop in
``runtime.meeting_loops`` runs the same overlap math inline, by design,
because it operates on a single event at a time.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def _overlap_ms(a_start: int, a_end: int, b_start: int, b_end: int) -> int:
    return max(0, min(a_end, b_end) - max(a_start, b_start))


def _attach_speakers_to_events(
    events: list[dict],
    diarize_segments: list[dict],
    exclusive_segments: list[dict] | None = None,
) -> None:
    """Attach diarization clusters to ASR events, preserving overlap.

    Primary speaker per event is decided by the **exclusive_segments**
    timeline (community-1's single-speaker-per-frame output): the
    cluster with the most exclusive duration during the event window.
    Secondary speakers are added from the **standard segments** when
    their overlap meets both:
      * secondary speaker's overlap ≥ 30 % of the event's duration
      * secondary speaker's overlap ≥ 50 % of the primary's overlap

    These thresholds keep a 200 ms "mhm" interjection from being
    flagged as co-speech.

    ``exclusive_segments`` is required in the production pipeline
    (community-1 always emits it).  When it is empty (e.g., a degraded
    chunk), this function falls back to selecting the primary from the
    standard segments — which matches the pre-community-1 behaviour.
    """
    if not diarize_segments and not exclusive_segments:
        # No diarization at all — leave events with empty speakers
        # (fallback to time-proximity clustering done elsewhere).
        return

    exclusive_segments = exclusive_segments or []

    _MIN_SECONDARY_FRAC_OF_EVENT = 0.30
    _MIN_SECONDARY_FRAC_OF_PRIMARY = 0.50

    for event in events:
        s, e = event.get("start_ms", 0), event.get("end_ms", 0)
        ev_dur = max(1, e - s)

        # Per-cluster overlap on the standard segmentation (used for
        # cross-talk / secondary-speaker detection).
        overlap_by_cluster: dict[int, float] = {}
        conf_by_cluster: dict[int, float] = {}
        for seg in diarize_segments:
            overlap = _overlap_ms(s, e, seg["start_ms"], seg["end_ms"])
            if overlap <= 0:
                continue
            cid = seg["cluster_id"]
            overlap_by_cluster[cid] = overlap_by_cluster.get(cid, 0) + overlap
            # Keep the best confidence if multiple sub-segments exist.
            conf_by_cluster[cid] = max(conf_by_cluster.get(cid, 0.0), seg.get("confidence", 1.0))

        # Per-cluster duration on the exclusive segmentation (used for
        # primary-speaker assignment).  Each frame is owned by exactly
        # one speaker, so summed durations are unambiguous.
        exclusive_by_cluster: dict[int, float] = {}
        for seg in exclusive_segments:
            overlap = _overlap_ms(s, e, seg["start_ms"], seg["end_ms"])
            if overlap <= 0:
                continue
            cid = seg["cluster_id"]
            exclusive_by_cluster[cid] = exclusive_by_cluster.get(cid, 0) + overlap

        if not overlap_by_cluster and not exclusive_by_cluster:
            continue

        if exclusive_by_cluster:
            primary_cid = max(exclusive_by_cluster.items(), key=lambda kv: kv[1])[0]
            primary_overlap = overlap_by_cluster.get(primary_cid, exclusive_by_cluster[primary_cid])
            primary_source = "diarization_exclusive"
        else:
            ranked = sorted(overlap_by_cluster.items(), key=lambda kv: -kv[1])
            primary_cid, primary_overlap = ranked[0]
            primary_source = "diarization"

        speakers_list = [
            {
                "cluster_id": primary_cid,
                "identity": None,
                "identity_confidence": conf_by_cluster.get(primary_cid, 1.0),
                "source": primary_source,
                "display_name": None,
            }
        ]

        # Cross-talk detection: any other cluster with significant
        # *standard* overlap (we deliberately use standard here — the
        # whole reason it can have multiple speakers in the same window
        # is to surface co-speech; exclusive output collapses to one).
        for cid, ov in sorted(overlap_by_cluster.items(), key=lambda kv: -kv[1]):
            if cid == primary_cid:
                continue
            if ov / ev_dur < _MIN_SECONDARY_FRAC_OF_EVENT:
                continue
            if primary_overlap > 0 and ov / primary_overlap < _MIN_SECONDARY_FRAC_OF_PRIMARY:
                continue
            speakers_list.append(
                {
                    "cluster_id": cid,
                    "identity": None,
                    "identity_confidence": conf_by_cluster.get(cid, 1.0),
                    "source": "diarization_overlap",
                    "display_name": None,
                }
            )
        event["speakers"] = speakers_list
        if len(speakers_list) > 1:
            event["overlapping_speakers"] = True

    # Second pass: minority-speaker recovery.  Some speakers (especially
    # the ones who only said "yes" or "mmm-hmm") have segments that never
    # WIN the primary overlap on any ASR event — so they'd disappear from
    # detected_speakers even though pyannote correctly identified them
    # as distinct voices.  Catch this by cross-referencing the union of
    # clusters seen across BOTH segmentations vs the set of clusters that
    # ended up as primary on at least one event.  Any cluster that got
    # attributed to NO events gets rescued here: find the event with the
    # highest total overlap on that cluster (preferring exclusive — it's
    # the one we use for primary assignment) and promote the cluster to
    # primary on THAT event (the previous primary becomes a secondary).
    diarize_clusters = {
        seg["cluster_id"]
        for src in (diarize_segments, exclusive_segments)
        for seg in src
    }
    primary_clusters_seen: set[int] = set()
    for ev in events:
        sp = ev.get("speakers") or []
        if sp:
            primary_clusters_seen.add(sp[0].get("cluster_id"))
    missing = diarize_clusters - primary_clusters_seen

    for miss_cid in missing:
        # Find the event with the most total overlap on this cluster
        # (preferring exclusive_segments for the search since that's
        # the source of truth; fall back to standard if the cluster
        # only ever appeared in standard).
        rescue_source = exclusive_segments if any(
            s["cluster_id"] == miss_cid for s in exclusive_segments
        ) else diarize_segments
        best_ev = None
        best_ov = 0
        for ev in events:
            s, e = ev.get("start_ms", 0), ev.get("end_ms", 0)
            ov = 0
            for seg in rescue_source:
                if seg["cluster_id"] != miss_cid:
                    continue
                ov += _overlap_ms(s, e, seg["start_ms"], seg["end_ms"])
            if ov > best_ov:
                best_ov = ov
                best_ev = ev
        if best_ev is None or best_ov <= 0:
            cluster_ranges = [
                (seg["start_ms"], seg["end_ms"])
                for seg in rescue_source
                if seg["cluster_id"] == miss_cid
            ]
            total_s = sum(e - s for s, e in cluster_ranges) / 1000 if cluster_ranges else 0
            logger.warning(
                "diarization_minority_rescue: cluster %s was found by "
                "diarize (%d segments, %.1fs total) but has ZERO overlap "
                "with any ASR event — this speaker will be invisible "
                "unless you run /reprocess to re-chunk ASR.",
                miss_cid,
                len(cluster_ranges),
                total_s,
            )
            continue
        existing = best_ev.get("speakers") or []
        existing_primary = existing[0] if existing else None
        new_sp = [
            {
                "cluster_id": miss_cid,
                "identity": None,
                "identity_confidence": 1.0,
                "source": "diarization_minority_rescue",
                "display_name": None,
            }
        ]
        if existing_primary is not None:
            existing_primary = dict(existing_primary)
            existing_primary["source"] = "diarization_demoted"
            new_sp.append(existing_primary)
        for sp_entry in existing[1:]:
            if sp_entry.get("cluster_id") not in (miss_cid,):
                new_sp.append(sp_entry)
        best_ev["speakers"] = new_sp
        if len(new_sp) > 1:
            best_ev["overlapping_speakers"] = True
        logger.info(
            "diarization_minority_rescue: cluster %s had no primary — "
            "promoted on event at %d-%dms (overlap=%dms)",
            miss_cid,
            best_ev.get("start_ms", 0),
            best_ev.get("end_ms", 0),
            best_ov,
        )
