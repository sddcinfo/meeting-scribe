"""Re-process a meeting's raw PCM audio from scratch.

Walks the PCM file in 4s chunks, transcribes with vLLM Qwen3-ASR,
extracts ECAPA-TDNN speaker embeddings, clusters speakers, and writes
a new journal with PCM-aligned timestamps.

Usage:
    PYTHONPATH=src .venv/bin/python3 scripts/reprocess_meeting.py \
        --meeting-id f38d5807-bbdf-4c5c-96fb-cb8267e55ed0 \
        --num-speakers 6
"""

from __future__ import annotations

import base64
import io
import json
import sys
import uuid
from pathlib import Path

import click
import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

SAMPLE_RATE = 16000
CHUNK_SECONDS = 4.0
CHUNK_SAMPLES = int(CHUNK_SECONDS * SAMPLE_RATE)
CHUNK_BYTES = CHUNK_SAMPLES * 2
MIN_RMS = 0.005
ASR_URL = "http://localhost:8003"


def transcribe_chunk(audio: np.ndarray) -> tuple[str, str]:
    """Transcribe audio via vLLM Qwen3-ASR. Returns (text, language)."""
    import urllib.request

    buf = io.BytesIO()
    sf.write(buf, audio, SAMPLE_RATE, format="WAV")
    wav_b64 = base64.b64encode(buf.getvalue()).decode()

    req = urllib.request.Request(
        f"{ASR_URL}/v1/chat/completions",
        data=json.dumps(
            {
                "model": "Qwen/Qwen3-ASR-1.7B",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_audio",
                                "input_audio": {"data": wav_b64, "format": "wav"},
                            },
                            {"type": "text", "text": "<|startoftranscript|>"},
                        ],
                    }
                ],
                "max_tokens": 200,
                "temperature": 0.0,
            }
        ).encode(),
        headers={"Content-Type": "application/json"},
    )
    result = json.loads(urllib.request.urlopen(req, timeout=60).read())
    raw = result["choices"][0]["message"]["content"]

    if "<asr_text>" in raw:
        prefix, _, text = raw.partition("<asr_text>")
        text = text.strip()
        lang_raw = prefix.replace("language", "").strip()
        lang = "ja" if lang_raw.lower() in ("japanese", "ja") else "en"
    else:
        text = raw.strip()
        lang = "en"

    return text, lang


@click.command()
@click.option("--meeting-id", required=True)
@click.option("--meetings-dir", default=None)
@click.option("--num-speakers", default=6)
def reprocess(meeting_id: str, meetings_dir: str | None, num_speakers: int) -> None:
    """Re-process meeting audio: ASR + speaker clustering from scratch."""
    storage_dir = Path(meetings_dir) if meetings_dir else Path(__file__).parent.parent / "meetings"
    meeting_dir = storage_dir / meeting_id
    pcm_path = meeting_dir / "audio" / "recording.pcm"

    if not pcm_path.exists():
        click.secho(f"No audio at {pcm_path}", fg="red")
        sys.exit(1)

    pcm_size = pcm_path.stat().st_size
    total_chunks = pcm_size // CHUNK_BYTES
    total_seconds = pcm_size / (SAMPLE_RATE * 2)
    click.echo(f"PCM: {total_seconds / 60:.1f} min, {total_chunks} chunks")

    # Load speaker embedding model
    click.echo("Loading ECAPA-TDNN...")
    from speechbrain.inference.speaker import EncoderClassifier

    device = "cuda" if torch.cuda.is_available() else "cpu"
    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": device},
    )

    # Process every chunk
    segments = []
    embeddings = []
    embedding_indices = []

    click.echo(f"Processing {total_chunks} chunks...")
    with open(pcm_path, "rb") as f:
        for chunk_idx in range(total_chunks):
            data = f.read(CHUNK_BYTES)
            if len(data) < CHUNK_BYTES:
                break

            audio = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            rms = float(np.sqrt(np.mean(audio**2)))

            start_ms = int(chunk_idx * CHUNK_SECONDS * 1000)
            end_ms = int((chunk_idx + 1) * CHUNK_SECONDS * 1000)

            if rms < MIN_RMS:
                continue

            # Transcribe
            try:
                text, lang = transcribe_chunk(audio)
            except Exception as e:
                click.echo(f"  ASR error at {start_ms / 1000:.0f}s: {e}")
                continue

            if not text.strip():
                continue

            seg_id = str(uuid.uuid4())
            seg = {
                "segment_id": seg_id,
                "revision": 1,
                "is_final": True,
                "start_ms": start_ms,
                "end_ms": end_ms,
                "language": lang,
                "text": text,
                "speakers": [],
                "translation": None,
            }
            segments.append(seg)

            # Extract speaker embedding
            audio_tensor = torch.from_numpy(audio).unsqueeze(0).to(device)
            with torch.no_grad():
                emb = classifier.encode_batch(audio_tensor)
            embeddings.append(emb.squeeze().cpu().numpy())
            embedding_indices.append(len(segments) - 1)

            if (chunk_idx + 1) % 50 == 0:
                click.echo(
                    f"  {chunk_idx + 1}/{total_chunks} "
                    f"({len(segments)} segments, {start_ms / 1000 / 60:.1f} min)"
                )

    click.echo(f"\nTranscribed {len(segments)} segments, {len(embeddings)} embeddings")

    # Cluster speakers
    if len(embeddings) >= 2:
        click.echo(f"Clustering into {num_speakers} speakers...")
        from scipy.cluster.hierarchy import fcluster, linkage
        from scipy.spatial.distance import pdist

        emb_matrix = np.array(embeddings)
        norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1
        emb_matrix = emb_matrix / norms

        distances = pdist(emb_matrix, metric="cosine")
        Z = linkage(distances, method="average")
        labels = fcluster(Z, t=num_speakers, criterion="maxclust")

        # Assign speaker labels to segments
        for emb_idx, seg_idx in enumerate(embedding_indices):
            speaker_id = int(labels[emb_idx])
            segments[seg_idx]["speakers"] = [
                {
                    "cluster_id": speaker_id,
                    "identity": None,
                    "identity_confidence": 0.0,
                    "source": "embedding_cluster",
                }
            ]

        # Speaker stats
        from collections import Counter

        speaker_counts = Counter(int(labels[i]) for i in range(len(labels)))
        click.echo("Speaker distribution:")
        for spk_id, count in speaker_counts.most_common():
            mins = count * CHUNK_SECONDS / 60
            click.echo(f"  Speaker {spk_id}: {count} segments ({mins:.1f} min)")

        # Save embeddings
        np.savez_compressed(
            meeting_dir / "speaker_embeddings.npz",
            embeddings=emb_matrix,
            labels=np.array(labels),
            valid_indices=np.array(embedding_indices),
        )

        # Update detected speakers
        detected = []
        for spk_id, count in speaker_counts.most_common():
            detected.append(
                {
                    "display_name": f"Speaker {spk_id}",
                    "cluster_id": spk_id,
                    "segment_count": count,
                    "total_speaking_ms": int(count * CHUNK_SECONDS * 1000),
                    "first_seen_ms": 0,
                    "last_seen_ms": segments[-1]["end_ms"],
                }
            )
        with open(meeting_dir / "detected_speakers.json", "w") as f2:
            json.dump(detected, f2, indent=2)

        # Update room layout
        seats = []
        for i, sp in enumerate(detected):
            x = 20 + (30 * (i % 3))
            y = 28 if i < 3 else 72
            seats.append(
                {
                    "seat_id": str(uuid.uuid4()),
                    "x": round(x, 1),
                    "y": float(y),
                    "enrollment_id": None,
                    "speaker_name": sp["display_name"],
                }
            )
        room = {
            "preset": "rectangle",
            "tables": [
                {
                    "table_id": str(uuid.uuid4()),
                    "x": 50.0,
                    "y": 50.0,
                    "width": 44.0,
                    "height": 22.0,
                    "border_radius": 3.0,
                    "label": "",
                }
            ],
            "seats": seats,
        }
        with open(meeting_dir / "room.json", "w") as f2:
            json.dump(room, f2, indent=2)

    # Write new journal (PCM-aligned timestamps, no offset needed)
    with open(meeting_dir / "journal.jsonl", "w") as f2:
        for seg in segments:
            f2.write(json.dumps(seg) + "\n")

    # Write timeline
    timeline = [
        {
            "segment_id": s["segment_id"],
            "start_ms": s["start_ms"],
            "end_ms": s["end_ms"],
            "text": s["text"],
            "language": s["language"],
            "speakers": s.get("speakers", []),
        }
        for s in segments
    ]
    with open(meeting_dir / "timeline.json", "w") as f2:
        json.dump(timeline, f2)

    # Update meta: no offset needed since timestamps ARE PCM positions
    with open(meeting_dir / "meta.json") as f2:
        meta = json.load(f2)
    meta["audio_offset_ms"] = 0
    meta.pop("audio_scale_factor", None)
    with open(meeting_dir / "meta.json", "w") as f2:
        json.dump(meta, f2, indent=2)

    click.secho(
        f"\nDone! {len(segments)} segments, {num_speakers} speakers, audio_offset=0 (PCM-aligned)",
        fg="green",
    )


if __name__ == "__main__":
    reprocess()
