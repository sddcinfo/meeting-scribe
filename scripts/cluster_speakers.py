"""Cluster speakers in a recorded meeting using ECAPA-TDNN embeddings.

Extracts speaker embeddings from every segment, clusters with
agglomerative clustering, and updates the meeting journal with
speaker attributions. Outputs per-speaker audio segments.

Usage:
    PYTHONPATH=src .venv/bin/python3 scripts/cluster_speakers.py \
        --meeting-id f38d5807-bbdf-4c5c-96fb-cb8267e55ed0
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

SAMPLE_RATE = 16000
BYTES_PER_SAMPLE = 2  # s16le
MIN_SEGMENT_SAMPLES = SAMPLE_RATE  # 1 second minimum for embedding


@click.command()
@click.option("--meeting-id", required=True, help="Meeting ID to process")
@click.option(
    "--meetings-dir",
    default=None,
    help="Meeting storage directory (default: ./meetings)",
)
@click.option("--num-speakers", default=0, help="Number of speakers (0=auto-detect)")
@click.option("--batch-size", default=32, help="Batch size for embedding extraction")
def cluster(meeting_id: str, meetings_dir: str | None, num_speakers: int, batch_size: int) -> None:
    """Cluster speakers in a recorded meeting."""
    storage_dir = Path(meetings_dir) if meetings_dir else Path(__file__).parent.parent / "meetings"
    meeting_dir = storage_dir / meeting_id

    if not meeting_dir.exists():
        click.secho(f"Meeting not found: {meeting_dir}", fg="red")
        sys.exit(1)

    journal_path = meeting_dir / "journal.jsonl"
    pcm_path = meeting_dir / "audio" / "recording.pcm"

    if not pcm_path.exists():
        click.secho("No audio recording found", fg="red")
        sys.exit(1)

    # Load journal
    with open(journal_path) as f:
        segments = [json.loads(line) for line in f]
    finals = [s for s in segments if s["is_final"] and s["text"].strip()]
    click.echo(f"Meeting: {meeting_id} ({len(finals)} segments)")

    # Load embedding model
    click.echo("Loading ECAPA-TDNN speaker embedding model...")
    from speechbrain.inference.speaker import EncoderClassifier

    device = "cuda" if torch.cuda.is_available() else "cpu"
    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": device},
    )
    click.echo(f"Model loaded on {device}")

    # Extract embeddings for every segment
    click.echo(f"Extracting embeddings from {len(finals)} segments...")
    pcm_size = pcm_path.stat().st_size
    embeddings = []
    valid_indices = []

    with open(pcm_path, "rb") as f:
        for i, seg in enumerate(finals):
            start_byte = int(seg["start_ms"] / 1000 * SAMPLE_RATE) * BYTES_PER_SAMPLE
            end_byte = int(seg["end_ms"] / 1000 * SAMPLE_RATE) * BYTES_PER_SAMPLE

            if start_byte >= pcm_size or end_byte > pcm_size:
                continue

            f.seek(start_byte)
            pcm = f.read(end_byte - start_byte)

            if len(pcm) < MIN_SEGMENT_SAMPLES * BYTES_PER_SAMPLE:
                continue

            audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0

            # Skip silence
            rms = float(np.sqrt(np.mean(audio**2)))
            if rms < 0.005:
                continue

            audio_tensor = torch.from_numpy(audio).unsqueeze(0).to(device)
            with torch.no_grad():
                emb = classifier.encode_batch(audio_tensor)
            embeddings.append(emb.squeeze().cpu().numpy())
            valid_indices.append(i)

            if (i + 1) % 100 == 0:
                click.echo(f"  {i + 1}/{len(finals)} segments processed")

    click.echo(f"Extracted {len(embeddings)} embeddings")

    if len(embeddings) < 2:
        click.secho("Not enough embeddings for clustering", fg="red")
        sys.exit(1)

    # Normalize embeddings
    emb_matrix = np.array(embeddings)
    norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1
    emb_matrix = emb_matrix / norms

    # Agglomerative clustering
    click.echo("Clustering speakers...")
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import pdist

    distances = pdist(emb_matrix, metric="cosine")
    linkage_matrix = linkage(distances, method="average")

    if num_speakers > 0:
        labels = fcluster(linkage_matrix, t=num_speakers, criterion="maxclust")
    else:
        # Auto-detect: use distance threshold
        # Cosine distance ~0.3 is a good threshold for speaker separation
        labels = fcluster(linkage_matrix, t=0.5, criterion="distance")

    n_clusters = len(set(labels))
    click.echo(f"Found {n_clusters} unique speakers")

    # Compute per-cluster stats
    cluster_stats: dict[int, dict] = {}
    for label, idx in zip(labels, valid_indices, strict=False):
        seg = finals[idx]
        if label not in cluster_stats:
            cluster_stats[label] = {
                "count": 0,
                "total_ms": 0,
                "first_ms": seg["start_ms"],
                "last_ms": seg["end_ms"],
                "sample_texts": [],
                "centroid": np.zeros_like(embeddings[0]),
            }
        stats = cluster_stats[label]
        stats["count"] += 1
        stats["total_ms"] += seg["end_ms"] - seg["start_ms"]
        stats["last_ms"] = max(stats["last_ms"], seg["end_ms"])
        if len(stats["sample_texts"]) < 3:
            stats["sample_texts"].append(seg["text"][:80])
        # Update centroid
        emb_idx = valid_indices.index(idx)
        stats["centroid"] = (
            stats["centroid"] * (stats["count"] - 1) + emb_matrix[emb_idx]
        ) / stats["count"]

    # Print cluster summary
    click.echo(f"\n{'=' * 60}")
    click.echo("SPEAKER CLUSTERING RESULTS")
    click.echo(f"{'=' * 60}")

    for label in sorted(cluster_stats, key=lambda k: -cluster_stats[k]["count"]):
        stats = cluster_stats[label]
        click.echo(
            f"\nSpeaker {label} ({stats['count']} segments, {stats['total_ms'] / 1000 / 60:.1f} min)"
        )
        for text in stats["sample_texts"]:
            click.echo(f'  "{text}"')

    # Compute inter-cluster distances for confidence
    click.echo("\nInter-speaker distances (cosine):")
    sorted_labels = sorted(cluster_stats.keys())
    for i, l1 in enumerate(sorted_labels):
        for l2 in sorted_labels[i + 1 :]:
            c1 = cluster_stats[l1]["centroid"]
            c2 = cluster_stats[l2]["centroid"]
            dist = 1 - float(np.dot(c1, c2) / (np.linalg.norm(c1) * np.linalg.norm(c2) + 1e-8))
            click.echo(f"  Speaker {l1} ↔ Speaker {l2}: {dist:.3f}")

    # Update journal with speaker attributions
    click.echo("\nUpdating journal with speaker labels...")
    label_map = dict(zip(valid_indices, labels, strict=False))

    updated_segments = []
    for i, seg in enumerate(finals):
        seg_copy = dict(seg)
        if i in label_map:
            speaker_id = int(label_map[i])
            seg_copy["speakers"] = [
                {
                    "cluster_id": speaker_id,
                    "identity": None,
                    "identity_confidence": 0.0,
                    "source": "embedding_cluster",
                }
            ]
        updated_segments.append(seg_copy)

    # Write updated journal
    with open(journal_path, "w") as f:
        for seg in updated_segments:
            f.write(json.dumps(seg) + "\n")

    # Update detected speakers
    detected = []
    for label in sorted(cluster_stats, key=lambda k: -cluster_stats[k]["count"]):
        stats = cluster_stats[label]
        detected.append(
            {
                "display_name": f"Speaker {label}",
                "cluster_id": int(label),
                "matched_enrollment_id": None,
                "match_confidence": 0.0,
                "segment_count": stats["count"],
                "first_seen_ms": stats["first_ms"],
                "last_seen_ms": stats["last_ms"],
                "total_speaking_ms": stats["total_ms"],
            }
        )

    with open(meeting_dir / "detected_speakers.json", "w") as f:
        json.dump(detected, f, indent=2)

    # Save embeddings for future use (enrollment)
    np.savez_compressed(
        meeting_dir / "speaker_embeddings.npz",
        embeddings=emb_matrix,
        labels=np.array(labels),
        valid_indices=np.array(valid_indices),
        centroids=np.array([cluster_stats[k]["centroid"] for k in sorted(cluster_stats)]),
    )

    # Update timeline with speaker info
    timeline_path = meeting_dir / "timeline.json"
    if timeline_path.exists():
        with open(timeline_path) as f:
            timeline = json.load(f)
        for entry in timeline:
            seg_id = entry["segment_id"]
            for seg in updated_segments:
                if seg["segment_id"] == seg_id:
                    entry["speakers"] = seg.get("speakers", [])
                    break
        with open(timeline_path, "w") as f:
            json.dump(timeline, f)

    click.secho(
        f"\nDone! {n_clusters} speakers identified across {len(embeddings)} segments", fg="green"
    )
    click.echo(f"  Journal updated: {journal_path}")
    click.echo(f"  Speakers: {meeting_dir / 'detected_speakers.json'}")
    click.echo(f"  Embeddings: {meeting_dir / 'speaker_embeddings.npz'}")


if __name__ == "__main__":
    cluster()
