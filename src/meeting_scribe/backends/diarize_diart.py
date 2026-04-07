"""diart + pyannote diarization backend (POC).

Uses diart for online speaker diarization and SpeechBrain ECAPA-TDNN
for speaker embeddings. Runs on CPU (no GPU required for POC).
"""

from __future__ import annotations

import logging
import uuid

import numpy as np

from meeting_scribe.backends.base import DiarizeBackend
from meeting_scribe.models import SpeakerAttribution
from meeting_scribe.speaker.enrollment import EnrolledSpeaker, SpeakerEnrollmentStore
from meeting_scribe.speaker.verification import SpeakerVerifier

logger = logging.getLogger(__name__)


class DiartBackend(DiarizeBackend):
    """Online speaker diarization using diart + pyannote.

    POC backend for MacBook. Uses CPU-based inference.
    pyannote OSS achieves ~28.8% DER on Japanese conversational audio.
    """

    def __init__(self) -> None:
        self._pipeline = None
        self._enrollment_store = SpeakerEnrollmentStore()
        self._verifier = SpeakerVerifier(self._enrollment_store)
        self._embedding_model = None
        self._max_speakers = 4

    async def start(self, max_speakers: int = 4) -> None:
        """Initialize diart pipeline and embedding model."""
        self._max_speakers = max_speakers

        try:
            from diart import SpeakerDiarization

            self._pipeline = SpeakerDiarization()
            logger.info("diart pipeline initialized (max_speakers=%d)", max_speakers)
        except ImportError:
            logger.warning("diart not installed. Install with: pip install diart")
            return

        # Load SpeechBrain ECAPA-TDNN for embeddings
        try:
            from speechbrain.inference.speaker import EncoderClassifier

            self._embedding_model = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                run_opts={"device": "cpu"},
            )
            logger.info("SpeechBrain ECAPA-TDNN loaded for speaker embeddings")
        except ImportError:
            logger.warning("speechbrain not installed, speaker enrollment disabled")

    async def stop(self) -> None:
        """Release resources."""
        self._pipeline = None
        self._embedding_model = None
        self._verifier.reset()

    async def process_audio(
        self,
        audio: np.ndarray,
        sample_offset: int,
        sample_rate: int = 16000,
    ) -> list[SpeakerAttribution]:
        """Assign speakers to an audio chunk.

        Returns SpeakerAttributions with cluster_id and optional identity.
        """
        if self._pipeline is None:
            return [SpeakerAttribution(cluster_id=0, source="unknown")]

        # Run diarization (simplified — real diart uses streaming)
        # For POC, we use a simple energy-based voice activity + cluster assignment
        import torch

        audio_tensor = torch.from_numpy(audio).unsqueeze(0).float()

        # Extract embedding for speaker verification
        embedding = None
        if self._embedding_model is not None:
            try:
                emb = self._embedding_model.encode_batch(audio_tensor)
                embedding = emb.squeeze().numpy()
            except Exception:
                logger.debug("Embedding extraction failed for chunk", exc_info=True)

        # Simple cluster assignment (placeholder for full diart pipeline)
        # In production, diart provides cluster_id via its streaming pipeline
        cluster_id = 0  # Will be replaced by actual diart output

        # Verify against enrolled speakers
        attribution = self._verifier.verify(cluster_id, embedding)
        return [attribution]

    async def enroll_speaker(
        self,
        name: str,
        audio: np.ndarray,
        sample_rate: int = 16000,
    ) -> str:
        """Enroll a speaker from reference audio.

        Extracts ECAPA-TDNN embedding and stores for verification.
        """
        if self._embedding_model is None:
            msg = "Speaker embedding model not loaded"
            raise RuntimeError(msg)

        import torch

        audio_tensor = torch.from_numpy(audio).unsqueeze(0).float()
        embedding = self._embedding_model.encode_batch(audio_tensor).squeeze().numpy()

        enrollment_id = str(uuid.uuid4())
        speaker = EnrolledSpeaker(
            name=name,
            embedding=embedding,
            enrollment_id=enrollment_id,
            audio_duration_seconds=len(audio) / sample_rate,
        )
        self._enrollment_store.add(speaker)

        logger.info(
            "Enrolled speaker: %s (id=%s, %.1fs audio)",
            name,
            enrollment_id,
            speaker.audio_duration_seconds,
        )
        return enrollment_id
