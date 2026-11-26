from __future__ import annotations

import wave
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median

import numpy as np

from .schema import ClipError, Segment, Transcript


@dataclass
class AudioEnrichment:
    transcript: Transcript
    peaks: dict[int, float] = field(default_factory=dict)
    events: dict[int, list[str]] = field(default_factory=dict)


def _read_wav(path: Path) -> tuple[int, np.ndarray]:
    try:
        with wave.open(str(path), "rb") as handle:
            sample_rate = handle.getframerate()
            width = handle.getsampwidth()
            channels = handle.getnchannels()
            raw = handle.readframes(handle.getnframes())
    except (OSError, EOFError, wave.Error) as exc:
        raise ClipError(f"No se pudo leer el WAV {path}: {exc}") from exc
    if width != 2:
        raise ClipError(f"Se esperaba WAV PCM de 16 bits: {path}")
    samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        usable = (len(samples) // channels) * channels
        samples = samples[:usable].reshape(-1, channels).mean(axis=1, dtype=np.float32)
    return sample_rate, samples


def _rms(samples: np.ndarray, sample_rate: int, start: float, end: float) -> float:
    first = max(0, int(start * sample_rate))
    last = min(len(samples), int(end * sample_rate))
    if last <= first:
        return 0.0
    window = samples[first:last].astype(np.float64)
    return float(np.sqrt(np.mean(window * window)))


def _noise_floor(samples: np.ndarray, sample_rate: int, window: float = 0.25) -> float:
    step = max(1, int(sample_rate * window))
    usable = (len(samples) // step) * step
    if usable == 0:
        return 0.0
    windows = samples[:usable].reshape(-1, step).astype(np.float64)
    values = np.sqrt(np.mean(windows * windows, axis=1))
    values.sort()
    return float(values[min(len(values) - 1, int(len(values) * 0.2))])


def _peak_time(samples: np.ndarray, sample_rate: int, start: float, end: float) -> float:
    first = max(0, int(start * sample_rate))
    last = min(len(samples), int(end * sample_rate))
    if last <= first:
        return max(0.0, start)
    return float((first + int(np.argmax(np.abs(samples[first:last])))) / sample_rate)


def enrich_transcript_with_audio(
    transcript: Transcript,
    audio: Path,
    *,
    pause_seconds: float = 1.4,
    emphasis_ratio: float = 1.35,
    reaction_ratio: float = 1.4,
) -> AudioEnrichment:
    sample_rate, samples = _read_wav(audio)
    duration = len(samples) / sample_rate
    segments = sorted(transcript.segments, key=lambda item: item.start)
    segment_rms = [
        _rms(samples, sample_rate, segment.start, min(segment.end, duration)) for segment in segments
    ]
    valid_rms = [value for value in segment_rms if value > 0]
    median_rms = median(valid_rms) if valid_rms else 0.0
    baseline = _noise_floor(samples, sample_rate)

    enriched_segments: list[Segment] = []
    peaks: dict[int, float] = {}
    events: dict[int, list[str]] = {}
    for index, segment in enumerate(segments):
        text = segment.text.strip()
        tags: list[str] = []
        rms = segment_rms[index]
        if median_rms > 0 and rms > median_rms * emphasis_ratio:
            tags.append("[ÉNFASIS ALTO]")

        post_start = min(segment.end, duration)
        post_end = min(post_start + 2.5, duration)
        post_rms = _rms(samples, sample_rate, post_start, post_end)
        if baseline > 0 and post_rms > baseline * reaction_ratio and post_rms > median_rms * 0.8:
            tags.append("[REACCIÓN/RISAS]")

        following = segments[index + 1] if index + 1 < len(segments) else None
        if following and following.start - segment.end >= pause_seconds:
            gap = following.start - segment.end
            tags.append(f"[PAUSA {gap:.1f}s]")

        if tags:
            text = f"{text} {' '.join(tags)}".strip()
            events[index] = tags
        peak_window_end = min(post_end, segment.end + 2.5)
        peaks[index] = _peak_time(samples, sample_rate, segment.start, max(peak_window_end, segment.end))
        enriched_segments.append(
            Segment(start=segment.start, end=segment.end, text=text, words=list(segment.words))
        )

    return AudioEnrichment(
        transcript=Transcript(enriched_segments, transcript.language, transcript.duration),
        peaks=peaks,
        events=events,
    )
