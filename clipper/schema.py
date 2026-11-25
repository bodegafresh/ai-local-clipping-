from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class ClipError(Exception):
    pass


@dataclass
class Word:
    start: float
    end: float
    text: str

    def to_dict(self) -> dict[str, Any]:
        return {"inicio": round(self.start, 3), "fin": round(self.end, 3), "texto": self.text}


@dataclass
class Segment:
    start: float
    end: float
    text: str
    words: list[Word] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "inicio": round(self.start, 3),
            "fin": round(self.end, 3),
            "texto": self.text,
            "palabras": [word.to_dict() for word in self.words],
        }


@dataclass
class Transcript:
    segments: list[Segment]
    language: str | None = None
    duration: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "idioma": self.language,
            "duracion_segundos": self.duration,
            "segmentos": [segment.to_dict() for segment in self.segments],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Transcript:
        raw_segments = data.get("segmentos", data.get("segments", []))
        segments: list[Segment] = []
        for raw in raw_segments:
            start = _as_float(raw.get("inicio", raw.get("start")))
            end = _as_float(raw.get("fin", raw.get("end")))
            text = str(raw.get("texto", raw.get("text", ""))).strip()
            raw_words = raw.get("palabras", raw.get("words", []))
            words: list[Word] = []
            for raw_word in raw_words:
                word_start = _as_float(raw_word.get("inicio", raw_word.get("start")))
                word_end = _as_float(raw_word.get("fin", raw_word.get("end")))
                word_text = str(raw_word.get("texto", raw_word.get("word", raw_word.get("text", "")))).strip()
                if word_text:
                    words.append(Word(word_start, word_end, word_text))
            if text or words:
                segments.append(Segment(start, end, text or " ".join(w.text for w in words), words))
        if not segments:
            raise ClipError("La transcripción no contiene segmentos válidos")
        return cls(
            segments=segments,
            language=data.get("idioma", data.get("language")),
            duration=_as_optional_float(data.get("duracion_segundos", data.get("duration"))),
        )

    def words(self) -> list[Word]:
        return [word for segment in self.segments for word in segment.words]

    def text_for_range(self, start: float, end: float) -> str:
        words = self.words()
        if words:
            chosen = [word for word in words if word.end > start and word.start < end]
            if chosen:
                return _clean_text(" ".join(word.text for word in chosen))
        parts: list[str] = []
        for segment in self.segments:
            if segment.end > start and segment.start < end:
                if segment.text:
                    parts.append(segment.text.strip())
        return _clean_text(" ".join(parts))

    def snap_range(self, start: float, end: float) -> tuple[float, float]:
        words = self.words()
        if not words:
            return start, end
        starts = [word.start for word in words if word.end > start and word.start <= start + 1.5]
        if starts:
            start = min(starts)
        ends = [word.end for word in words if word.start < end and word.end >= end - 1.5]
        if ends:
            end = max(ends)
        return start, end

    def total_duration(self) -> float:
        if self.duration:
            return self.duration
        return max(segment.end for segment in self.segments)


@dataclass
class ClipCandidate:
    start: float
    end: float
    title: str
    description: str
    tags: list[str]
    reason: str
    score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "inicio": round(self.start, 3),
            "fin": round(self.end, 3),
            "titulo": self.title,
            "descripcion": self.description,
            "tags": self.tags,
            "razon": self.reason,
            "score": self.score,
        }


@dataclass
class ClipSettings:
    clip_count: int = 5
    min_duration: float = 15.0
    max_duration: float = 60.0

    def validate(self) -> None:
        if self.clip_count < 1:
            raise ClipError("--clips debe ser al menos 1")
        if self.min_duration < 1:
            raise ClipError("--min-duration debe ser al menos 1")
        if self.max_duration <= self.min_duration:
            raise ClipError("--max-duration debe ser mayor que --min-duration")


SUMMARY_KEYS = ("general", "temas", "tono", "audiencia")


def normalize_summary(raw: Any) -> dict[str, Any]:
    data = raw if isinstance(raw, dict) else {}
    temas = data.get("temas", [])
    if isinstance(temas, str):
        temas = [tema.strip() for tema in re.split(r"[,;]", temas) if tema.strip()]
    return {
        "general": str(data.get("general", data.get("resumen", ""))).strip(),
        "temas": [str(tema).strip() for tema in temas if str(tema).strip()][:10],
        "tono": str(data.get("tono", "")).strip(),
        "audiencia": str(data.get("audiencia", "")).strip(),
    }


def normalize_selection(
    raw: dict[str, Any],
    transcript: Transcript,
    settings: ClipSettings,
) -> tuple[dict[str, Any], list[ClipCandidate]]:
    summary = normalize_summary(raw.get("resumen", raw.get("summary", {})))
    duration = transcript.total_duration()
    raw_clips = raw.get("clips", [])
    if not isinstance(raw_clips, list):
        raw_clips = []
    candidates: list[ClipCandidate] = []
    for item in raw_clips:
        if not isinstance(item, dict):
            continue
        start = _as_optional_float(item.get("inicio", item.get("start")))
        end = _as_optional_float(item.get("fin", item.get("end")))
        if start is None or end is None or end <= start:
            continue
        start, end = transcript.snap_range(start, end)
        if end - start > settings.max_duration:
            end = start + settings.max_duration
        if end - start < settings.min_duration:
            extended_end = min(duration, start + settings.min_duration)
            start = max(0.0, extended_end - settings.min_duration)
            end = min(duration, start + settings.min_duration)
        if end - start < max(2.0, settings.min_duration * 0.6):
            continue
        start = max(0.0, min(start, duration - 1.0))
        end = min(duration, max(end, start + 1.0))
        tags = item.get("tags", [])
        if isinstance(tags, str):
            tags = [tag.strip() for tag in re.split(r"[,;]", tags) if tag.strip()]
        candidates.append(
            ClipCandidate(
                start=start,
                end=end,
                title=str(item.get("titulo", item.get("title", "Clip destacado"))).strip() or "Clip destacado",
                description=str(item.get("descripcion", item.get("description", ""))).strip(),
                tags=_normalize_tags([str(tag) for tag in tags]),
                reason=str(item.get("razon", item.get("reason", ""))).strip(),
                score=_as_float(item.get("score", 0.5)),
            )
        )
    return summary, dedupe_candidates(candidates, settings.clip_count)


def heuristic_candidates(transcript: Transcript, settings: ClipSettings) -> list[ClipCandidate]:
    segments = transcript.segments
    target = min(settings.max_duration, max(settings.min_duration, 30.0))
    scored: list[tuple[float, int]] = []
    for index, segment in enumerate(segments):
        text = segment.text
        score = 0.0
        score += 2.0 if "?" in text else 0.0
        score += 1.5 if "!" in text else 0.0
        score += 1.0 if re.search(r"\b(clave|importante|mejor|peor|nunca|siempre|primera|último|hoy|leyenda)\b", text, re.I) else 0.0
        score += min(len(text.split()), 30) / 30
        score -= max(0.0, (segment.end - segment.start) - 20) / 20
        scored.append((score, index))
    scored.sort(reverse=True)
    chosen: list[ClipCandidate] = []
    used: set[int] = set()
    for _, index in scored:
        if len(chosen) >= settings.clip_count:
            break
        start = max(0.0, segments[index].start - target / 3)
        end = min(transcript.total_duration(), start + target)
        overlap = any(not (end <= c.start or start >= c.end) for c in chosen)
        if overlap:
            continue
        indexes = {i for i, seg in enumerate(segments) if not (seg.end <= start or seg.start >= end)}
        if indexes & used:
            continue
        used |= indexes
        chosen.append(
            ClipCandidate(
                start=start,
                end=end,
                title=segments[index].text[:70].strip() or "Clip destacado",
                description="Fragmento seleccionado automáticamente por contenido destacado.",
                tags=["clips", "podcast", "highlights"],
                reason="Selección heurística: Ollama no devolvió clips válidos.",
                score=0.0,
            )
        )
    return sorted(chosen, key=lambda clip: clip.start)


def build_output(
    *,
    source: str,
    source_path: Path | None,
    url: str | None,
    duration: float,
    transcript: Transcript,
    summary: dict[str, Any],
    clips: list[ClipCandidate],
    whisper_model: str,
    ollama_model: str,
    render_dir: Path | None,
    video_name: str,
    warnings: list[str],
    crops: list[tuple[int, int, int, int] | None] | None = None,
    subtitles_burned: bool = False,
) -> dict[str, Any]:
    rendered_clips: list[dict[str, Any]] = []
    for index, clip in enumerate(clips, start=1):
        crop = crops[index - 1] if crops and index - 1 < len(crops) else None
        clip_id = f"clip_{index:02d}"
        video_file = render_dir / f"{clip_id}.mp4" if render_dir else None
        srt_file = render_dir / f"{clip_id}.srt" if render_dir else None
        hashtags = _hashtags(clip.tags)
        title = clip.title
        description = clip.description
        rendered_clips.append(
            {
                "id": clip_id,
                "inicio": round(clip.start, 3),
                "fin": round(clip.end, 3),
                "duracion_segundos": round(clip.end - clip.start, 3),
                "inicio_tiempo": format_timestamp(clip.start),
                "fin_tiempo": format_timestamp(clip.end),
                "titulo": title,
                "descripcion": description,
                "tags": clip.tags,
                "hashtags": hashtags,
                "transcripcion": transcript.text_for_range(clip.start, clip.end),
                "razon": clip.reason,
                "score": round(clip.score, 3),
                "formato": {
                    "relacion_aspecto": "9:16",
                    "resolucion": "1080x1920",
                    "orientacion": "vertical",
                    "encuadre": "rostro" if crop else "centro",
                    "recorte": list(crop) if crop else None,
                    "subtitulos_incrustados": subtitles_burned and srt_file is not None,
                },
                "publicacion": {
                    "tiktok": {"titulo": title[:150], "descripcion": f"{description}\n\n{hashtags}".strip()},
                    "instagram_reels": {"titulo": title[:220], "descripcion": f"{description}\n\n{hashtags}".strip()},
                    "youtube_shorts": {
                        "titulo": title[:100],
                        "descripcion": f"{description}\n\n{hashtags}".strip(),
                        "tags": ", ".join(clip.tags),
                    },
                },
                "miniatura_tiempo": round(clip.start + min(2.0, (clip.end - clip.start) / 4), 3),
                "archivo": str(video_file) if video_file and video_file.exists() else None,
                "subtitulos": str(srt_file) if srt_file and srt_file.exists() else None,
            }
        )
    return {
        "version": 1,
        "generado_en": datetime.now(timezone.utc).isoformat(),
        "fuente": {
            "entrada": source,
            "archivo": str(source_path) if source_path else None,
            "url": url,
            "duracion_segundos": round(duration, 3),
            "duracion": format_timestamp(duration),
            "idioma": transcript.language,
            "whisper": whisper_model,
            "ollama": ollama_model,
        },
        "resumen": summary,
        "transcripcion": {
            "segmentos": [
                {
                    "inicio": round(segment.start, 3),
                    "fin": round(segment.end, 3),
                    "texto": segment.text,
                }
                for segment in transcript.segments
            ]
        },
        "clips": rendered_clips,
        "advertencias": warnings,
        "total_clips": len(rendered_clips),
        "video_referencia": video_name,
    }


def format_timestamp(seconds: float, comma: bool = False) -> str:
    total_ms = int(round(max(0.0, seconds) * 1000))
    hours, total_ms = divmod(total_ms, 3_600_000)
    minutes, total_ms = divmod(total_ms, 60_000)
    secs, millis = divmod(total_ms, 1000)
    separator = "," if comma else "."
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{separator}{millis:03d}"


def srt_from_transcript(transcript: Transcript, start: float, end: float) -> str:
    entries: list[tuple[float, float, str]] = []

    def add_entry(entry_start: float, entry_end: float, text: str) -> None:
        clean = _clean_text(text)
        if not clean:
            return
        relative_start = max(0.0, entry_start - start)
        relative_end = max(relative_start + 0.2, min(entry_end, end) - start)
        entries.append((relative_start, relative_end, clean))

    words = [word for word in transcript.words() if word.end > start and word.start < end]
    if words:
        line: list[str] = []
        line_start = max(words[0].start, start)
        line_end = words[0].end
        for word in words:
            if not line:
                line_start = max(word.start, start)
            line.append(word.text)
            line_end = word.end
            if len(line) >= 7 or line_end - line_start >= 2.5:
                add_entry(line_start, line_end, " ".join(line))
                line = []
        if line:
            add_entry(line_start, line_end, " ".join(line))
    else:
        for segment in transcript.segments:
            if segment.end > start and segment.start < end:
                add_entry(max(segment.start, start), min(segment.end, end), segment.text)
    blocks: list[str] = []
    for index, (entry_start, entry_end, text) in enumerate(entries, start=1):
        blocks.append(
            f"{index}\n{format_timestamp(entry_start, comma=True)} --> {format_timestamp(entry_end, comma=True)}\n{text}\n"
        )
    return "\n".join(blocks)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _as_optional_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _normalize_tags(tags: list[str]) -> list[str]:
    result: list[str] = []
    for tag in tags:
        clean = re.sub(r"^#+", "", tag).strip().lower().replace(" ", "_")
        if clean and clean not in result:
            result.append(clean)
    return result[:12]


def _hashtags(tags: list[str]) -> str:
    return " ".join(f"#{tag}" for tag in tags)


def dedupe_candidates(candidates: list[ClipCandidate], limit: int) -> list[ClipCandidate]:
    chosen: list[ClipCandidate] = []
    for candidate in sorted(candidates, key=lambda item: (-item.score, item.start)):
        if any(
            min(candidate.end, other.end) - max(candidate.start, other.start)
            > 0.5 * min(candidate.end - candidate.start, other.end - other.start)
            for other in chosen
        ):
            continue
        chosen.append(candidate)
    return sorted(chosen[:limit], key=lambda item: item.start)
