from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


FILLER_TOKENS = {"ah", "oh", "eh", "mm", "mhm", "um"}
GAMING_CUES = (
    "morí",
    "moriste",
    "gané",
    "ganaste",
    "victoria",
    "derrota",
    "emboscada",
    "escopeta",
    "cabeza",
    "jugada",
    "partida",
    "nivel",
    "jefe",
)


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
    rank: int = 0
    score_total: float = 0.0
    subestilo: str = ""
    gancho_3s: str = ""
    razonamiento_editorial: str = ""
    evidencia_acustica_visual: str = ""
    hook_visual_pantalla: str = ""
    titulo_corto: str = ""
    descripcion_redes: str = ""
    hashtags: list[str] = field(default_factory=list)
    diagram_roi: tuple[float, float, float, float] | None = None
    webcam_corner: str = "none"

    def to_dict(self) -> dict[str, Any]:
        return {
            "inicio": round(self.start, 3),
            "fin": round(self.end, 3),
            "titulo": self.title,
            "descripcion": self.description,
            "tags": self.tags,
            "razon": self.reason,
            "score": self.score,
            "rank": self.rank,
            "score_total": round(self.score_total, 1),
            "subestilo": self.subestilo,
            "gancho_3s": self.gancho_3s,
            "razonamiento_editorial": self.razonamiento_editorial,
            "evidencia_acustica_visual": self.evidencia_acustica_visual,
            "hook_visual_pantalla": self.hook_visual_pantalla,
            "titulo_corto": self.titulo_corto,
            "descripcion_redes": self.descripcion_redes,
            "hashtags": self.hashtags,
            "diagrama_region": list(self.diagram_roi) if self.diagram_roi else None,
            "webcam_esquina": self.webcam_corner,
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
        hashtags = item.get("hashtags", [])
        if isinstance(hashtags, str):
            hashtags = [tag.strip() for tag in re.split(r"[,;]", hashtags) if tag.strip()]
        diagram_roi = _as_roi(item.get("region_diagrama", item.get("diagrama_region")))
        webcam_corner = str(item.get("posicion_webcam", item.get("webcam_esquina", "none"))).strip().lower()
        title = str(item.get("titulo", item.get("title", "Clip destacado"))).strip() or "Clip destacado"
        description = str(item.get("descripcion", item.get("description", ""))).strip()
        short_title = str(item.get("titulo_corto", "")).strip() or title
        social_description = str(item.get("descripcion_redes", "")).strip() or description
        score_total = _as_float(item.get("score_total", 0.0))
        candidates.append(
            ClipCandidate(
                start=start,
                end=end,
                title=title,
                description=description,
                tags=_normalize_tags([str(tag) for tag in tags]),
                reason=str(item.get("razon", item.get("reason", ""))).strip(),
                score=_as_float(item.get("score", 0.5)),
                rank=int(_as_float(item.get("rank", 0))),
                score_total=score_total,
                subestilo=str(item.get("subestilo", "")).strip(),
                gancho_3s=str(item.get("gancho_3s", "")).strip(),
                razonamiento_editorial=str(item.get("razonamiento_editorial", "")).strip(),
                evidencia_acustica_visual=str(item.get("evidencia_acustica_visual", "")).strip(),
                hook_visual_pantalla=str(item.get("hook_visual_pantalla", "")).strip(),
                titulo_corto=short_title[:100],
                descripcion_redes=social_description,
                hashtags=_normalize_tags([str(tag) for tag in hashtags])[:5],
                diagram_roi=diagram_roi,
                webcam_corner=webcam_corner,
            )
        )
    return summary, dedupe_candidates(candidates, settings.clip_count)


def detect_gaming_style(transcript: Transcript, native_vertical: bool) -> bool:
    if not native_vertical:
        return False
    texts = [segment.text for segment in transcript.segments]
    joined = " ".join(texts)
    lowered = joined.lower()
    dynamic_tags = sum(joined.count(tag) for tag in ("[ÉNFASIS ALTO]", "[REACCIÓN/RISAS]"))
    exclamations = sum(text.count("!") for text in texts)
    short_bursts = sum(1 for text in texts if len(_word_tokens(text)) <= 5 and "!" in text)
    cues = sum(1 for cue in GAMING_CUES if cue in lowered)
    return dynamic_tags >= 1 and (exclamations >= 2 or short_bursts >= 2 or cues >= 1)


def heuristic_candidates(transcript: Transcript, settings: ClipSettings, style: str = "auto") -> list[ClipCandidate]:
    segments = transcript.segments
    gaming = style == "gaming"
    target = min(settings.max_duration, max(settings.min_duration, 30.0))
    scored: list[tuple[float, int]] = []
    for index, segment in enumerate(segments):
        text = segment.text
        score = 0.0
        score += 2.0 if "?" in text else 0.0
        score += 1.5 if "!" in text else 0.0
        score += 1.0 if re.search(r"\b(clave|importante|mejor|peor|nunca|siempre|primera|último|hoy|leyenda)\b", text, re.I) else 0.0
        if gaming:
            evidence = " ".join(word.text for word in segment.words) + " " + text
            score += 2.5 if any(tag in evidence for tag in ("[ÉNFASIS ALTO]", "[REACCIÓN/RISAS]")) else 0.0
            score += 2.0 if any(cue in text.lower() for cue in GAMING_CUES) else 0.0
        else:
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
    framing_modes: list[str] | None = None,
    style: str = "auto",
    source_url: str | None = None,
    timings: dict[str, float] | None = None,
    render_details: list[dict[str, float]] | None = None,
) -> dict[str, Any]:
    rendered_clips: list[dict[str, Any]] = []
    for index, clip in enumerate(clips, start=1):
        crop = crops[index - 1] if crops and index - 1 < len(crops) else None
        framing = framing_modes[index - 1] if framing_modes and index - 1 < len(framing_modes) else "centro"
        clip_id = f"clip_{index:02d}"
        video_file = render_dir / f"{clip_id}.mp4" if render_dir else None
        srt_file = render_dir / f"{clip_id}.srt" if render_dir else None
        metadata_file = render_dir / f"{clip_id}.json" if render_dir else None
        text_file = render_dir / f"{clip_id}.txt" if render_dir else None
        link = source_url_with_timestamp(source_url, clip.start)
        stamp = format_short_timestamp(clip.start)
        hashtags = _hashtags((clip.hashtags or clip.tags)[:5])
        title = clip.titulo_corto or clip.title
        description = clip.descripcion_redes or clip.description
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
                "rank": clip.rank or index,
                "score_total": round(clip.score_total, 1),
                "subestilo": clip.subestilo,
                "gancho_3s": clip.gancho_3s,
                "razonamiento_editorial": clip.razonamiento_editorial,
                "evidencia_acustica_visual": clip.evidencia_acustica_visual,
                "hook_visual_pantalla": clip.hook_visual_pantalla,
                "titulo_corto": title,
                "descripcion_redes": description,
                "source_url": normalize_source_url(source_url),
                "source_url_timestamp": link,
                "source_timestamp": stamp if link else None,
                "texto_publicacion": clip_post_text(clip, source_url),
                "primer_comentario_fijado": primer_comentario_fijado(clip, source_url),
                "formato": {
                    "relacion_aspecto": "9:16",
                    "resolucion": "1080x1920",
                    "orientacion": "vertical",
                    "encuadre": {"face": "rostro", "fit": "completo", "center": "centro"}.get(framing, framing),
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
                "metadatos": str(metadata_file) if metadata_file and metadata_file.exists() else None,
                "texto": str(text_file) if text_file and text_file.exists() else None,
            }
        )
    return {
        "version": 1,
        "generado_en": datetime.now(timezone.utc).isoformat(),
        "fuente": {
            "entrada": source,
            "archivo": str(source_path) if source_path else None,
            "url": normalize_source_url(source_url or url),
            "duracion_segundos": round(duration, 3),
            "duracion": format_timestamp(duration),
            "idioma": transcript.language,
            "whisper": whisper_model,
            "ollama": ollama_model,
        },
        "estilo_aplicado": summary.get("estilo_aplicado", style),
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
        "tiempos_segundos": dict(timings or {}),
        "tiempos_render_por_clip": [
            {f"clip_{index:02d}.{key}": round(value, 3) for key, value in detail.items()}
            for index, detail in enumerate(render_details or [], start=1)
        ],
    }


def format_timestamp(seconds: float, comma: bool = False) -> str:
    total_ms = int(round(max(0.0, seconds) * 1000))
    hours, total_ms = divmod(total_ms, 3_600_000)
    minutes, total_ms = divmod(total_ms, 60_000)
    secs, millis = divmod(total_ms, 1000)
    separator = "," if comma else "."
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{separator}{millis:03d}"


def format_short_timestamp(seconds: float) -> str:
    total = int(max(0.0, seconds))
    return f"{total // 60:02d}:{total % 60:02d}"


def normalize_source_url(value: str | None) -> str | None:
    if not value:
        return None
    url = value.strip()
    if not url:
        return None
    parsed = urlsplit(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        return None
    return url


def source_url_with_timestamp(source_url: str | None, start: float) -> str | None:
    url = normalize_source_url(source_url)
    if not url:
        return None
    seconds = int(max(0.0, start))
    parsed = urlsplit(url)
    host = parsed.netloc.lower().split("@")[-1].split(":")[0]
    query = [(key, value) for key, value in parse_qsl(parsed.query, keep_blank_values=True) if key not in {"t", "start"}]
    if host in {"youtu.be", "www.youtu.be"}:
        video_id = parsed.path.strip("/").split("/")[0]
        if video_id:
            query.append(("t", f"{seconds}s"))
            return urlunsplit(("https", "youtu.be", f"/{video_id}", urlencode(query), ""))
    if host.endswith("youtube.com"):
        params = dict(query)
        video_id = params.get("v")
        if not video_id:
            parts = parsed.path.strip("/").split("/")
            if len(parts) >= 2 and parts[0] in {"shorts", "embed", "live"}:
                video_id = parts[1]
        if video_id:
            ordered = [("v", video_id)]
            ordered.extend((key, value) for key, value in query if key != "v")
            ordered.append(("t", f"{seconds}s"))
            return urlunsplit(("https", "www.youtube.com", "/watch", urlencode(ordered), ""))
    query.append(("t", f"{seconds}s"))
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, urlencode(query), parsed.fragment))


def primer_comentario_fijado(clip: ClipCandidate, source_url: str | None) -> str | None:
    link = source_url_with_timestamp(source_url, clip.start)
    if not link:
        return None
    stamp = format_short_timestamp(clip.start)
    return f"📌 Mira el contexto completo desde el minuto {stamp} aquí 👉 {link}"


def clip_post_text(clip: ClipCandidate, source_url: str | None = None) -> str:
    title = clip.titulo_corto or clip.title
    description = clip.descripcion_redes or clip.description
    hashtags = _hashtags((clip.hashtags or clip.tags)[:5])
    blocks = [
        f"=== TÍTULO ===\n{title}",
        f"=== DESCRIPCIÓN (TIKTOK / REELS / SHORTS) ===\n{description}",
    ]
    link = source_url_with_timestamp(source_url, clip.start)
    if link:
        stamp = format_short_timestamp(clip.start)
        blocks.append(f"🎥 Video completo (minuto {stamp}): {link}")
    if hashtags:
        blocks.append(hashtags)
    comment = primer_comentario_fijado(clip, source_url)
    if comment:
        blocks.append("=== PRIMER COMENTARIO FIJADO (RECOMENDADO PARA REELS / TIKTOK) ===\n" + comment)
    return "\n\n".join(blocks) + "\n"


def write_clip_text(path: Path, clip: ClipCandidate, source_url: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(clip_post_text(clip, source_url), encoding="utf-8")


def subtitle_entries(transcript: Transcript, start: float, end: float) -> list[tuple[float, float, str]]:
    entries: list[tuple[float, float, str]] = []

    def add_entry(entry_start: float, entry_end: float, text: str) -> None:
        clean = _clean_text(text)
        if not clean or _is_filler_only(clean):
            return
        relative_start = max(0.0, entry_start - start)
        relative_end = max(relative_start + 0.2, min(entry_end, end) - start)
        entries.append((relative_start, relative_end, clean))

    words: list[Word] = []
    for segment in transcript.segments:
        if _is_filler_only(segment.text):
            continue
        words.extend(word for word in segment.words if word.end > start and word.start < end)
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
    return entries


def srt_from_transcript(transcript: Transcript, start: float, end: float) -> str:
    blocks: list[str] = []
    for index, (entry_start, entry_end, text) in enumerate(subtitle_entries(transcript, start, end), start=1):
        blocks.append(
            f"{index}\n{format_timestamp(entry_start, comma=True)} --> {format_timestamp(entry_end, comma=True)}\n{text}\n"
        )
    return "\n".join(blocks)


def _wrap_ass_text(text: str, max_chars: int = 22) -> str:
    wrapped: list[str] = []
    current = ""
    for word in text.split():
        candidate = f"{current} {word}" if current else word
        if current and len(candidate) > max_chars:
            wrapped.append(current)
            current = word
        else:
            current = candidate
    if current:
        wrapped.append(current)
    return "\\N".join(wrapped)


def _display_text(text: str) -> str:
    return _clean_text(re.sub(r"\[[^\]]*\]", "", text))


def viral_word_timings(transcript: Transcript, start: float, end: float) -> list[tuple[float, float, str]]:
    words: list[tuple[float, float, str]] = []
    for word in transcript.words():
        if word.end <= start or word.start >= end:
            continue
        text = _display_text(word.text)
        tokens = text.split()
        if not tokens or all(token in FILLER_TOKENS for token in _word_tokens(text)):
            continue
        word_start = max(0.0, word.start - start)
        word_end = max(0.0, min(word.end, end) - start)
        step = (word_end - word_start) / len(tokens)
        for index, token in enumerate(tokens):
            words.append(
                (
                    word_start + index * step,
                    word_start + (index + 1) * step,
                    token,
                )
            )
    if words:
        return words
    for segment in transcript.segments:
        if segment.end <= start or segment.start >= end:
            continue
        text = _display_text(segment.text)
        tokens = text.split()
        if not tokens or all(token in FILLER_TOKENS for token in _word_tokens(text)):
            continue
        segment_start = max(0.0, segment.start - start)
        segment_end = max(segment_start + 0.2, min(segment.end, end) - start)
        step = (segment_end - segment_start) / len(tokens)
        for index, token in enumerate(tokens):
            words.append(
                (
                    segment_start + index * step,
                    segment_start + (index + 1) * step,
                    token,
                )
            )
    return words


def viral_highlight_times(transcript: Transcript, start: float, end: float) -> tuple[list[float], list[float]]:
    emphasis: list[float] = []
    pauses: list[float] = []
    for segment in transcript.segments:
        if segment.end <= start or segment.start >= end:
            continue
        relative = max(0.0, (segment.start + segment.end) / 2 - start)
        marker = " ".join(word.text for word in segment.words) + " " + segment.text
        if "[ÉNFASIS ALTO]" in marker or "[REACCIÓN/RISAS]" in marker:
            emphasis.append(relative)
        if re.search(r"\[PAUSA\s+[\d.,]+s\]", marker):
            pauses.append(relative)
    return emphasis, pauses


def group_viral_words(
    words: list[tuple[float, float, str]],
    *,
    max_words: int = 3,
    pause_seconds: float = 0.35,
) -> list[list[tuple[float, float, str]]]:
    groups: list[list[tuple[float, float, str]]] = []
    current: list[tuple[float, float, str]] = []
    for index, word in enumerate(words):
        if current and (
            len(current) >= max_words or word[0] - current[-1][1] > pause_seconds
        ):
            groups.append(current)
            current = []
        current.append(word)
    if current:
        groups.append(current)
    return groups


def _viral_block_text(group: list[tuple[float, float, str]]) -> str:
    total_cs = max(1, int(round((group[-1][1] - group[0][0]) * 100)))
    durations = [max(1, int(round((word_end - word_start) * 100))) for word_start, word_end, _ in group]
    scale = total_cs / max(1, sum(durations))
    assigned = [max(1, int(round(value * scale))) for value in durations]
    assigned[-1] += total_cs - sum(assigned)
    colors = ("&H00CBEB91&", "&H0000E5FF&")
    parts = [
        r"{\fscx92\fscy92\t(0,70,\fscx104\fscy104)\t(70,130,\fscx100\fscy100)}"
    ]
    for index, (word_start, word_end, text) in enumerate(group):
        escaped = text.replace("{", "\\{").replace("}", "\\}")
        parts.append(f"{{\\k{assigned[index]}\\1c{colors[index % 2]}}}{escaped} ")
    return "".join(parts).rstrip()


def _hook_text(hook: str | None, title: str) -> str:
    source = _display_text(hook or " ".join(title.split()[:5])).upper()
    lines = _wrap_ass_text(source, max_chars=22).split("\\N")[:2]
    words = [word for line in lines for word in line.split()]
    accent_from = len(words) - 2 if len(words) > 3 or len(lines) > 1 else len(words)
    rendered_lines: list[str] = []
    index = 0
    for line in lines:
        rendered_words: list[str] = []
        for word in line.split():
            color = "&H00CBEB91&" if index >= accent_from else "&H00FFFFFF&"
            escaped = word.replace("{", "\\{").replace("}", "\\}")
            rendered_words.append(f"{{\\1c{color}}}{escaped}")
            index += 1
        rendered_lines.append(" ".join(rendered_words))
    return "\\N".join(rendered_lines)


def ass_from_transcript(
    transcript: Transcript,
    start: float,
    end: float,
    *,
    width: int = 1080,
    height: int = 1920,
    hook: str | None = None,
    title: str = "",
    viral_fx: bool = True,
) -> str:
    if not viral_fx:
        header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,DejaVu Sans,64,&H00FFFFFF,&H000000FF,&H00000000,&H90000000,-1,0,0,0,100,100,0,0,1,5,2,2,80,80,210,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
        lines = [header.strip()]
        for entry_start, entry_end, text in subtitle_entries(transcript, start, end):
            escaped = _wrap_ass_text(_display_text(text)).replace("{", "\\{").replace("}", "\\}")
            lines.append(
                f"Dialogue: 0,{ass_timestamp(entry_start)},{ass_timestamp(entry_end)},Default,,0,0,0,,{escaped}"
            )
        return "\n".join(lines) + "\n"

    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,DejaVu Sans,68,&H00FFFFFF,&H00CBEB91,&H000A0A0A,&H80000000,-1,0,0,0,100,100,0,0,1,5,2,2,70,70,390,1
Style: HookBanner,DejaVu Sans,54,&H00FFFFFF,&H00FFFFFF,&H2810141D,&H60000000,-1,0,0,0,100,100,1,0,3,18,0,8,70,70,175,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    lines = [header.strip()]
    banner = _hook_text(hook, title)
    if banner:
        lines.append(
            "Dialogue: 1,0:00:00.00,0:00:03.80,HookBanner,,0,0,0,,"
            r"{\fad(150,250)\fscx95\fscy95\t(0,120,\fscx100\fscy100)}" + banner
        )
    for group in group_viral_words(viral_word_timings(transcript, start, end)):
        lines.append(
            f"Dialogue: 0,{ass_timestamp(group[0][0])},{ass_timestamp(group[-1][1])},"
            f"Default,,0,0,0,,{_viral_block_text(group)}"
        )
    return "\n".join(lines) + "\n"


def ass_timestamp(seconds: float) -> str:
    total_cs = int(round(max(0.0, seconds) * 100))
    hours, total_cs = divmod(total_cs, 360_000)
    minutes, total_cs = divmod(total_cs, 6_000)
    secs, centis = divmod(total_cs, 100)
    return f"{hours}:{minutes:02d}:{secs:02d}.{centis:02d}"


def clip_metadata(clip: ClipCandidate, index: int, source_url: str | None = None) -> dict[str, Any]:
    return {
        "rank": clip.rank or index,
        "score_total": round(clip.score_total, 1),
        "subestilo": clip.subestilo,
        "hook_visual_pantalla": clip.hook_visual_pantalla,
        "titulo_corto": clip.titulo_corto or clip.title,
        "descripcion_redes": clip.descripcion_redes or clip.description,
        "hashtags": clip.hashtags or clip.tags,
        "gancho_3s": clip.gancho_3s,
        "source_url": normalize_source_url(source_url),
        "source_url_timestamp": source_url_with_timestamp(source_url, clip.start),
        "source_timestamp": format_short_timestamp(clip.start) if source_url else None,
        "texto_publicacion": clip_post_text(clip, source_url),
        "primer_comentario_fijado": primer_comentario_fijado(clip, source_url),
    }


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


def _word_tokens(text: str) -> list[str]:
    tokens: list[str] = []
    for raw in text.split():
        token = re.sub(r"[^a-záéíóúñü]+", "", raw.lower())
        if token:
            tokens.append(token)
    return tokens


def _is_filler_only(text: str) -> bool:
    tokens = _word_tokens(_display_text(text))
    return bool(tokens) and all(token in FILLER_TOKENS for token in tokens)


def _as_roi(value: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(value, dict):
        return None
    raw = [_as_float(value.get(key, 0.0)) for key in ("x_pct", "y_pct", "w_pct", "h_pct")]
    values = [number / 100.0 if number > 1.0 else number for number in raw]
    x, y, width, height = (max(0.0, min(1.0, number)) for number in values)
    width = min(width, 1.0 - x)
    height = min(height, 1.0 - y)
    if width <= 0.0 or height <= 0.0:
        return None
    return (round(x, 4), round(y, 4), round(width, 4), round(height, 4))


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
    for candidate in sorted(
        candidates,
        key=lambda item: (-(item.score_total or item.score * 100), item.start),
    ):
        if any(
            min(candidate.end, other.end) - max(candidate.start, other.start)
            > 0.5 * min(candidate.end - candidate.start, other.end - other.start)
            for other in chosen
        ):
            continue
        chosen.append(candidate)
    return sorted(chosen[:limit], key=lambda item: item.start)
