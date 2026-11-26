from __future__ import annotations

import json
import os
import re
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .schema import (
    ClipCandidate,
    ClipSettings,
    ClipError,
    Transcript,
    dedupe_candidates,
    normalize_selection,
)

RESPONSE_FORMAT: dict[str, Any] = {
    "type": "object",
    "properties": {
        "resumen": {
            "type": "object",
            "properties": {
                "general": {"type": "string"},
                "temas": {"type": "array", "items": {"type": "string"}},
                "tono": {"type": "string"},
                "audiencia": {"type": "string"},
            },
            "required": ["general", "temas", "tono", "audiencia"],
        },
        "clips": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "inicio": {"type": "number"},
                    "fin": {"type": "number"},
                    "titulo": {"type": "string"},
                    "descripcion": {"type": "string"},
                    "tags": {"type": "array", "items": {"type": "string"}},
                    "razon": {"type": "string"},
                    "score": {"type": "number"},
                    "rank": {"type": "integer"},
                    "score_total": {"type": "number"},
                    "subestilo": {"type": "string"},
                    "gancho_3s": {"type": "string"},
                    "razonamiento_editorial": {"type": "string"},
                    "evidencia_acustica_visual": {"type": "string"},
                    "hook_visual_pantalla": {"type": "string"},
                    "titulo_corto": {"type": "string"},
                    "descripcion_redes": {"type": "string"},
                    "hashtags": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["inicio", "fin", "titulo", "descripcion", "tags", "razon", "score"],
            },
        },
    },
    "required": ["resumen", "clips"],
}

SYSTEM_PROMPT = (
    "Eres un editor profesional de clips virales para TikTok, Instagram Reels y YouTube Shorts. "
    "Trabajas con transcripciones de video en español enriquecidas con etiquetas acústicas. "
    "Razon primero la intención editorial, luego fija los tiempos exactos. "
    "Respondes únicamente con JSON válido, sin texto adicional, sin bloques de código y sin explicaciones."
)

STYLE_GUIDES: dict[str, str] = {
    "auto": (
        "Detecta el género dominante (humor, noticias, venta, educación, debate o gaming) y aplica sus reglas. "
        "Si el video es vertical y hay exclamaciones cortas con picos de energía, elige gaming. "
        "Devuelve el género elegido en 'estilo_aplicado'."
    ),
    "humor": (
        "Busca contradicciones absurdas, exageraciones, remates inesperados y sarcasmo respaldado por "
        "[PAUSA] o [REACCIÓN/RISAS]. Incluye la premisa completa al inicio y extiende el cierre al menos "
        "2.5 segundos después de la reacción o remate final."
    ),
    "noticias": (
        "Busca el titular o hecho central, cifras clave y consecuencias inmediatas. Abre directo con la "
        "noticia o dato más fuerte en los primeros 3 segundos y cierra con la conclusión o pregunta abierta."
    ),
    "vender": (
        "Busca la estructura Problema/Dolor -> Solución/Demostración -> Beneficio/Prueba -> Llamado a la acción. "
        "Elimina introducciones largas: arranca en el problema del cliente y cierra tras la propuesta de valor."
    ),
    "educar": (
        "Busca resolución de problemas técnicos, trucos accionables, comparativas rápidas o desmontar un mito. "
        "Cada clip debe ser una unidad autocontenida: problema y solución completos sin contexto previo."
    ),
    "debate": (
        "Busca posturas categóricas, frases disruptivas, preguntas retóricas y cruces de opinión intensos. "
        "Prioriza el gancho emocional o intelectual más fuerte exactamente en el segundo 0.0 del clip."
    ),
    "gaming": (
        "Busca una unidad narrativa Tensión -> Clímax -> Reacción: arranca en el momento de mayor peligro o "
        "decisión, incluye la jugada decisiva y cierra con la reacción de voz o el resultado. "
        "Prioriza exclamaciones cortas, picos de energía y sonido de gameplay; no penalices la baja densidad "
        "de palabras si la acción y el sonido sostienen el clip. Conserva 1.5 segundos antes y después del clímax."
    ),
}

VISION_RESPONSE_FORMAT: dict[str, Any] = {
    "type": "object",
    "properties": {
        "score_visual": {"type": "number"},
        "coherencia_emocional": {"type": "string"},
        "atractivo_visual": {"type": "string"},
        "start_ajustado": {"type": "number"},
        "end_ajustado": {"type": "number"},
        "razon": {"type": "string"},
        "contiene_diagrama_o_texto": {"type": "boolean"},
        "region_diagrama": {
            "type": "object",
            "properties": {
                "x_pct": {"type": "number"},
                "y_pct": {"type": "number"},
                "w_pct": {"type": "number"},
                "h_pct": {"type": "number"},
            },
            "required": ["x_pct", "y_pct", "w_pct", "h_pct"],
        },
        "tiene_webcam_esquina": {"type": "boolean"},
        "posicion_webcam": {
            "type": "string",
            "enum": ["bottom-right", "bottom-left", "top-right", "top-left", "none"],
        },
    },
    "required": ["score_visual", "start_ajustado", "end_ajustado"],
}


def style_prompt(style: str) -> str:
    return STYLE_GUIDES.get(style, STYLE_GUIDES["auto"])


def _resolve_style(summary: dict[str, Any], requested: str) -> str:
    applied = str(summary.get("estilo_aplicado", "")).strip().lower()
    if applied in STYLE_GUIDES:
        return applied
    return requested if requested in STYLE_GUIDES else "auto"


class OllamaClient:
    def __init__(self, host: str | None = None, model: str | None = None, timeout: int = 900) -> None:
        raw_host = host or os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
        if not raw_host.startswith(("http://", "https://")):
            raw_host = f"http://{raw_host}"
        self.host = raw_host.rstrip("/")
        self.model = model or os.environ.get("OLLAMA_MODEL", "qwen2.5:7b")
        self.timeout = timeout
        self._models: list[str] | None = None

    def list_models(self) -> list[str]:
        if self._models is not None:
            return list(self._models)
        try:
            with urlopen(f"{self.host}/api/tags", timeout=10) as response:
                data = json.load(response)
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise ClipError(
                f"No se pudo conectar con Ollama en {self.host}. Inicia el servicio con 'ollama serve'. Detalle: {exc}"
            ) from exc
        self._models = [str(item.get("name", "")) for item in data.get("models", []) if item.get("name")]
        return list(self._models)

    def check_model(self) -> None:
        names = set(self.list_models())
        if self.model not in names and f"{self.model}:latest" not in names:
            available = ", ".join(sorted(names)) or "ninguno"
            raise ClipError(
                f"El modelo '{self.model}' no está en Ollama. Modelos disponibles: {available}. "
                f"Descárgalo con: ollama pull {self.model}"
            )

    def generate(self, prompt: str, *, temperature: float = 0.2) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": SYSTEM_PROMPT,
            "stream": False,
            "format": RESPONSE_FORMAT,
            "keep_alive": "5m",
            "options": {"temperature": temperature, "num_ctx": 16384},
        }
        request = Request(
            f"{self.host}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                data = json.load(response)
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise ClipError(f"Ollama devolvió HTTP {exc.code}: {detail[:500]}") from exc
        except (URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise ClipError(f"Error al consultar Ollama: {exc}") from exc
        return _parse_json(data.get("response", ""))

    def chat_with_images(self, model: str, prompt: str, images: list[str]) -> dict[str, Any]:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt, "images": images}],
            "stream": False,
            "format": VISION_RESPONSE_FORMAT,
            "keep_alive": "5m",
            "options": {"temperature": 0.1, "num_ctx": 8192},
        }
        request = Request(
            f"{self.host}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                data = json.load(response)
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise ClipError(f"Ollama visión devolvió HTTP {exc.code}: {detail[:500]}") from exc
        except (URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise ClipError(f"Error al consultar Ollama visión: {exc}") from exc
        content = data.get("message", {}).get("content", data.get("response", ""))
        return _parse_json(str(content))


def analyze_transcript(
    client: OllamaClient,
    transcript: Transcript,
    settings: ClipSettings,
    *,
    chunk_chars: int = 12000,
    style: str = "auto",
) -> tuple[dict[str, Any], list[ClipCandidate]]:
    chunks = chunk_transcript(transcript, chunk_chars)
    if len(chunks) == 1:
        prompt = selection_prompt(chunks[0], settings, style=style)
        summary, candidates = normalize_selection(client.generate(prompt), transcript, settings)
        summary["estilo_aplicado"] = _resolve_style(summary, style)
        return summary, candidates

    settings_per_chunk = ClipSettings(
        clip_count=max(settings.clip_count * 2, 8),
        min_duration=settings.min_duration,
        max_duration=settings.max_duration,
    )
    summaries: list[dict[str, Any]] = []
    all_candidates: list[ClipCandidate] = []
    for number, chunk in enumerate(chunks, start=1):
        raw = client.generate(selection_prompt(chunk, settings_per_chunk, style=style))
        summary, candidates = normalize_selection(raw, chunk, settings_per_chunk)
        if summary["general"]:
            summaries.append(summary)
        all_candidates.extend(candidates)
    if not all_candidates:
        raise ClipError("Ollama no devolvió clips válidos en ningún fragmento")
    raw = client.generate(
        ranking_prompt(all_candidates, settings, transcript.language),
        temperature=0.1,
    )
    summary, candidates = normalize_selection(raw, transcript, settings)
    summary["estilo_aplicado"] = _resolve_style(summary, style)
    if not candidates:
        candidates = dedupe_candidates(all_candidates, settings.clip_count)
    if not summary["general"] and summaries:
        summary = dict(summaries[0])
        summary["temas"] = list(dict.fromkeys(tema for item in summaries for tema in item["temas"]))[:10]
    return summary, candidates


def selection_prompt(transcript: Transcript, settings: ClipSettings, *, style: str = "auto") -> str:
    gaming_note = (
        "Para gaming: cada clip debe contener Tensión, Clímax y Reacción completos; prioriza sonido de acción "
        "y exclamaciones sobre cantidad de palabras.\n"
        if style == "gaming"
        else ""
    )
    return (
        f"Selecciona hasta {settings.clip_count} clips de la siguiente transcripción.\n"
        f"PERFIL EDITORIAL ({style}): {style_prompt(style)}\n"
        f"Duración total: {transcript.total_duration():.1f} segundos. Idioma: {transcript.language or 'detectar'}.\n"
        f"Cada clip debe durar entre {settings.min_duration:.0f} y {settings.max_duration:.0f} segundos.\n"
        f"{gaming_note}"
        "Elige momentos con gancho, tensión, humor, revelación, consejo práctico o cierre memorable. "
        "Cada clip debe entenderse sin contexto previo. No inventes contenido que no esté en la transcripción.\n"
        "Devuelve 'resumen' con: general (2-4 frases), temas (3-6 palabras clave), tono y audiencia.\n"
        "Devuelve 'estilo_aplicado' y 'clips' con: inicio y fin en segundos decimales dentro de la duración total, "
        "titulo (máximo 70 caracteres, gancho), descripcion (1-2 frases lista para publicar), "
        "tags (5-8 palabras clave sin #), razon (por qué puede viralizarse), score (0 a 1), "
        "rank, score_total (1-100), subestilo, gancho_3s, razonamiento_editorial, "
        "evidencia_acustica_visual, hook_visual_pantalla (4-5 palabras), titulo_corto, "
        "descripcion_redes (copy listo con pregunta abierta) y hashtags (5 sin #).\n"
        "Usa las etiquetas [PAUSA x.xs], [ÉNFASIS ALTO] y [REACCIÓN/RISAS] como evidencia.\n\n"
        "TRANSCRIPCIÓN CON MARCAS DE TIEMPO [inicio-fin]:\n"
        f"{format_transcript(transcript)}\n\n"
        "Responde solo con el JSON solicitado."
    )


def ranking_prompt(
    candidates: list[ClipCandidate],
    settings: ClipSettings,
    language: str | None,
) -> str:
    return (
        f"Del siguiente listado de candidatos elige los {settings.clip_count} mejores clips para redes sociales.\n"
        f"Idioma: {language or 'detectar'}. Duración deseada: {settings.min_duration:.0f}-{settings.max_duration:.0f} segundos.\n"
        "Debes devolver exactamente el mismo esquema JSON: resumen general del video completo y clips con "
        "inicio, fin, titulo, descripcion, tags, razon y score. Puedes ajustar ligeramente los tiempos "
        "para mejorar el gancho, sin salir de los límites de duración. Elimina duplicados.\n\n"
        f"CANDIDATOS:\n{json.dumps([item.to_dict() for item in candidates], ensure_ascii=False, indent=2)}\n\n"
        "Responde solo con el JSON solicitado."
    )


def chunk_transcript(transcript: Transcript, max_chars: int) -> list[Transcript]:
    chunks: list[Transcript] = []
    current: list[Any] = []
    size = 0
    for segment in transcript.segments:
        length = len(segment.text) + 24
        if current and size + length > max_chars:
            chunks.append(Transcript(current, transcript.language, transcript.duration))
            current = []
            size = 0
        current.append(segment)
        size += length
    if current:
        chunks.append(Transcript(current, transcript.language, transcript.duration))
    return chunks or [transcript]


def format_transcript(transcript: Transcript) -> str:
    lines: list[str] = []
    for segment in transcript.segments:
        text = re.sub(r"\s+", " ", segment.text).strip()
        if text:
            lines.append(f"[{segment.start:.1f}-{segment.end:.1f}] {text}")
    return "\n".join(lines)


def _parse_json(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end <= start:
        raise ClipError(f"Ollama devolvió una respuesta que no es JSON: {cleaned[:300]}")
    try:
        data = json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError as exc:
        raise ClipError(f"No se pudo parsear la respuesta de Ollama: {exc}") from exc
    if not isinstance(data, dict):
        raise ClipError("La respuesta de Ollama no es un objeto JSON")
    return data
