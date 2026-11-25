from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .media import require_binary, run
from .schema import ClipError, Segment, Transcript, Word


def transcribe_audio(
    audio: Path,
    workdir: Path,
    *,
    model_name: str,
    language: str | None,
    device: str | None,
    backend: str = "auto",
) -> Transcript:
    if backend in {"auto", "faster"}:
        transcript = _transcribe_faster_whisper(audio, model_name, language, device)
        if transcript:
            return transcript
        if backend == "faster":
            raise ClipError("faster-whisper no está instalado (pip install faster-whisper)")
    if backend in {"auto", "openai"}:
        transcript = _transcribe_openai_whisper(audio, model_name, language, device)
        if transcript:
            return transcript
        if backend == "openai":
            raise ClipError("openai-whisper no está instalado (pip install openai-whisper)")
    if backend == "cli":
        return _transcribe_whisper_cli(audio, workdir, model_name, language)
    if backend == "auto":
        raise ClipError(
            "No hay ningún backend de Whisper instalado. Ejecuta: "
            "python3 -m pip install openai-whisper (o instala el paquete con 'python3 -m pip install -e .')"
        )
    raise ClipError(f"Backend de Whisper desconocido: {backend}")


def _transcribe_openai_whisper(
    audio: Path,
    model_name: str,
    language: str | None,
    device: str | None,
) -> Transcript | None:
    try:
        import whisper
    except ImportError:
        return None
    try:
        model = whisper.load_model(model_name, device=device) if device else whisper.load_model(model_name)
        result = model.transcribe(
            str(audio),
            language=language,
            word_timestamps=True,
            verbose=False,
        )
    except Exception as exc:
        raise ClipError(f"Error al transcribir con openai-whisper: {exc}") from exc
    return _build_transcript(result.get("segments", []), result.get("language", language))


def _transcribe_faster_whisper(
    audio: Path,
    model_name: str,
    language: str | None,
    device: str | None,
) -> Transcript | None:
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        return None
    try:
        compute_type = "float16" if device == "cuda" else "int8"
        model = WhisperModel(model_name, device=device or "cpu", compute_type=compute_type)
        segments, info = model.transcribe(
            str(audio),
            language=language,
            word_timestamps=True,
        )
        raw_segments = [
            {
                "start": segment.start,
                "end": segment.end,
                "text": segment.text,
                "words": [
                    {"start": word.start, "end": word.end, "word": word.word}
                    for word in (segment.words or [])
                ],
            }
            for segment in segments
        ]
        return _build_transcript(raw_segments, info.language or language)
    except Exception as exc:
        raise ClipError(f"Error al transcribir con faster-whisper: {exc}") from exc


def _transcribe_whisper_cli(
    audio: Path,
    workdir: Path,
    model_name: str,
    language: str | None,
) -> Transcript:
    require_binary("whisper")
    output_json = workdir / f"{audio.stem}.json"
    output_json.unlink(missing_ok=True)
    command = [
        "whisper",
        str(audio),
        "--model",
        model_name,
        "--word_timestamps",
        "True",
        "--output_format",
        "json",
        "--output_dir",
        str(workdir),
    ]
    if language:
        command += ["--language", language]
    run(command, description="Whisper CLI")
    if not output_json.exists():
        raise ClipError(f"Whisper no generó {output_json}")
    try:
        data = json.loads(output_json.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ClipError(f"JSON de Whisper inválido: {exc}") from exc
    return _build_transcript(data.get("segments", []), data.get("language", language))


def _build_transcript(raw_segments: list[dict[str, Any]], language: str | None) -> Transcript:
    segments: list[Segment] = []
    for raw in raw_segments:
        words: list[Word] = []
        for raw_word in raw.get("words") or []:
            text = str(raw_word.get("word", raw_word.get("text", ""))).strip()
            if not text:
                continue
            start = float(raw_word.get("start", 0.0))
            end = float(raw_word.get("end", start + 0.2))
            words.append(Word(start=start, end=end, text=text))
        text = str(raw.get("text", "")).strip()
        if not text and words:
            text = " ".join(word.text for word in words)
        start = float(raw.get("start", words[0].start if words else 0.0))
        end = float(raw.get("end", words[-1].end if words else start + 0.1))
        if text:
            segments.append(Segment(start=start, end=end, text=text, words=words))
    if not segments:
        raise ClipError("Whisper no devolvió ningún segmento con texto")
    return Transcript(segments=segments, language=language, duration=segments[-1].end)
