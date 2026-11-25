from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path
from urllib.parse import urlparse

from .schema import ClipError, Transcript, srt_from_transcript


def is_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def require_binary(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise ClipError(f"No se encontró '{name}' en el PATH. Instálalo para continuar.")
    return path


def run(command: list[str], *, capture: bool = False, description: str | None = None) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            check=True,
            text=True,
            stdout=subprocess.PIPE if capture else None,
            stderr=subprocess.PIPE if capture else None,
        )
    except FileNotFoundError as exc:
        raise ClipError(f"No se pudo ejecutar '{command[0]}'") from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip() if capture else ""
        label = description or " ".join(command[:2])
        raise ClipError(f"Falló {label} (código {exc.returncode}). {detail}".strip()) from exc


def probe_duration(path: Path) -> float:
    require_binary("ffprobe")
    result = run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture=True,
        description="ffprobe",
    )
    try:
        return float(result.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError) as exc:
        raise ClipError(f"No se pudo obtener la duración de {path}") from exc


def download_video(url: str, workdir: Path) -> Path:
    require_binary("yt-dlp")
    result = run(
        [
            "yt-dlp",
            "-f",
            "bv*+ba/b",
            "--merge-output-format",
            "mp4",
            "--no-playlist",
            "--print",
            "after_move:filepath",
            "-o",
            str(workdir / "%(id)s.%(ext)s"),
            url,
        ],
        capture=True,
        description="yt-dlp",
    )
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        raise ClipError("yt-dlp no devolvió la ruta del video descargado")
    path = Path(lines[-1])
    if not path.exists():
        raise ClipError(f"No existe el archivo descargado: {path}")
    return path


def extract_audio(video: Path, workdir: Path) -> Path:
    require_binary("ffmpeg")
    audio = workdir / "audio_16k.wav"
    run(
        [
            "ffmpeg",
            "-nostdin",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(video),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(audio),
        ],
        description="extracción de audio",
    )
    return audio


def render_clip(
    video: Path,
    start: float,
    end: float,
    output: Path,
    vertical: bool,
    *,
    crop: tuple[int, int, int, int] | None = None,
    srt: Path | None = None,
    burn_subtitles: bool = True,
) -> None:
    require_binary("ffmpeg")
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-nostdin",
        "-y",
        "-loglevel",
        "error",
        "-ss",
        f"{start:.3f}",
        "-to",
        f"{end:.3f}",
        "-i",
        str(video),
    ]
    filters: list[str] = []
    if vertical:
        if crop:
            x, y, crop_width, crop_height = crop
            filters.append(f"crop={crop_width}:{crop_height}:{x}:{y}")
        filters.append("scale=1080:1920:force_original_aspect_ratio=increase:flags=lanczos,crop=1080:1920,setsar=1")
    if burn_subtitles and srt and srt.exists():
        filters.append(_subtitle_filter(srt))
    if filters:
        command += ["-vf", ",".join(filters)]
    command += [
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "22",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-movflags",
        "+faststart",
        str(output),
    ]
    run(command, description=f"render {output.name}")


def write_srt(transcript: Transcript, start: float, end: float, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(srt_from_transcript(transcript, start, end), encoding="utf-8")


def safe_stem(name: str) -> str:
    stem = re.sub(r"[^a-zA-Z0-9_-]+", "_", Path(name).stem).strip("_")
    return stem or "video"


def _subtitle_filter(srt: Path) -> str:
    escaped = str(srt.resolve()).replace("\\", "\\\\").replace("'", "\\'")
    style = (
        "Fontname=DejaVu Sans,Fontsize=54,PrimaryColour=&H00FFFFFF,"
        "OutlineColour=&H00000000,BackColour=&H90000000,BorderStyle=1,"
        "Outline=4,Shadow=2,Alignment=2,MarginV=140,Bold=1"
    )
    return f"subtitles='{escaped}':force_style='{style}'"
