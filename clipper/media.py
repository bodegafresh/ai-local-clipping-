from __future__ import annotations

import importlib.util
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .schema import ClipError, Transcript, ass_from_transcript, srt_from_transcript

FIT_FILTER = (
    "[0:v]split=2[bg][fg];"
    "[bg]scale=1080:1920:force_original_aspect_ratio=increase:flags=lanczos,"
    "crop=1080:1920,boxblur=24:2,setsar=1[bgout];"
    "[fg]scale=1080:-2:flags=lanczos,setsar=1[fgout];"
    "[bgout][fgout]overlay=(W-w)/2:(H-h)/2:format=auto"
)

CLEAN_AUDIO_FILTER = (
    "highpass=f=90,lowpass=f=7800,afftdn=nf=-22,"
    "equalizer=f=2800:width_type=h:width=1200:g=4.5,"
    "acompressor=threshold=-22dB:ratio=4:attack=5:release=60:makeup=4,"
    "dynaudnorm=f=150:g=15:p=0.90:m=12.0"
)

VOCAL_BOOST_FILTER = (
    "highpass=f=100,lowpass=f=7500,volume=6dB,"
    "equalizer=f=2600:width_type=h:width=1400:g=5.0,"
    "equalizer=f=4500:width_type=h:width=1500:g=3.0,"
    "acompressor=threshold=-24dB:ratio=5:attack=4:release=50:makeup=5,"
    "dynaudnorm=f=100:g=11:p=0.92:m=15.0,"
    "alimiter=limit=0.95:attack=2:release=25"
)

CLEAN_RENDER_FILTER = VOCAL_BOOST_FILTER
STUDIO_AUDIO_FILTER = "highpass=f=75,loudnorm=I=-15:TP=-1.5:LRA=11,alimiter=limit=0.95"
NOISE_FLOOR_DB = -45.0
DEMUCS_AMBIENT_LEVEL = 0.12

ENHANCE_FAST_FILTER = (
    "hqdn3d=4.0:3.0:6.0:4.5,cas=0.45,eq=contrast=1.04:saturation=1.06"
)

REALESRGAN_BINARY = "realesrgan-ncnn-vulkan"
VAAPI_DEVICE = Path("/dev/dri/renderD128")
SFX_NAMES = ("whoosh.wav", "pop.wav", "ding.wav", "bass_drop.wav")
SFX_LEVEL_DB = -14
PROGRESS_Y = 0
PROGRESS_HEIGHT = 6
DIAGRAM_CONTEXT_SECONDS = 2.5
DIAGRAM_CLOSE_SECONDS = 3.5
DIAGRAM_ZOOM = 1.75
WEBCAM_WIDTH = 440
WEBCAM_HEIGHT = 260
WEBCAM_Y = 150
DIAGRAM_Y = 440
DIAGRAM_HEIGHT = 960
DIAGRAM_BORDER_COLOR = "0x91EBCB@0.95"


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


def extract_audio(
    video: Path,
    workdir: Path,
    *,
    sample_rate: int = 16000,
    filename: str = "audio_16k.wav",
    start: float | None = None,
    duration: float | None = None,
) -> Path:
    require_binary("ffmpeg")
    audio = workdir / filename
    command = [
        "ffmpeg",
        "-nostdin",
        "-y",
        "-loglevel",
        "error",
    ]
    if start is not None:
        command += ["-ss", f"{start:.3f}"]
    if duration is not None:
        command += ["-t", f"{duration:.3f}"]
    command += [
        "-i",
        str(video),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-c:a",
        "pcm_s16le",
        str(audio),
    ]
    run(command, description="extracción de audio")
    return audio


def detect_noise_floor(source: Path, *, seconds: float = 1.0) -> float:
    require_binary("ffmpeg")
    result = run(
        [
            "ffmpeg",
            "-nostdin",
            "-y",
            "-loglevel",
            "info",
            "-ss",
            "0",
            "-t",
            f"{max(0.1, seconds):.3f}",
            "-i",
            str(source),
            "-vn",
            "-af",
            "volumedetect",
            "-f",
            "null",
            "-",
        ],
        capture=True,
        description="volumedetect",
    )
    output = f"{result.stderr or ''}\n{result.stdout or ''}"
    for key in ("mean_volume", "max_volume"):
        match = re.search(rf"{key}:\s*(-?\d+(?:\.\d+)?)\s*dB", output)
        if match:
            return float(match.group(1))
    return -60.0


def resolve_audio_mode(mode: str, noise_floor: float | None = None) -> str:
    if mode not in {"auto", "studio", "noisy"}:
        raise ClipError(f"Modo de audio desconocido: {mode}")
    if mode != "auto":
        return mode
    if noise_floor is None:
        return "noisy"
    return "studio" if noise_floor < NOISE_FLOOR_DB else "noisy"


def render_audio_filter(mode: str, vocal_boost: bool = True) -> str:
    if not vocal_boost:
        return "anull"
    return STUDIO_AUDIO_FILTER if mode == "studio" else VOCAL_BOOST_FILTER


def build_clean_audio_command(
    video_path: Path,
    output_wav: Path,
    *,
    duration: float | None = None,
    audio_mode: str = "noisy",
) -> list[str]:
    command = [
        "ffmpeg",
        "-nostdin",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-af",
        STUDIO_AUDIO_FILTER if audio_mode == "studio" else CLEAN_AUDIO_FILTER,
        "-c:a",
        "pcm_s16le",
    ]
    if duration is not None:
        command += ["-t", f"{duration:.3f}"]
    command.append(str(output_wav))
    return command


def _find_external_cleaner() -> str | None:
    for name in ("deepFilter", "deep-filter"):
        if shutil.which(name):
            return name
    return None


def _run_external_cleaner(binary: str, audio: Path, workdir: Path) -> Path | None:
    outdir = workdir / "deepfilter"
    outdir.mkdir(parents=True, exist_ok=True)
    run([binary, str(audio), "-o", str(outdir)], description="DeepFilter")
    candidates = sorted(path for path in outdir.rglob("*.wav") if path.resolve() != audio.resolve())
    return candidates[0] if candidates else None


def _run_deepfilter_module(audio: Path, output: Path) -> bool:
    try:
        try:
            from df import enhance, init_df, load_audio, save_audio
        except ImportError:
            from df.enhance import enhance, init_df, load_audio, save_audio

        model, df_state, *_ = init_df()
        samples, _ = load_audio(str(audio), sr=df_state.sr())
        enhanced = enhance(model, df_state, samples, pad=True)
        save_audio(str(output), enhanced, df_state.sr())
    except Exception:
        return False
    return output.is_file()


def _demucs_command(audio: Path, outdir: Path) -> list[str] | None:
    arguments = ["--two-stems=vocals", "-o", str(outdir), str(audio)]
    binary = shutil.which("demucs")
    if binary:
        return [binary, *arguments]
    if importlib.util.find_spec("demucs") is not None:
        return [sys.executable, "-m", "demucs", *arguments]
    return None


def build_demucs_mix_command(vocals: Path, no_vocals: Path, output: Path) -> list[str]:
    filter_complex = (
        f"[0:a][1:a]amix=inputs=2:duration=first:weights=1 {DEMUCS_AMBIENT_LEVEL}:normalize=0,"
        "alimiter=limit=0.95:attack=2:release=25[mixed]"
    )
    return [
        "ffmpeg",
        "-nostdin",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(vocals),
        "-i",
        str(no_vocals),
        "-filter_complex",
        filter_complex,
        "-map",
        "[mixed]",
        "-ac",
        "2",
        "-ar",
        "48000",
        "-c:a",
        "pcm_s16le",
        str(output),
    ]


def _isolate_with_demucs(audio: Path, workdir: Path) -> Path | None:
    outdir = workdir / "demucs"
    outdir.mkdir(parents=True, exist_ok=True)
    command = _demucs_command(audio, outdir)
    if not command:
        return None
    run(command, description="demucs")
    candidates = sorted(outdir.rglob("*vocals*.wav"))
    return candidates[0] if candidates else None


def _isolate_segment_vocals(audio: Path, workdir: Path) -> Path | None:
    outdir = workdir / "demucs_segmento"
    outdir.mkdir(parents=True, exist_ok=True)
    command = _demucs_command(audio, outdir)
    if not command:
        return None
    run(command, description="demucs del segmento")
    candidates = list(outdir.rglob("*.wav"))
    vocals = next((path for path in candidates if path.name == "vocals.wav"), None)
    no_vocals = next((path for path in candidates if path.name == "no_vocals.wav"), None)
    if not vocals or not no_vocals:
        return None
    mixed = workdir / "voz_con_ambiente.wav"
    run(build_demucs_mix_command(vocals, no_vocals, mixed), description="mezcla voz/ambiente")
    return mixed


def preparar_audio_limpio(
    video_path: Path,
    output_wav: Path,
    *,
    isolate_vocals: bool = False,
    audio_mode: str = "noisy",
    workdir: Path | None = None,
    warnings: list[str] | None = None,
    duration: float | None = None,
) -> Path:
    require_binary("ffmpeg")
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    workdir = workdir or output_wav.parent
    workdir.mkdir(parents=True, exist_ok=True)
    duration = probe_duration(video_path) if duration is None else duration
    source = video_path

    if audio_mode == "studio":
        if isolate_vocals and warnings is not None:
            warnings.append("Modo studio: se omitió el aislamiento con demucs para conservar naturalidad.")
        isolate_vocals = False
    if isolate_vocals:
        raw = extract_audio(video_path, workdir)
        isolated = _isolate_with_demucs(raw, workdir)
        if isolated:
            source = isolated
            if warnings is not None:
                warnings.append("Voz aislada con demucs antes de la limpieza FFmpeg.")
        elif warnings is not None:
            warnings.append("demucs no está instalado; se usó la cadena FFmpeg mejorada.")
    elif audio_mode != "studio":
        binary = _find_external_cleaner()
        if binary:
            raw = extract_audio(video_path, workdir, sample_rate=48000, filename="audio_48k.wav")
            enhanced = _run_external_cleaner(binary, raw, workdir)
            if enhanced:
                source = enhanced
                if warnings is not None:
                    warnings.append(f"Reducción de ruido aplicada con {binary}.")
        elif importlib.util.find_spec("df") is not None:
            raw = extract_audio(video_path, workdir, sample_rate=48000, filename="audio_48k.wav")
            enhanced_path = workdir / "audio_deepfilter.wav"
            if _run_deepfilter_module(raw, enhanced_path):
                source = enhanced_path
                if warnings is not None:
                    warnings.append("Reducción de ruido aplicada con DeepFilterNet (Python).")

    run(
        build_clean_audio_command(source, output_wav, duration=duration, audio_mode=audio_mode),
        description="limpieza de audio",
    )
    return output_wav


def _probe_video_field(path: Path, field: str) -> str:
    require_binary("ffprobe")
    result = run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            f"stream={field}",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture=True,
        description="ffprobe video",
    )
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        raise ClipError(f"No se pudo leer {field} de {path}")
    return lines[0]


def probe_video_height(path: Path) -> int:
    try:
        return int(_probe_video_field(path, "height"))
    except ValueError as exc:
        raise ClipError(f"Altura de video inválida en {path}") from exc


def probe_video_dimensions(path: Path) -> tuple[int, int]:
    try:
        width = int(_probe_video_field(path, "width"))
        height = int(_probe_video_field(path, "height"))
    except ValueError as exc:
        raise ClipError(f"Dimensiones de video inválidas en {path}") from exc
    if width <= 0 or height <= 0:
        raise ClipError(f"Dimensiones de video inválidas en {path}")
    return width, height


def is_vertical(path: Path) -> bool:
    width, height = probe_video_dimensions(path)
    return height >= width


def probe_video_fps(path: Path) -> float:
    raw = _probe_video_field(path, "r_frame_rate")
    try:
        numerator, denominator = raw.split("/", 1)
        fps = float(numerator) / float(denominator)
    except (ValueError, ZeroDivisionError) as exc:
        raise ClipError(f"FPS de video inválido en {path}") from exc
    if fps <= 0:
        raise ClipError(f"FPS de video inválido en {path}")
    return fps


def resolve_enhance_mode(mode: str, video: Path, height: int | None = None) -> str:
    if mode not in {"auto", "off", "fast", "ai"}:
        raise ClipError(f"Modo de mejora desconocido: {mode}")
    if height is None:
        try:
            height = probe_video_height(video)
        except ClipError:
            height = 0
    if mode == "auto":
        return "fast" if height < 720 else "off"
    if mode == "ai":
        return "ai" if height < 720 else "off"
    return mode


def ffmpeg_supports_encoder(name: str) -> bool:
    require_binary("ffmpeg")
    result = run(["ffmpeg", "-hide_banner", "-encoders"], capture=True, description="ffmpeg encoders")
    return name in f"{result.stdout or ''}\n{result.stderr or ''}"


def resolve_encoder(device: Path | None = None) -> tuple[str, Path | None]:
    device = device or VAAPI_DEVICE
    if device.exists() and ffmpeg_supports_encoder("h264_vaapi"):
        return "h264_vaapi", device
    return "libx264", None


def encoder_options(encoder: str) -> list[str]:
    if encoder == "h264_vaapi":
        return ["-c:v", "h264_vaapi", "-qp", "20", "-pix_fmt", "nv12"]
    return ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p"]


def video_hw_filter(encoder: str) -> str:
    return ",format=nv12,hwupload" if encoder == "h264_vaapi" else ""


def _build_ai_segment(video: Path, start: float, end: float, workdir: Path) -> Path:
    require_binary(REALESRGAN_BINARY)
    frames_in = workdir / "frames_in"
    frames_out = workdir / "frames_out"
    for directory in (frames_in, frames_out):
        directory.mkdir(parents=True, exist_ok=True)
        for old in directory.glob("*.png"):
            old.unlink()
    run(
        [
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
            "-an",
            "-vsync",
            "0",
            str(frames_in / "%06d.png"),
        ],
        description="extracción de frames para restauración IA",
    )
    run(
        [
            REALESRGAN_BINARY,
            "-i",
            str(frames_in),
            "-o",
            str(frames_out),
            "-n",
            "realesrgan-x4plus",
            "-s",
            "2",
            "-f",
            "png",
        ],
        description="realesrgan-ncnn-vulkan",
    )
    fps = probe_video_fps(video)
    intermediate = workdir / "ai_enhanced.mkv"
    run(
        [
            "ffmpeg",
            "-nostdin",
            "-y",
            "-loglevel",
            "error",
            "-framerate",
            f"{fps:.6f}",
            "-i",
            str(frames_out / "%06d.png"),
            "-ss",
            f"{start:.3f}",
            "-t",
            f"{end - start:.3f}",
            "-i",
            str(video),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0?",
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "pcm_s16le",
            str(intermediate),
        ],
        description="reensamble de frames restaurados",
    )
    return intermediate


def _zoom_windows(duration: float, emphasis: list[float], pauses: list[float]) -> list[tuple[float, float]]:
    if duration <= 1.0:
        return []
    candidates = sorted(
        {round(value, 2) for value in [*emphasis, *pauses] if 0.5 <= value <= duration - 1.0}
    )
    chosen: list[float] = []
    for value in candidates:
        if all(abs(value - existing) >= 2.5 for existing in chosen):
            chosen.append(value)
        if len(chosen) >= 4:
            break
    fallback = 3.5
    while len(chosen) < 2 and fallback < duration - 1.0:
        if all(abs(fallback - existing) >= 2.5 for existing in chosen):
            chosen.append(fallback)
        fallback += 7.0
    return [(value, min(3.2, duration)) for value in chosen]


def _max_pairwise(expressions: list[str]) -> str:
    if len(expressions) == 1:
        return expressions[0]
    middle = len(expressions) // 2
    return f"max({_max_pairwise(expressions[:middle])},{_max_pairwise(expressions[middle:])})"


def _microzoom_filter(
    duration: float,
    emphasis: list[float],
    pauses: list[float],
    fps: float = 30.0,
) -> str:
    windows = _zoom_windows(duration, emphasis, pauses)
    if not windows:
        return ""
    terms: list[str] = []
    for center, width in windows:
        half = width / 2
        ramp = width * 0.25
        rise = f"min(1,max(0,(t-{center - half:.3f})/{ramp:.3f}))"
        fall = f"min(1,max(0,({center + half:.3f}-t)/{ramp:.3f}))"
        terms.append(f"({rise}*{fall})")
    combined = _max_pairwise(terms)
    zoom = f"1+min(0.12,{combined})"
    width = f"trunc(iw/({zoom})/2)*2"
    height = f"trunc(ih/({zoom})/2)*2"
    return (
        f"scale=w='{width}':h='{height}':eval=frame:flags=bilinear,"
        "scale=1080:1920:flags=lanczos,setsar=1"
    )


def _progress_filter(duration: float) -> str:
    if duration <= 0:
        return ""
    width = f"min(1080,trunc(1080*t/{duration:.3f}/2)*2)"
    return (
        f"drawbox=x=0:y={PROGRESS_Y}:w=1080:h={PROGRESS_HEIGHT}:color=0x101414@0.55:t=fill,"
        f"drawbox=x=0:y={PROGRESS_Y}:w='{width}':h={PROGRESS_HEIGHT}:color=0x91EBCB@0.9:t=fill"
    )


def _diagram_fit(area_height: int) -> str:
    return (
        f"scale=1080:{area_height}:force_original_aspect_ratio=decrease:flags=lanczos,"
        f"pad=1080:{area_height}:(ow-iw)/2:(oh-ih)/2:color=0x0B0F12"
    )


def _diagram_crop(roi: tuple[float, float, float, float], duration: float, area_height: int) -> str:
    pan_end = max(DIAGRAM_CONTEXT_SECONDS, duration - DIAGRAM_CLOSE_SECONDS)
    span = max(1.0, duration - 6.0)
    pan = f"min(1,max(0,(t-{DIAGRAM_CONTEXT_SECONDS:.3f})/{span:.3f}))"
    active = f"if(lt(t,{DIAGRAM_CONTEXT_SECONDS:.3f}),0,if(lt(t,{pan_end:.3f}),1,0))"
    zoom = f"if(eq({active},1),{DIAGRAM_ZOOM},1)"
    crop_width = f"trunc(iw/({zoom})/2)*2"
    crop_height = f"trunc(ih/({zoom})/2)*2"
    x = f"if(eq({active},1),trunc((iw-({crop_width}))*({pan})/2)*2,0)"
    center_y = roi[1] + roi[3] / 2
    y = (
        f"if(eq({active},1),min(ih-({crop_height}),"
        f"max(0,trunc(({center_y:.5f}*ih-({crop_height})/2)/2)*2)),0)"
    )
    return (
        f"crop=w='{crop_width}':h='{crop_height}':x='{x}':y='{y}',"
        f"scale=1080:{area_height}:flags=lanczos,setsar=1"
    )


def _webcam_crop(corner: str) -> str:
    width = "trunc(iw*0.28/2)*2"
    height = "trunc(ih*0.24/2)*2"
    if corner == "bottom-left":
        x, y = "0", f"ih-({height})"
    elif corner == "top-right":
        x, y = f"iw-({width})", "0"
    elif corner == "top-left":
        x, y = "0", "0"
    else:
        x, y = f"iw-({width})", f"ih-({height})"
    return f"crop=w='{width}':h='{height}':x='{x}':y='{y}'"


def build_video_filter_complex(
    *,
    vertical: bool,
    enhance: str,
    subtitles: Path | None,
    burn_subtitles: bool,
    viral_fx: bool,
    duration: float,
    emphasis: list[float],
    pauses: list[float],
    fps: float,
    diagram_roi: tuple[float, float, float, float] | None,
    split_webcam: bool,
    webcam_corner: str = "none",
    native_vertical: bool = False,
) -> str | None:
    if native_vertical or not vertical or (not diagram_roi and not split_webcam):
        return None
    diagram_area_height = DIAGRAM_HEIGHT if split_webcam else 1920
    diagram_crop = (
        f"{_diagram_fit(diagram_area_height)},{_diagram_crop(diagram_roi, duration, diagram_area_height)}"
        if diagram_roi
        else _diagram_fit(diagram_area_height)
    )
    tail: list[str] = []
    if viral_fx and not diagram_roi:
        microzoom = _microzoom_filter(duration, emphasis, pauses, fps)
        if microzoom:
            tail.append(microzoom)
    if burn_subtitles and subtitles and subtitles.exists():
        tail.append(_ass_filter(subtitles))
    if viral_fx:
        progress = _progress_filter(duration)
        if progress:
            tail.append(progress)
    suffix = "," + ",".join(tail) if tail else ""
    source = f"[0:v]{ENHANCE_FAST_FILTER}," if enhance == "fast" else "[0:v]"
    if split_webcam:
        border_x = (1080 - WEBCAM_WIDTH) // 2
        return ";".join(
            [
                f"color=c=0x0B0F12:s=1080x1920:d={duration:.3f}[canvas]",
                f"{source}split=2[web][diag]",
                f"[web]{_webcam_crop(webcam_corner)},"
                f"scale={WEBCAM_WIDTH}:{WEBCAM_HEIGHT}:flags=lanczos,cas=0.35,setsar=1[webcam]",
                f"[diag]{diagram_crop}[diagram]",
                f"[canvas][diagram]overlay=0:{DIAGRAM_Y}:format=auto[base]",
                f"[base][webcam]overlay=(W-w)/2:{WEBCAM_Y}:format=auto,"
                f"drawbox=x={border_x}:y={WEBCAM_Y}:w={WEBCAM_WIDTH}:h={WEBCAM_HEIGHT}:"
                f"color={DIAGRAM_BORDER_COLOR}:t=2{suffix},setsar=1[vout]",
            ]
        )
    return ";".join(
        [
            f"{source}split=2[bg][fg]",
            "[bg]scale=1080:1920:force_original_aspect_ratio=increase:flags=lanczos,"
            "crop=1080:1920,boxblur=24:2,setsar=1[bgout]",
            f"[fg]{diagram_crop}[fgout]",
            f"[bgout][fgout]overlay=(W-w)/2:(H-h)/2:format=auto{suffix},setsar=1[vout]",
        ]
    )


def build_video_filter(
    *,
    vertical: bool,
    crop: tuple[int, int, int, int] | None,
    framing: str,
    enhance: str,
    subtitles: Path | None,
    burn_subtitles: bool,
    viral_fx: bool,
    duration: float,
    emphasis: list[float],
    pauses: list[float],
    fps: float = 30.0,
    native_vertical: bool = False,
) -> str:
    if native_vertical and vertical:
        parts: list[str] = []
        if enhance == "fast":
            parts.append(ENHANCE_FAST_FILTER)
        parts.append("scale=1080:1920:flags=lanczos,setsar=1")
        graph = ",".join(parts)
        tail: list[str] = []
        if burn_subtitles and subtitles and subtitles.exists():
            tail.append(_ass_filter(subtitles))
        if viral_fx:
            progress = _progress_filter(duration)
            if progress:
                tail.append(progress)
        return f"{graph},{','.join(tail)}" if tail else graph
    if vertical and framing == "fit":
        graph = (
            "split=2[bg][fg];"
            "[bg]scale=1080:1920:force_original_aspect_ratio=increase:flags=lanczos,"
            "crop=1080:1920,boxblur=24:2,setsar=1[bgout];"
            "[fg]scale=1080:-2:flags=lanczos,setsar=1[fgout];"
            "[bgout][fgout]overlay=(W-w)/2:(H-h)/2:format=auto,setsar=1"
        )
    else:
        parts: list[str] = []
        if enhance == "fast":
            parts.append(ENHANCE_FAST_FILTER)
        if vertical:
            if crop:
                x, y, crop_width, crop_height = crop
                parts.append(f"crop={crop_width}:{crop_height}:{x}:{y}")
            parts.append(
                "scale=1080:1920:force_original_aspect_ratio=increase:flags=lanczos,crop=1080:1920,setsar=1"
            )
        graph = ",".join(parts) if parts else "null"

    tail: list[str] = []
    if viral_fx and vertical:
        microzoom = _microzoom_filter(duration, emphasis, pauses, fps)
        if microzoom:
            tail.append(microzoom)
    if burn_subtitles and subtitles and subtitles.exists():
        tail.append(_ass_filter(subtitles))
    if viral_fx and vertical:
        progress = _progress_filter(duration)
        if progress:
            tail.append(progress)
    return f"{graph},{','.join(tail)}" if tail else graph


def find_sfx_dir(explicit: Path | None = None) -> Path | None:
    candidates = [explicit] if explicit else [Path.cwd() / "assets" / "sfx", Path(__file__).resolve().parent.parent / "assets" / "sfx"]
    for candidate in candidates:
        if candidate and candidate.is_dir():
            return candidate
    return None


def build_sfx_schedule(
    directory: Path | None,
    duration: float,
    style: str,
    emphasis: list[float],
    pauses: list[float],
) -> list[tuple[Path, float]]:
    if not directory or not directory.is_dir():
        return []
    available = {path.name.lower(): path for path in directory.glob("*.wav")}
    schedule: list[tuple[Path, float]] = []

    def add(name: str, time: float) -> None:
        path = available.get(name)
        if path and 0 <= time <= max(0.0, duration - 0.1):
            schedule.append((path, round(time, 3)))

    add("whoosh.wav", 0.05)
    for index, value in enumerate(emphasis[:2]):
        add("pop.wav" if index == 0 else "ding.wav", value)
    if not emphasis and "ding.wav" in available:
        add("ding.wav", min(2.0, duration / 2))
    if style in {"humor", "debate"} and pauses:
        add("bass_drop.wav", max(0.0, pauses[0] - 0.4))
    seen: set[tuple[str, float]] = set()
    return [item for item in schedule if not (item[0].name, item[1]) in seen and not seen.add((item[0].name, item[1]))]


def build_sfx_filter(
    schedule: list[tuple[Path, float]],
    voice_input: int,
    vocal_boost: bool,
    render_filter: str | None = None,
) -> str:
    filters: list[str] = []
    voice_chain = render_filter or (VOCAL_BOOST_FILTER if vocal_boost else "anull")
    voice = f"[{voice_input}:a]{voice_chain}[voice]"
    filters.append(voice)
    labels = ["[voice]"]
    for index, (_, at) in enumerate(schedule):
        delay_ms = int(round(at * 1000))
        filters.append(f"[{voice_input + 1 + index}:a]adelay={delay_ms}|{delay_ms},volume={SFX_LEVEL_DB}dB[sfx{index}]")
        labels.append(f"[sfx{index}]")
    filters.append(f"{''.join(labels)}amix=inputs={len(labels)}:duration=first:normalize=0[outa]")
    return ";".join(filters)


def build_render_command(
    video: Path,
    start: float,
    end: float,
    output: Path,
    vertical: bool,
    *,
    crop: tuple[int, int, int, int] | None = None,
    subtitles: Path | None = None,
    burn_subtitles: bool = True,
    framing: str = "auto",
    clean_audio: bool | None = None,
    vocal_boost: bool = True,
    audio_source: Path | None = None,
    audio_seek: float | None = None,
    enhance: str = "off",
    viral_fx: bool = True,
    duration: float | None = None,
    emphasis: list[float] | None = None,
    pauses: list[float] | None = None,
    sfx: list[tuple[Path, float]] | None = None,
    fps: float = 30.0,
    diagram_roi: tuple[float, float, float, float] | None = None,
    split_webcam: bool = False,
    webcam_corner: str = "none",
    audio_mode: str = "auto",
    encoder: str | None = None,
    native_vertical: bool = False,
) -> list[str]:
    if clean_audio is not None:
        vocal_boost = clean_audio
    sfx = sfx or []
    clip_duration = duration if duration is not None else end - start
    if encoder is None:
        encoder, encoder_device = resolve_encoder()
    else:
        encoder_device = VAAPI_DEVICE if encoder == "h264_vaapi" else None
    render_filter = render_audio_filter(audio_mode, vocal_boost)
    hardware = video_hw_filter(encoder)
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
    if encoder == "h264_vaapi" and encoder_device is not None:
        command[2:2] = ["-vaapi_device", str(encoder_device)]
    if audio_source is not None:
        if audio_seek is not None:
            command += ["-ss", f"{max(0.0, audio_seek):.3f}", "-t", f"{max(0.0, clip_duration):.3f}"]
        command += ["-i", str(audio_source)]
    for path, _ in sfx:
        command += ["-i", str(path)]
    voice_input = 1 if audio_source is not None else 0
    complex_video = build_video_filter_complex(
        vertical=vertical,
        enhance=enhance,
        subtitles=subtitles,
        burn_subtitles=burn_subtitles,
        viral_fx=viral_fx,
        duration=clip_duration,
        emphasis=emphasis or [],
        pauses=pauses or [],
        fps=fps,
        diagram_roi=diagram_roi,
        split_webcam=split_webcam,
        webcam_corner=webcam_corner,
        native_vertical=native_vertical,
    )
    if complex_video:
        if hardware:
            complex_video = complex_video.replace("[vout]", f"{hardware}[vout]")
        graphs = [complex_video]
        if sfx:
            graphs.append(build_sfx_filter(sfx, voice_input, vocal_boost, render_filter))
        else:
            graphs.append(f"[{voice_input}:a]{render_filter}[outa]")
        command += ["-filter_complex", ";".join(graphs), "-map", "[vout]", "-map", "[outa]"]
    else:
        command += ["-map", "0:v:0"]
        video_filter = build_video_filter(
            vertical=vertical,
            crop=crop,
            framing=framing,
            enhance=enhance,
            subtitles=subtitles,
            burn_subtitles=burn_subtitles,
            viral_fx=viral_fx,
            duration=clip_duration,
            emphasis=emphasis or [],
            pauses=pauses or [],
            fps=fps,
            native_vertical=native_vertical,
        )
        if video_filter:
            command += ["-vf", f"{video_filter}{hardware}"]
        elif hardware:
            command += ["-vf", hardware.lstrip(",")]
        if sfx:
            command += ["-filter_complex", build_sfx_filter(sfx, voice_input, vocal_boost, render_filter)]
            command += ["-map", "[outa]"]
        else:
            command += ["-map", "1:a:0" if audio_source is not None else "0:a:0?"]
            if render_filter != "anull":
                command += ["-af", render_filter]
    command += ["-ac", "2", "-ar", "48000"]
    command += encoder_options(encoder)
    command += [
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-movflags",
        "+faststart",
        str(output),
    ]
    return command


def render_clip(
    video: Path,
    start: float,
    end: float,
    output: Path,
    vertical: bool,
    *,
    crop: tuple[int, int, int, int] | None = None,
    subtitles: Path | None = None,
    burn_subtitles: bool = True,
    framing: str = "auto",
    clean_audio: bool | None = None,
    vocal_boost: bool = True,
    audio_source: Path | None = None,
    audio_seek: float | None = None,
    isolate_vocals: bool | None = None,
    enhance: str = "off",
    viral_fx: bool = True,
    emphasis: list[float] | None = None,
    pauses: list[float] | None = None,
    style: str = "auto",
    sfx_dir: Path | None = None,
    verbose: bool = False,
    fps: float | None = None,
    diagram_roi: tuple[float, float, float, float] | None = None,
    split_webcam: bool = False,
    webcam_corner: str = "none",
    audio_mode: str = "auto",
    encoder: str | None = None,
    native_vertical: bool = False,
    workdir: Path | None = None,
    warnings: list[str] | None = None,
) -> None:
    require_binary("ffmpeg")
    output.parent.mkdir(parents=True, exist_ok=True)
    source = video
    render_start, render_end = start, end
    if enhance == "ai":
        cache = workdir / "enhance_ai" if workdir else output.parent / ".enhance_ai"
        if shutil.which(REALESRGAN_BINARY):
            try:
                source = _build_ai_segment(video, start, end, cache)
                render_start, render_end = 0.0, end - start
                enhance = "off"
            except (ClipError, OSError) as exc:
                if warnings is not None:
                    warnings.append(f"Restauración IA falló ({exc}); se usó mejora rápida FFmpeg.")
                enhance = "fast"
        else:
            if warnings is not None:
                warnings.append("realesrgan-ncnn-vulkan no está instalado; se usó mejora rápida FFmpeg.")
            enhance = "fast"

    resolved_audio_mode = resolve_audio_mode(audio_mode)
    if audio_source is None and resolved_audio_mode != "studio" and isolate_vocals is not False:
        clip_workdir = (workdir or output.parent) / f"audio_{output.stem}"
        clip_workdir.mkdir(parents=True, exist_ok=True)
        segment_audio = extract_audio(
            source,
            clip_workdir,
            sample_rate=48000,
            filename="segmento.wav",
            start=render_start,
            duration=render_end - render_start,
        )
        try:
            isolated = _isolate_segment_vocals(segment_audio, clip_workdir)
        except (ClipError, OSError) as exc:
            isolated = None
            if warnings is not None:
                warnings.append(f"Aislamiento con demucs falló ({exc}); se usó audio original.")
        if isolated:
            audio_source = isolated
            if warnings is not None:
                warnings.append("Voz aislada con demucs: 100 % voz + 12 % ambiente.")
        else:
            binary = _find_external_cleaner()
            if binary:
                enhanced = _run_external_cleaner(binary, segment_audio, clip_workdir)
                if enhanced:
                    audio_source = enhanced
                    if warnings is not None:
                        warnings.append(f"Reducción de ruido aplicada con {binary} al segmento.")
            elif importlib.util.find_spec("df") is not None:
                enhanced_path = clip_workdir / "segmento_deepfilter.wav"
                if _run_deepfilter_module(segment_audio, enhanced_path):
                    audio_source = enhanced_path
                    if warnings is not None:
                        warnings.append("DeepFilterNet aplicado al segmento.")
            if audio_source is None and isolate_vocals is True and warnings is not None:
                warnings.append("demucs/DeepFilterNet no disponibles; se aplicó solo la cadena FFmpeg.")

    clip_duration = max(0.0, render_end - render_start)
    render_fps = fps
    if render_fps is None:
        try:
            render_fps = probe_video_fps(source)
        except ClipError:
            render_fps = 30.0
    emphasis_times = emphasis or []
    pause_times = pauses or []
    sfx_schedule = build_sfx_schedule(
        find_sfx_dir(sfx_dir),
        clip_duration,
        style,
        emphasis_times,
        pause_times,
    ) if viral_fx else []
    options: dict[str, Any] = dict(
        crop=crop,
        subtitles=subtitles,
        burn_subtitles=burn_subtitles,
        framing=framing,
        clean_audio=clean_audio,
        vocal_boost=vocal_boost,
        audio_source=audio_source,
        audio_seek=audio_seek,
        enhance=enhance,
        viral_fx=viral_fx,
        duration=clip_duration,
        emphasis=emphasis_times,
        pauses=pause_times,
        sfx=sfx_schedule,
        fps=render_fps,
        diagram_roi=diagram_roi,
        split_webcam=split_webcam,
        webcam_corner=webcam_corner,
        audio_mode=resolved_audio_mode,
        encoder=encoder,
        native_vertical=native_vertical,
    )
    command = build_render_command(source, render_start, render_end, output, vertical, **options)
    if verbose:
        print(f"FFmpeg: {shlex.join(command)}")
    try:
        run(command, description=f"render {output.name}")
    except ClipError as exc:
        if encoder != "h264_vaapi":
            raise
        if warnings is not None:
            warnings.append(f"VA-API falló ({exc}); se usó libx264 veryfast.")
        options["encoder"] = "libx264"
        command = build_render_command(source, render_start, render_end, output, vertical, **options)
        if verbose:
            print(f"FFmpeg fallback: {shlex.join(command)}")
        run(command, description=f"render {output.name} (fallback)")


def write_srt(transcript: Transcript, start: float, end: float, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(srt_from_transcript(transcript, start, end), encoding="utf-8")


def write_ass(
    transcript: Transcript,
    start: float,
    end: float,
    path: Path,
    *,
    hook: str | None = None,
    title: str = "",
    viral_fx: bool = True,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        ass_from_transcript(transcript, start, end, hook=hook, title=title, viral_fx=viral_fx),
        encoding="utf-8",
    )


def safe_stem(name: str) -> str:
    stem = re.sub(r"[^a-zA-Z0-9_-]+", "_", Path(name).stem).strip("_")
    return stem or "video"


def _ass_filter(subtitles: Path) -> str:
    escaped = str(subtitles.resolve()).replace("\\", "\\\\").replace("'", "\\'")
    return f"ass='{escaped}'"
