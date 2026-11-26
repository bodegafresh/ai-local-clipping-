from __future__ import annotations

import argparse
import json
import shutil
from concurrent.futures import ThreadPoolExecutor
import sys
import tempfile
from pathlib import Path
from time import perf_counter
from typing import Any

from .audio_enrich import enrich_transcript_with_audio
from .framing import find_speaker_crop
from .media import (
    detect_noise_floor,
    download_video,
    extract_audio,
    is_url,
    preparar_audio_limpio,
    probe_duration,
    probe_video_dimensions,
    probe_video_fps,
    render_clip,
    resolve_audio_mode,
    resolve_encoder,
    resolve_enhance_mode,
    safe_stem,
    write_ass,
    write_srt,
)
from .ollama import OllamaClient, analyze_transcript
from .schema import (
    ClipError,
    ClipSettings,
    Transcript,
    build_output,
    clip_metadata,
    detect_gaming_style,
    heuristic_candidates,
    normalize_source_url,
    viral_highlight_times,
    write_clip_text,
    write_json,
)
from .transcribe import transcribe_audio
from .vision import detect_layout_cv, find_vision_model, refine_candidates


class StageTimer:
    def __init__(self) -> None:
        self.times: dict[str, float] = {}

    def start(self) -> float:
        return perf_counter()

    def mark(self, label: str, started: float) -> float:
        elapsed = round(perf_counter() - started, 3)
        self.times[label] = elapsed
        print(f"Tiempo {label}: {elapsed:.2f}s")
        return elapsed


def resolve_duration_bounds(min_duration: float | None, max_duration: float | None, style: str) -> tuple[float, float]:
    default_min, default_max = (20.0, 42.0) if style == "gaming" else (15.0, 60.0)
    return (
        default_min if min_duration is None else min_duration,
        default_max if max_duration is None else max_duration,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="clips",
        description="Genera clips listos para redes sociales usando Whisper + Ollama.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input", help="Archivo de video/audio local o URL compatible con yt-dlp")
    parser.add_argument(
        "--source-url",
        default=None,
        help="URL original del video para enlazar cada clip con minuto exacto (YouTube/Youtu.be)",
    )
    parser.add_argument("-o", "--output", default="clips.json", help="Archivo JSON de salida")
    parser.add_argument("--clips", type=int, default=5, help="Número máximo de clips")
    parser.add_argument("--min-duration", type=float, default=None, help="Duración mínima de cada clip en segundos (gaming: 20)")
    parser.add_argument("--max-duration", type=float, default=None, help="Duración máxima de cada clip en segundos (gaming: 42)")
    parser.add_argument("--whisper-model", default="small", help="Modelo de Whisper (tiny, base, small, medium, large-v3)")
    parser.add_argument("--whisper-language", default="es", help="Idioma para Whisper o 'auto' para detección automática")
    parser.add_argument("--whisper-device", default=None, help="Dispositivo de Whisper: cpu, cuda, mps")
    parser.add_argument(
        "--whisper-backend",
        choices=("auto", "openai", "faster", "cli"),
        default="auto",
        help="Implementación de Whisper a utilizar",
    )
    parser.add_argument("--ollama-model", default=None, help="Modelo de Ollama (por defecto OLLAMA_MODEL o qwen2.5:7b)")
    parser.add_argument("--ollama-host", default=None, help="URL de Ollama (por defecto OLLAMA_HOST)")
    parser.add_argument("--chunk-chars", type=int, default=12000, help="Tamaño de cada bloque para transcripciones largas")
    parser.add_argument(
        "--style",
        choices=("auto", "humor", "noticias", "vender", "educar", "debate", "gaming"),
        default="auto",
        help="Perfil editorial usado para seleccionar y redactar los clips",
    )
    vision_group = parser.add_mutually_exclusive_group()
    vision_group.add_argument("--vision", action="store_true", help="Forzar verificación visual con Ollama multimodal")
    vision_group.add_argument("--no-vision", action="store_true", help="Desactivar verificación visual")
    parser.add_argument(
        "--vision-model",
        default=None,
        help="Modelo multimodal de Ollama (por defecto detecta qwen3-vl u otro modelo visión)",
    )
    parser.add_argument("--transcript", default=None, help="Reutilizar transcripción JSON en vez de transcribir")
    parser.add_argument("--transcript-out", default=None, help="Guardar la transcripción en este archivo JSON")
    parser.add_argument("--render-dir", default=None, help="Carpeta donde renderizar los clips en MP4 + SRT + TXT + JSON")
    parser.add_argument(
        "--framing",
        choices=("auto", "face", "fit", "center"),
        default="auto",
        help="Encuadre vertical: face=foco en el rostro, fit=cuadro completo con fondo difuminado, center=recorte central",
    )
    parser.add_argument("--no-subtitles", action="store_true", help="No quemar los subtítulos en el video")
    parser.add_argument(
        "--no-clean-audio",
        action="store_true",
        help="No limpiar/normalizar el audio antes de transcribir (la limpieza está activa por defecto)",
    )
    parser.add_argument(
        "--clean-render-audio",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--no-vocal-boost",
        action="store_true",
        help="No aplicar la cadena de alto impacto vocal al exportar los clips",
    )
    parser.add_argument(
        "--isolate-vocals",
        action="store_true",
        help="Forzar aislamiento vocal con demucs durante la preparación del audio",
    )
    parser.add_argument(
        "--no-isolate-vocals",
        action="store_true",
        help="No aislar la voz con demucs aunque esté disponible",
    )
    parser.add_argument(
        "--audio-mode",
        choices=("auto", "studio", "noisy"),
        default="auto",
        help="Perfil de audio: auto mide el piso de ruido, studio conserva naturalidad, noisy realza voz",
    )
    parser.add_argument(
        "--enhance-video",
        choices=("auto", "off", "fast", "ai"),
        default="auto",
        help="Restauración visual: auto activa fast si la fuente mide menos de 720p",
    )
    parser.add_argument("--no-vertical", action="store_true", help="No recortar los clips a formato vertical 9:16")
    parser.add_argument(
        "--viral-fx",
        choices=("on", "off"),
        default="on",
        help="Activar/desactivar hook, subtítulos dinámicos, micro-zoom, barra y SFX",
    )
    parser.add_argument(
        "--diagram-zoom",
        choices=("auto", "on", "off"),
        default="auto",
        help="Pan & Scan guiado sobre diagramas o diapositivas",
    )
    parser.add_argument(
        "--split-webcam",
        choices=("auto", "on", "off"),
        default="auto",
        help="Split-screen con la webcam del presentador en la esquina",
    )
    parser.add_argument("--verbose", action="store_true", help="Imprimir el comando FFmpeg de cada clip")
    parser.add_argument("--name", default=None, help="Nombre base para los archivos renderizados")
    parser.add_argument("--keep-workdir", action="store_true", help="Conservar el audio temporal para depuración")
    return parser


def _render_one_clip(
    args: argparse.Namespace,
    index: int,
    clip: Any,
    video: Path,
    transcript: Transcript,
    render_dir: Path,
    workdir: Path,
    enhance_mode: str,
    audio_mode: str,
    encoder: str,
    audio_source: Path | None,
    audio_seek: float | None,
    source_fps: float | None,
    source_url: str | None,
    style: str,
    native_vertical: bool,
) -> tuple[tuple[int, int, int, int] | None, str, list[str], dict[str, float]]:
    local_warnings: list[str] = []
    timings: dict[str, float] = {}
    crop = None
    framing_mode = args.framing
    started = perf_counter()
    if native_vertical and not args.no_vertical:
        crop = None
        framing_mode = "center"
    elif not args.no_vertical and args.framing in {"auto", "face"}:
        detection = find_speaker_crop(video, clip.start, clip.end)
        if detection:
            crop, confidence = detection
            if args.framing == "auto" and confidence < 0.25:
                crop = None
                framing_mode = "fit"
        elif args.framing == "face":
            framing_mode = "fit"
            local_warnings.append(f"No se detectó rostro en clip_{index:02d}; se usó encuadre completo.")
    timings["face"] = round(perf_counter() - started, 3)
    if not native_vertical and args.framing == "fit":
        framing_mode = "fit"
    if not native_vertical and args.framing == "center":
        framing_mode = "center"
        crop = None

    srt_path = render_dir / f"clip_{index:02d}.srt"
    ass_path = render_dir / f"clip_{index:02d}.ass"
    viral_fx = args.viral_fx == "on"
    diagram_roi = clip.diagram_roi
    webcam_corner = clip.webcam_corner or "none"
    started = perf_counter()
    if native_vertical or args.no_vertical:
        diagram_roi = None
        webcam_corner = "none"
    elif args.diagram_zoom != "off" and diagram_roi is None:
        layout = detect_layout_cv(video, clip.start, clip.end, workdir / "diagramas", index)
        diagram_roi = layout.roi if layout.has_diagram else None
        if webcam_corner == "none":
            webcam_corner = layout.webcam_corner
    timings["layout_cv"] = round(perf_counter() - started, 3)
    if not native_vertical and args.diagram_zoom == "on" and diagram_roi is None:
        diagram_roi = (0.05, 0.30, 0.90, 0.40)
    if args.diagram_zoom == "off":
        diagram_roi = None
    if not native_vertical and diagram_roi and args.diagram_zoom == "auto":
        local_warnings.append(f"Diagrama detectado en clip_{index:02d}; se aplicó Pan & Scan.")
    split_webcam = False
    if not native_vertical and args.split_webcam == "on":
        split_webcam = True
    elif not native_vertical and args.split_webcam == "auto" and webcam_corner != "none":
        split_webcam = True
    if split_webcam and webcam_corner == "none":
        webcam_corner = "bottom-right"

    emphasis, pauses = viral_highlight_times(transcript, clip.start, clip.end)
    started = perf_counter()
    write_srt(transcript, clip.start, clip.end, srt_path)
    write_ass(
        transcript,
        clip.start,
        clip.end,
        ass_path,
        hook=clip.hook_visual_pantalla,
        title=clip.titulo_corto or clip.title,
        viral_fx=viral_fx,
    )
    timings["subtitulos"] = round(perf_counter() - started, 3)
    started = perf_counter()
    render_clip(
        video,
        clip.start,
        clip.end,
        render_dir / f"clip_{index:02d}.mp4",
        vertical=not args.no_vertical,
        crop=crop,
        subtitles=ass_path,
        burn_subtitles=not args.no_subtitles,
        framing=framing_mode,
        vocal_boost=not args.no_vocal_boost,
        audio_source=audio_source,
        audio_seek=audio_seek if audio_source is not None else None,
        isolate_vocals=None if args.no_isolate_vocals else (args.isolate_vocals or None),
        enhance=enhance_mode,
        viral_fx=viral_fx,
        emphasis=emphasis,
        pauses=pauses,
        style=style,
        verbose=args.verbose,
        fps=source_fps,
        diagram_roi=diagram_roi,
        split_webcam=split_webcam,
        webcam_corner=webcam_corner,
        audio_mode=audio_mode,
        encoder=encoder,
        native_vertical=native_vertical and not args.no_vertical,
        workdir=workdir,
        warnings=local_warnings,
    )
    timings["ffmpeg_render"] = round(perf_counter() - started, 3)
    started = perf_counter()
    write_clip_text(render_dir / f"clip_{index:02d}.txt", clip, source_url)
    write_json(render_dir / f"clip_{index:02d}.json", clip_metadata(clip, index, source_url))
    timings["metadata"] = round(perf_counter() - started, 3)
    return crop, framing_mode, local_warnings, timings


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    workdir = Path(tempfile.mkdtemp(prefix="ai-local-clipping-"))
    timer = StageTimer()
    started = timer.start()
    try:
        return _run(args, workdir, timer)
    except ClipError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Cancelado.", file=sys.stderr)
        return 130
    finally:
        timer.mark("total", started)
        if args.keep_workdir:
            print(f"Directorio temporal: {workdir}")
        else:
            shutil.rmtree(workdir, ignore_errors=True)


def _run(args: argparse.Namespace, workdir: Path, timer: StageTimer | None = None) -> int:
    timer = timer or StageTimer()
    if args.chunk_chars < 1000:
        raise ClipError("--chunk-chars debe ser al menos 1000")
    if args.isolate_vocals and args.no_isolate_vocals:
        raise ClipError("--isolate-vocals y --no-isolate-vocals son incompatibles")

    url: str | None = None
    if is_url(args.input):
        url = args.input
        print(f"Descargando video: {url}")
        started = timer.start()
        video = download_video(url, workdir)
        timer.mark("descarga", started)
        source = url
    else:
        video = Path(args.input).expanduser().resolve()
        if not video.is_file():
            raise ClipError(f"No existe el archivo: {video}")
        source = str(video)
    source_url = normalize_source_url(args.source_url or url)
    if source_url:
        print(f"Video original: {source_url}")

    started = timer.start()
    duration = probe_duration(video)
    timer.mark("duracion", started)
    print(f"Video: {video.name} ({duration:.1f} s)")

    try:
        source_width, source_height = probe_video_dimensions(video)
    except ClipError:
        source_width, source_height = 0, 0
    native_vertical = source_height >= source_width > 0

    warnings: list[str] = []
    started = timer.start()
    enhance_mode = resolve_enhance_mode(args.enhance_video, video, height=source_height or None)
    noise_floor = detect_noise_floor(video)
    audio_mode = resolve_audio_mode(args.audio_mode, noise_floor)
    if native_vertical:
        audio_mode = "studio"
    encoder, encoder_device = resolve_encoder()
    try:
        source_fps = probe_video_fps(video)
    except ClipError:
        source_fps = None
    timer.mark("perfil", started)
    print(
        f"Perfil: audio={audio_mode} (piso {noise_floor:.1f} dB), "
        f"encoder={encoder}, enhance_video={enhance_mode}, vertical_native={native_vertical}"
    )
    if (args.audio_mode == "studio" or native_vertical) and args.isolate_vocals:
        warnings.append("Modo studio ignoró --isolate-vocals para evitar artefactos.")
    if encoder_device is not None:
        print(f"VA-API: {encoder_device}")

    audio: Path | None = None
    if args.transcript:
        transcript_path = Path(args.transcript).expanduser().resolve()
        if not transcript_path.is_file():
            raise ClipError(f"No existe la transcripción: {transcript_path}")
        print(f"Reutilizando transcripción: {transcript_path}")
        transcript_data = json.loads(transcript_path.read_text(encoding="utf-8"))
        transcript = Transcript.from_dict(transcript_data)
    else:
        started = timer.start()
        if args.no_clean_audio:
            audio = extract_audio(video, workdir)
        else:
            audio = preparar_audio_limpio(
                video,
                workdir / "audio_limpio.wav",
                isolate_vocals=args.isolate_vocals,
                audio_mode=audio_mode,
                workdir=workdir,
                warnings=warnings,
                duration=duration,
            )
            print(f"Audio limpio: {audio.name}")
        timer.mark("audio_prep", started)
        print("Transcribiendo con Whisper...")
        started = timer.start()
        language = None if args.whisper_language.lower() == "auto" else args.whisper_language
        transcript = transcribe_audio(
            audio,
            workdir,
            model_name=args.whisper_model,
            language=language,
            device=args.whisper_device,
            backend=args.whisper_backend,
        )
        timer.mark("whisper", started)

    if audio is None:
        started = timer.start()
        if args.no_clean_audio:
            audio = extract_audio(video, workdir)
        else:
            audio = preparar_audio_limpio(
                video,
                workdir / "audio_limpio.wav",
                isolate_vocals=args.isolate_vocals,
                audio_mode=audio_mode,
                workdir=workdir,
                warnings=warnings,
                duration=duration,
            )
        timer.mark("audio_prep_enrich", started)
    started = timer.start()
    enrichment = enrich_transcript_with_audio(transcript, audio)
    timer.mark("audio_enrich", started)
    transcript = enrichment.transcript
    print("Enriquecimiento acústico aplicado")
    transcript.duration = duration
    if args.transcript_out:
        transcript_out = Path(args.transcript_out).expanduser().resolve()
        write_json(transcript_out, transcript.to_dict())
        print(f"Transcripción guardada: {transcript_out}")

    effective_style = args.style
    if args.style == "auto" and detect_gaming_style(transcript, native_vertical):
        effective_style = "gaming"
        print("Estilo detectado automáticamente: gaming")
    min_duration, max_duration = resolve_duration_bounds(args.min_duration, args.max_duration, effective_style)
    settings = ClipSettings(args.clips, min_duration, max_duration)
    settings.validate()

    client = OllamaClient(args.ollama_host, args.ollama_model)
    print(f"Analizando con Ollama ({client.model})...")
    started = timer.start()
    client.check_model()
    timer.mark("ollama_model_check", started)

    vision_model: str | None = None
    if not args.no_vision:
        started = timer.start()
        try:
            vision_model = find_vision_model(client, args.vision_model)
        except ClipError as exc:
            warnings.append(f"No se pudo detectar modelo multimodal: {exc}")
        if vision_model:
            print(f"Verificación visual: {vision_model}")
        elif args.vision:
            warnings.append("Se pidió --vision pero no hay ningún modelo multimodal instalado en Ollama.")
        timer.mark("vision_model_check", started)

    started = timer.start()
    summary, candidates = analyze_transcript(
        client,
        transcript,
        settings,
        chunk_chars=args.chunk_chars,
        style=effective_style,
    )
    timer.mark("ollama_texto", started)
    if effective_style == "gaming":
        summary["estilo_aplicado"] = "gaming"

    if not candidates:
        warnings.append("Ollama no devolvió clips válidos; se usó selección heurística.")
        candidates = heuristic_candidates(transcript, settings, style=effective_style)
        summary["estilo_aplicado"] = effective_style
        if not summary.get("general"):
            summary["general"] = "Resumen no disponible: se seleccionaron los fragmentos automáticamente."
    if not candidates:
        raise ClipError("No se pudieron generar clips con la transcripción disponible")

    started = timer.start()
    candidates = refine_candidates(
        client,
        vision_model,
        video,
        candidates,
        transcript,
        effective_style,
        settings,
        workdir / "vision",
        peaks=enrichment.peaks,
        warnings=warnings,
    )
    timer.mark("vision_llm", started)

    render_dir = Path(args.render_dir).expanduser().resolve() if args.render_dir else None
    crops: list[tuple[int, int, int, int] | None] = []
    framing_modes: list[str] = []
    render_details: list[dict[str, float]] = []
    if render_dir:
        render_dir.mkdir(parents=True, exist_ok=True)
        print(f"Renderizando {len(candidates)} clips en {render_dir} (2 en paralelo)")
        started = timer.start()
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(
                    _render_one_clip,
                    args,
                    index,
                    clip,
                    video,
                    transcript,
                    render_dir,
                    workdir,
                    enhance_mode,
                    audio_mode,
                    encoder,
                    audio,
                    clip.start if audio is not None else None,
                    source_fps,
                    source_url,
                    effective_style,
                    native_vertical,
                )
                for index, clip in enumerate(candidates, start=1)
            ]
            results = [future.result() for future in futures]
        timer.mark("render_total", started)
        for crop, framing_mode, clip_warnings, clip_timings in results:
            crops.append(crop)
            framing_modes.append(framing_mode)
            warnings.extend(clip_warnings)
            render_details.append(clip_timings)

    output = build_output(
        source=source,
        source_path=video if not url else None,
        url=url,
        duration=duration,
        transcript=transcript,
        summary=summary,
        clips=candidates,
        whisper_model=args.whisper_model,
        ollama_model=client.model,
        render_dir=render_dir,
        video_name=args.name or safe_stem(video.name),
        warnings=warnings,
        crops=crops,
        subtitles_burned=bool(render_dir) and not args.no_subtitles,
        framing_modes=framing_modes,
        style=effective_style,
        source_url=source_url,
        timings=timer.times,
        render_details=render_details,
    )
    output_path = Path(args.output).expanduser().resolve()
    started = timer.start()
    write_json(output_path, output)
    timer.mark("json_output", started)
    print(f"Listo: {output_path} ({len(candidates)} clips)")
    for index, clip in enumerate(candidates, start=1):
        print(f"  {index:02d}. {clip.start:.1f}-{clip.end:.1f}s  {clip.title}")
    return 0
