from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

from .framing import find_speaker_crop
from .media import download_video, extract_audio, is_url, probe_duration, render_clip, safe_stem, write_srt
from .ollama import OllamaClient, analyze_transcript
from .schema import ClipError, ClipSettings, Transcript, build_output, heuristic_candidates, write_json
from .transcribe import transcribe_audio


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="clips",
        description="Genera clips listos para redes sociales usando Whisper + Ollama.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input", help="Archivo de video/audio local o URL compatible con yt-dlp")
    parser.add_argument("-o", "--output", default="clips.json", help="Archivo JSON de salida")
    parser.add_argument("--clips", type=int, default=5, help="Número máximo de clips")
    parser.add_argument("--min-duration", type=float, default=15.0, help="Duración mínima de cada clip en segundos")
    parser.add_argument("--max-duration", type=float, default=60.0, help="Duración máxima de cada clip en segundos")
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
    parser.add_argument("--transcript", default=None, help="Reutilizar transcripción JSON en vez de transcribir")
    parser.add_argument("--transcript-out", default=None, help="Guardar la transcripción en este archivo JSON")
    parser.add_argument("--render-dir", default=None, help="Carpeta donde renderizar los clips en MP4 + SRT")
    parser.add_argument("--framing", choices=("auto", "face", "center"), default="auto", help="Encuadre vertical: detecta al hablante o recorta al centro")
    parser.add_argument("--no-subtitles", action="store_true", help="No quemar los subtítulos en el video")
    parser.add_argument("--no-vertical", action="store_true", help="No recortar los clips a formato vertical 9:16")
    parser.add_argument("--name", default=None, help="Nombre base para los archivos renderizados")
    parser.add_argument("--keep-workdir", action="store_true", help="Conservar el audio temporal para depuración")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    workdir = Path(tempfile.mkdtemp(prefix="ai-local-clipping-"))
    try:
        return _run(args, workdir)
    except ClipError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Cancelado.", file=sys.stderr)
        return 130
    finally:
        if args.keep_workdir:
            print(f"Directorio temporal: {workdir}")
        else:
            shutil.rmtree(workdir, ignore_errors=True)


def _run(args: argparse.Namespace, workdir: Path) -> int:
    settings = ClipSettings(args.clips, args.min_duration, args.max_duration)
    settings.validate()
    if args.chunk_chars < 1000:
        raise ClipError("--chunk-chars debe ser al menos 1000")

    url: str | None = None
    if is_url(args.input):
        url = args.input
        print(f"Descargando video: {url}")
        video = download_video(url, workdir)
        source = url
    else:
        video = Path(args.input).expanduser().resolve()
        if not video.is_file():
            raise ClipError(f"No existe el archivo: {video}")
        source = str(video)

    duration = probe_duration(video)
    print(f"Video: {video.name} ({duration:.1f} s)")

    if args.transcript:
        transcript_path = Path(args.transcript).expanduser().resolve()
        if not transcript_path.is_file():
            raise ClipError(f"No existe la transcripción: {transcript_path}")
        print(f"Reutilizando transcripción: {transcript_path}")
        transcript_data = json.loads(transcript_path.read_text(encoding="utf-8"))
        transcript = Transcript.from_dict(transcript_data)
    else:
        audio = extract_audio(video, workdir)
        print("Transcribiendo con Whisper...")
        language = None if args.whisper_language.lower() == "auto" else args.whisper_language
        transcript = transcribe_audio(
            audio,
            workdir,
            model_name=args.whisper_model,
            language=language,
            device=args.whisper_device,
            backend=args.whisper_backend,
        )
    transcript.duration = duration
    if args.transcript_out:
        transcript_out = Path(args.transcript_out).expanduser().resolve()
        write_json(transcript_out, transcript.to_dict())
        print(f"Transcripción guardada: {transcript_out}")

    client = OllamaClient(args.ollama_host, args.ollama_model)
    print(f"Analizando con Ollama ({client.model})...")
    client.check_model()
    summary, candidates = analyze_transcript(client, transcript, settings, chunk_chars=args.chunk_chars)

    warnings: list[str] = []
    if not candidates:
        warnings.append("Ollama no devolvió clips válidos; se usó selección heurística.")
        candidates = heuristic_candidates(transcript, settings)
        if not summary.get("general"):
            summary["general"] = "Resumen no disponible: se seleccionaron los fragmentos automáticamente."
    if not candidates:
        raise ClipError("No se pudieron generar clips con la transcripción disponible")

    render_dir = Path(args.render_dir).expanduser().resolve() if args.render_dir else None
    crops: list[tuple[int, int, int, int] | None] = []
    if render_dir:
        render_dir.mkdir(parents=True, exist_ok=True)
        print(f"Renderizando {len(candidates)} clips en {render_dir}")
        for index, clip in enumerate(candidates, start=1):
            crop = None
            if not args.no_vertical and args.framing in {"auto", "face"}:
                crop = find_speaker_crop(video, clip.start, clip.end)
                if crop is None and args.framing == "face":
                    warnings.append(f"No se detectó rostro en clip_{index:02d}; se usó encuadre central.")
            crops.append(crop)
            srt_path = render_dir / f"clip_{index:02d}.srt"
            write_srt(transcript, clip.start, clip.end, srt_path)
            render_clip(
                video,
                clip.start,
                clip.end,
                render_dir / f"clip_{index:02d}.mp4",
                vertical=not args.no_vertical,
                crop=crop,
                srt=srt_path,
                burn_subtitles=not args.no_subtitles,
            )

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
    )
    output_path = Path(args.output).expanduser().resolve()
    write_json(output_path, output)
    print(f"Listo: {output_path} ({len(candidates)} clips)")
    for index, clip in enumerate(candidates, start=1):
        print(f"  {index:02d}. {clip.start:.1f}-{clip.end:.1f}s  {clip.title}")
    return 0
