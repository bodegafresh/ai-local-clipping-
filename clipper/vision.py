from __future__ import annotations

import base64
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .framing import _cascade_path
from .media import run
from .ollama import OllamaClient, style_prompt
from .schema import ClipCandidate, ClipError, ClipSettings, Transcript, _as_roi

VISION_HINTS = ("qwen3-vl", "qwen2-vl", "llava", "vision", "vl")
WEBCAM_CORNERS = {"bottom-right", "bottom-left", "top-right", "top-left", "none"}


@dataclass
class DiagramLayout:
    has_diagram: bool
    roi: tuple[float, float, float, float] | None
    webcam_corner: str = "none"


def parse_visual_layout(result: dict[str, Any]) -> DiagramLayout:
    has_diagram = bool(result.get("contiene_diagrama_o_texto", False))
    roi = _as_roi(result.get("region_diagrama")) if has_diagram else None
    if has_diagram and roi is None:
        roi = (0.05, 0.30, 0.90, 0.40)
    corner = str(result.get("posicion_webcam", "none")).strip().lower()
    if corner not in WEBCAM_CORNERS:
        corner = "none"
    if not bool(result.get("tiene_webcam_esquina", corner != "none")):
        corner = "none"
    return DiagramLayout(has_diagram, roi, corner)


def _detect_webcam_corner(gray: np.ndarray) -> str:
    cascade_path = _cascade_path()
    if cascade_path is None:
        return "none"
    cascade = cv2.CascadeClassifier(str(cascade_path))
    if cascade.empty():
        return "none"
    height, width = gray.shape[:2]
    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=(max(40, int(width * 0.06)), max(40, int(height * 0.05))),
        maxSize=(int(width * 0.35), int(height * 0.40)),
    )
    for x, y, face_width, face_height in sorted(faces, key=lambda face: face[2] * face[3], reverse=True):
        center_x = (x + face_width / 2) / width
        center_y = (y + face_height / 2) / height
        if center_x > 0.62 and center_y > 0.58:
            return "bottom-right"
        if center_x < 0.38 and center_y > 0.58:
            return "bottom-left"
        if center_x > 0.62 and center_y < 0.42:
            return "top-right"
        if center_x < 0.38 and center_y < 0.42:
            return "top-left"
    return "none"


def detect_layout_cv(
    video: Path,
    start: float,
    end: float,
    workdir: Path,
    index: int,
) -> DiagramLayout:
    workdir.mkdir(parents=True, exist_ok=True)
    moment = start + min(2.0, max(0.0, (end - start) / 2))
    frame = workdir / f"clip_{index:02d}_diagrama.png"
    run(
        [
            "ffmpeg",
            "-nostdin",
            "-y",
            "-loglevel",
            "error",
            "-ss",
            f"{moment:.3f}",
            "-i",
            str(video),
            "-frames:v",
            "1",
            "-vf",
            "scale=960:-2",
            str(frame),
        ],
        description=f"diagram frame {index}",
    )
    image = cv2.imread(str(frame))
    if image is None:
        return DiagramLayout(False, None, "none")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 160)
    height, width = gray.shape[:2]
    contours = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    rectangles: list[tuple[int, int, int, int]] = []
    for contour in contours:
        x, y, rect_width, rect_height = cv2.boundingRect(contour)
        if (
            rect_width > width * 0.08
            and rect_height > height * 0.035
            and rect_width / max(1, rect_height) > 1.2
            and height * 0.12 < y < height * 0.80
        ):
            rectangles.append((x, y, rect_width, rect_height))
    central = edges[int(height * 0.12) : int(height * 0.82), int(width * 0.04) : int(width * 0.96)]
    density = float(np.count_nonzero(central)) / max(1.0, float(central.size))
    has_diagram = len(rectangles) >= 3 or density > 0.035
    roi: tuple[float, float, float, float] | None = None
    if has_diagram:
        if len(rectangles) >= 3:
            x0 = min(item[0] for item in rectangles) / width
            y0 = min(item[1] for item in rectangles) / height
            x1 = max(item[0] + item[2] for item in rectangles) / width
            y1 = max(item[1] + item[3] for item in rectangles) / height
            roi = (
                round(max(0.0, x0), 4),
                round(max(0.0, y0), 4),
                round(min(1.0, x1 - x0), 4),
                round(min(1.0, y1 - y0), 4),
            )
        else:
            roi = (0.05, 0.30, 0.90, 0.40)
    return DiagramLayout(has_diagram, roi, _detect_webcam_corner(gray))


def find_vision_model(client: OllamaClient, preferred: str | None = None) -> str | None:
    names = client.list_models()
    if preferred:
        return preferred if any(preferred in name or name in preferred for name in names) else None
    for hint in VISION_HINTS:
        for name in names:
            if hint in name.lower():
                return name
    return None


def extract_keyframes(
    video: Path,
    start: float,
    end: float,
    workdir: Path,
    index: int,
    *,
    peak: float | None = None,
) -> list[Path]:
    workdir.mkdir(parents=True, exist_ok=True)
    middle = start + (end - start) / 2
    times = [start + 1.0, peak if peak is not None else middle, middle, end - 1.5]
    times = [max(start, min(end - 0.05, float(value))) for value in times]
    frames: list[Path] = []
    for number, moment in enumerate(times, start=1):
        frame = workdir / f"clip_{index:02d}_frame_{number}.png"
        run(
            [
                "ffmpeg",
                "-nostdin",
                "-y",
                "-loglevel",
                "error",
                "-ss",
                f"{moment:.3f}",
                "-i",
                str(video),
                "-frames:v",
                "1",
                "-vf",
                "scale=512:-2",
                str(frame),
            ],
            description=f"keyframe {index}.{number}",
        )
        if frame.is_file():
            frames.append(frame)
    return frames


def _keyframe_targets(
    candidates: list[ClipCandidate],
    peaks: dict[int, float] | None,
) -> list[tuple[int, int, float]]:
    targets: list[tuple[int, int, float]] = []
    for index, clip in enumerate(candidates[:8], start=1):
        middle = clip.start + (clip.end - clip.start) / 2
        peak = (peaks or {}).get(index - 1)
        moments = [clip.start + 1.0, peak if peak is not None else middle, middle, clip.end - 1.5]
        for slot, value in enumerate(moments, start=1):
            moment = max(clip.start, min(clip.end - 0.05, float(value)))
            targets.append((index, slot, moment))
    return targets


def build_keyframe_command(
    video: Path,
    targets: list[tuple[int, int, float]],
    workdir: Path,
) -> list[str]:
    command = ["ffmpeg", "-nostdin", "-y", "-loglevel", "error"]
    for _, _, moment in targets:
        command += ["-ss", f"{moment:.3f}", "-i", str(video)]
    graph = ";".join(f"[{number}:v]scale=384:-2[f{number}]" for number in range(len(targets)))
    command += ["-filter_complex", graph]
    for number, (index, slot, _) in enumerate(targets):
        command += [
            "-map",
            f"[f{number}]",
            "-frames:v",
            "1",
            str(workdir / f"clip_{index:02d}_frame_{slot}.png"),
        ]
    return command


def extract_keyframes_batch(
    video: Path,
    candidates: list[ClipCandidate],
    workdir: Path,
    *,
    peaks: dict[int, float] | None = None,
) -> dict[int, list[Path]]:
    workdir.mkdir(parents=True, exist_ok=True)
    targets = _keyframe_targets(candidates, peaks)
    if not targets:
        return {}
    run(build_keyframe_command(video, targets, workdir), description="keyframes batch")
    frames: dict[int, list[Path]] = {}
    for index, slot, _ in targets:
        path = workdir / f"clip_{index:02d}_frame_{slot}.png"
        if path.is_file():
            frames.setdefault(index, []).append(path)
    return frames


def _encode(frames: list[Path]) -> list[str]:
    return [base64.b64encode(frame.read_bytes()).decode("ascii") for frame in frames if frame.is_file()]


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def refine_candidates(
    client: OllamaClient,
    model: str | None,
    video: Path,
    candidates: list[ClipCandidate],
    transcript: Transcript,
    style: str,
    settings: ClipSettings,
    workdir: Path,
    *,
    peaks: dict[int, float] | None = None,
    warnings: list[str] | None = None,
) -> list[ClipCandidate]:
    if not model:
        if warnings is not None:
            warnings.append("Verificación visual desactivada: no hay modelo multimodal en Ollama.")
        return candidates
    if not candidates:
        return candidates

    refined: list[ClipCandidate] = []
    gaming_note = (
        "Para gaming prioriza fotogramas de acción real, disparos, combate, movimiento y reacción del jugador; "
        "penaliza menús, pantallas de carga, inventarios y pausas como gancho visual.\n"
        if style == "gaming"
        else ""
    )
    frames_by_clip = extract_keyframes_batch(video, candidates, workdir, peaks=peaks)
    for index, clip in enumerate(candidates[:8], start=1):
        frames = frames_by_clip.get(index, [])
        images = _encode(frames)
        if not images:
            refined.append(clip)
            continue
        prompt = (
            f"Perfil editorial: {style}. {style_prompt(style)}\n"
            f"{gaming_note}"
            f"Texto del candidato ({clip.start:.1f}-{clip.end:.1f}s): "
            f"{transcript.text_for_range(clip.start, clip.end)}\n"
            "Evalúa los 4 fotogramas en orden: inicio, pico, desarrollo y cierre. "
            "Describe coherencia emocional o ironía visual, atractivo para redes y ajusta start_ajustado "
            "y end_ajustado si la reacción visual ocurre fuera de los límites actuales. "
            "Indica si contiene un diagrama, diapositiva, código o pantalla con texto en "
            "'contiene_diagrama_o_texto'. Si es así, devuelve 'region_diagrama' con x_pct, y_pct, w_pct y h_pct "
            "normalizados entre 0 y 1 alrededor de las cajas o texto importante. "
            "Indica también 'tiene_webcam_esquina' y 'posicion_webcam' entre bottom-right, bottom-left, "
            "top-right, top-left o none. Devuelve score_visual de 1 a 100 y una razon breve."
        )
        try:
            result = client.chat_with_images(model, prompt, images)
        except ClipError as exc:
            if warnings is not None:
                warnings.append(f"Verificación visual falló en clip_{index:02d}: {exc}")
            refined.append(clip)
            continue

        start = max(clip.start - 2.0, min(clip.end - 1.0, _as_float(result.get("start_ajustado"), clip.start)))
        end = max(start + 1.0, min(clip.end + 2.0, _as_float(result.get("end_ajustado"), clip.end)))
        end = min(end, transcript.total_duration())
        if end - start < settings.min_duration:
            end = min(transcript.total_duration(), start + settings.min_duration)
        if end - start > settings.max_duration:
            start = max(0.0, end - settings.max_duration)
        visual_score = max(1.0, min(100.0, _as_float(result.get("score_visual"), 50.0)))
        previous_score = clip.score_total or clip.score * 100
        score_total = round(previous_score * 0.6 + visual_score * 0.4, 1)
        visual_reason = str(result.get("razon") or result.get("coherencia_emocional") or "").strip()
        evidence = " ".join(filter(None, [clip.evidencia_acustica_visual, visual_reason])).strip()
        layout = parse_visual_layout(result)
        refined.append(
            ClipCandidate(
                start=start,
                end=end,
                title=clip.title,
                description=clip.description,
                tags=clip.tags,
                reason=clip.reason,
                score=score_total / 100,
                rank=index,
                score_total=score_total,
                subestilo=clip.subestilo,
                gancho_3s=clip.gancho_3s,
                razonamiento_editorial=clip.razonamiento_editorial,
                evidencia_acustica_visual=evidence,
                hook_visual_pantalla=clip.hook_visual_pantalla,
                titulo_corto=clip.titulo_corto,
                descripcion_redes=clip.descripcion_redes,
                hashtags=clip.hashtags,
                diagram_roi=layout.roi if layout.has_diagram else None,
                webcam_corner=layout.webcam_corner,
            )
        )
    refined.extend(candidates[len(refined) :])
    refined.sort(key=lambda item: (-item.score_total, item.start))
    for rank, clip in enumerate(refined, start=1):
        clip.rank = rank
    return refined
