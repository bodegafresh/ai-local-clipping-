from __future__ import annotations

import re
import subprocess
import sys
import types
import unittest
import wave
from array import array
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from clipper.audio_enrich import enrich_transcript_with_audio
from clipper.cli import build_parser, resolve_duration_bounds
from clipper.media import (
    _max_pairwise,
    CLEAN_AUDIO_FILTER,
    CLEAN_RENDER_FILTER,
    ENHANCE_FAST_FILTER,
    PROGRESS_HEIGHT,
    PROGRESS_Y,
    STUDIO_AUDIO_FILTER,
    VOCAL_BOOST_FILTER,
    build_clean_audio_command,
    build_demucs_mix_command,
    build_render_command,
    build_sfx_filter,
    build_sfx_schedule,
    build_video_filter,
    build_video_filter_complex,
    detect_noise_floor,
    encoder_options,
    is_url,
    is_vertical,
    preparar_audio_limpio,
    probe_duration,
    resolve_audio_mode,
    resolve_encoder,
    resolve_enhance_mode,
    safe_stem,
    video_hw_filter,
)
from clipper.ollama import OllamaClient, chunk_transcript, selection_prompt
from clipper.schema import (
    ClipCandidate,
    ClipSettings,
    Segment,
    Transcript,
    ass_from_transcript,
    build_output,
    clip_metadata,
    clip_post_text,
    dedupe_candidates,
    detect_gaming_style,
    format_short_timestamp,
    format_timestamp,
    heuristic_candidates,
    normalize_selection,
    normalize_source_url,
    primer_comentario_fijado,
    srt_from_transcript,
    source_url_with_timestamp,
    write_clip_text,
)
from clipper.vision import build_keyframe_command, parse_visual_layout


def sample_transcript() -> Transcript:
    return Transcript.from_dict(
        {
            "idioma": "es",
            "duracion_segundos": 100.0,
            "segmentos": [
                {
                    "inicio": 0.0,
                    "fin": 8.0,
                    "texto": "Hoy vamos a hablar de clips virales.",
                    "palabras": [
                        {"inicio": 0.0, "fin": 0.6, "texto": "Hoy"},
                        {"inicio": 0.7, "fin": 1.2, "texto": "vamos"},
                        {"inicio": 1.3, "fin": 2.0, "texto": "a"},
                        {"inicio": 2.1, "fin": 3.0, "texto": "hablar"},
                        {"inicio": 3.1, "fin": 4.0, "texto": "de"},
                        {"inicio": 4.1, "fin": 5.5, "texto": "clips"},
                        {"inicio": 5.6, "fin": 8.0, "texto": "virales."},
                    ],
                },
                {
                    "inicio": 30.0,
                    "fin": 40.0,
                    "texto": "Este es el mejor consejo del episodio.",
                    "palabras": [
                        {"inicio": 30.0, "fin": 30.5, "texto": "Este"},
                        {"inicio": 30.6, "fin": 31.0, "texto": "es"},
                        {"inicio": 31.1, "fin": 32.0, "texto": "el"},
                        {"inicio": 32.1, "fin": 33.5, "texto": "mejor"},
                        {"inicio": 33.6, "fin": 35.0, "texto": "consejo"},
                        {"inicio": 35.1, "fin": 40.0, "texto": "del episodio."},
                    ],
                },
            ],
        }
    )


class TranscriptTests(unittest.TestCase):
    def test_ranges_and_snap(self) -> None:
        transcript = sample_transcript()
        self.assertEqual(transcript.total_duration(), 100.0)
        self.assertIn("clips virales", transcript.text_for_range(0, 8))
        self.assertEqual(transcript.snap_range(0.2, 5.8), (0.0, 8.0))
        self.assertEqual(len(transcript.words()), 13)

    def test_srt(self) -> None:
        srt = srt_from_transcript(sample_transcript(), 0.0, 8.0)
        self.assertIn("00:00:00,000 --> 00:00:03,000", srt)
        self.assertIn("00:00:03,100 --> 00:00:08,000", srt)
        self.assertIn("Hoy vamos a hablar", srt)
        self.assertIn("de clips virales.", srt)

    def test_srt_relative_times(self) -> None:
        srt = srt_from_transcript(sample_transcript(), 30.0, 40.0)
        self.assertIn("00:00:00,000", srt)
        self.assertNotIn("00:00:30", srt)
        self.assertIn("mejor", srt)
        self.assertIn("consejo", srt)

    def test_ass_viral_styles_and_karaoke(self) -> None:
        ass = ass_from_transcript(
            sample_transcript(),
            30.0,
            40.0,
            hook="ESTE ES EL HOOK PRINCIPAL",
            title="Título de prueba",
        )
        self.assertIn("PlayResX: 1080", ass)
        self.assertIn("PlayResY: 1920", ass)
        hook_style = next(line for line in ass.splitlines() if line.startswith("Style: HookBanner,"))
        hook_fields = [field.strip() for field in hook_style.split(",", 10)]
        self.assertIn("Style: HookBanner,DejaVu Sans,54,", ass)
        self.assertEqual(hook_fields[3], "&H00FFFFFF")
        self.assertEqual(hook_fields[5], "&H2810141D")
        self.assertNotEqual(hook_fields[3], hook_fields[5])
        self.assertIn(",3,18,0,8,70,70,175,1", hook_style)

        default_style = next(line for line in ass.splitlines() if line.startswith("Style: Default,"))
        self.assertIn("&H000A0A0A,", default_style)
        self.assertIn(",1,5,2,2,70,70,390,1", default_style)

        self.assertIn("0:00:03.80,HookBanner", ass)
        self.assertIn(r"{\fad(150,250)\fscx95\fscy95\t(0,120,\fscx100\fscy100)}", ass)
        self.assertNotIn(r"\pos(540,210)", ass)
        self.assertIn(r"{\1c&H00FFFFFF&}ESTE", ass)
        self.assertIn(r"{\1c&H00CBEB91&}HOOK", ass)
        self.assertIn(r"{\1c&H00CBEB91&}PRINCIPAL", ass)
        self.assertIn(r"\fscx92\fscy92\t(0,70,\fscx104\fscy104)", ass)
        self.assertIn(r"\k", ass)
        self.assertIn(r"\1c&H0000E5FF&", ass)
        self.assertNotIn("0:00:30", ass)

        dialogues = [line for line in ass.splitlines() if line.startswith("Dialogue: 0,")]
        for dialogue in dialogues:
            text = dialogue.rsplit(",,", 1)[1]
            plain = re.sub(r"\{[^}]*\}", "", text).strip()
            self.assertLessEqual(len(plain.split()), 3)

    def test_ass_falls_back_to_first_five_title_words(self) -> None:
        ass = ass_from_transcript(
            sample_transcript(),
            0.0,
            8.0,
            title="Uno dos tres cuatro cinco seis siete",
        )
        hook_line = next(line for line in ass.splitlines() if ",HookBanner," in line)
        plain = re.sub(r"\{[^}]*\}", "", hook_line.rsplit(",,", 1)[1])
        self.assertEqual(plain, "UNO DOS TRES CUATRO\\NCINCO")
        self.assertNotIn("SEIS", ass)

    def test_ass_viral_fx_off_keeps_legacy_style(self) -> None:
        ass = ass_from_transcript(sample_transcript(), 30.0, 40.0, viral_fx=False)
        self.assertIn("Style: Default,DejaVu Sans,64,", ass)
        self.assertIn(",2,80,80,210,1", ass)
        self.assertNotIn("HookBanner", ass)
        self.assertNotIn(r"\k", ass)

    def test_timestamp(self) -> None:
        self.assertEqual(format_timestamp(78.2), "00:01:18.200")
        self.assertEqual(format_timestamp(1.5, comma=True), "00:00:01,500")


class ClipTests(unittest.TestCase):
    def test_normalize_selection_clamps_and_dedupes(self) -> None:
        raw = {
            "resumen": {"general": "Resumen", "temas": ["a", "b"], "tono": "divulgativo", "audiencia": "general"},
            "clips": [
                {"inicio": 0.0, "fin": 20.0, "titulo": "Etiquetado", "descripcion": "d", "tags": ["#Tag", "Tag"], "score": 0.9},
                {"inicio": 1.0, "fin": 10.0, "titulo": "Duplicado", "descripcion": "d", "tags": [], "score": 0.8},
                {"inicio": 30.0, "fin": 90.0, "titulo": "Largo", "descripcion": "d", "tags": [], "score": 0.5},
            ],
        }
        summary, clips = normalize_selection(raw, sample_transcript(), ClipSettings(3, 10, 30))
        self.assertEqual(summary["general"], "Resumen")
        self.assertEqual(len(clips), 2)
        self.assertLessEqual(clips[0].end - clips[0].start, 30.0)
        self.assertEqual(clips[0].tags[0], "tag")
        self.assertEqual(len(clips[0].tags), 1)
        self.assertEqual([clip.start for clip in clips], sorted(clip.start for clip in clips))

    def test_normalize_enriched_schema(self) -> None:
        raw = {
            "resumen": {"general": "R", "estilo_aplicado": "humor"},
            "clips": [
                {
                    "start": 0.0,
                    "end": 10.0,
                    "titulo": "T",
                    "descripcion": "D",
                    "tags": ["x"],
                    "score": 0.8,
                    "rank": 1,
                    "score_total": 94,
                    "subestilo": "ironia_y_remate",
                    "gancho_3s": "frase exacta",
                    "razonamiento_editorial": "remate inesperado",
                    "evidencia_acustica_visual": "pausa y risas",
                    "hook_visual_pantalla": "TRABAJAR DOMINGOS",
                    "titulo_corto": "Título corto",
                    "descripcion_redes": "Copy listo 👇",
                    "hashtags": ["#Humor", "#Ironia"],
                }
            ],
        }
        _, clips = normalize_selection(raw, sample_transcript(), ClipSettings(1, 5, 20))
        clip = clips[0]
        self.assertEqual(clip.start, 0.0)
        self.assertEqual(clip.score_total, 94)
        self.assertEqual(clip.subestilo, "ironia_y_remate")
        self.assertEqual(clip.hook_visual_pantalla, "TRABAJAR DOMINGOS")
        self.assertEqual(clip.hashtags, ["humor", "ironia"])

    def test_heuristic_fallback(self) -> None:
        clips = heuristic_candidates(sample_transcript(), ClipSettings(2, 5, 15))
        self.assertGreater(len(clips), 0)
        self.assertTrue(all(clip.end > clip.start for clip in clips))

    def test_build_output(self) -> None:
        _, clips = normalize_selection(
            {
                "resumen": {"general": "R"},
                "clips": [{"inicio": 0, "fin": 8, "titulo": "T", "descripcion": "D", "tags": ["x"], "score": 0.7}],
            },
            sample_transcript(),
            ClipSettings(1, 5, 20),
        )
        with TemporaryDirectory() as tmp:
            output = build_output(
                source="raw.mp4",
                source_path=Path(tmp) / "raw.mp4",
                url=None,
                duration=100.0,
                transcript=sample_transcript(),
                summary={"general": "R"},
                clips=clips,
                whisper_model="small",
                ollama_model="qwen2.5:7b",
                render_dir=None,
                video_name="raw",
                warnings=[],
            )
        self.assertEqual(output["total_clips"], 1)
        self.assertIn("transcripcion", output["clips"][0])
        self.assertIn("tiktok", output["clips"][0]["publicacion"])
        self.assertEqual(output["clips"][0]["hashtags"], "#x")

    def test_source_url_timestamp_normalization(self) -> None:
        self.assertEqual(format_short_timestamp(252.7), "04:12")
        self.assertIsNone(normalize_source_url("no-es-una-url"))
        self.assertEqual(
            source_url_with_timestamp("https://www.youtube.com/watch?v=abc123&t=5s", 252.7),
            "https://www.youtube.com/watch?v=abc123&t=252s",
        )
        self.assertEqual(
            source_url_with_timestamp("https://youtu.be/abc123?si=x&t=5s", 252.7),
            "https://youtu.be/abc123?si=x&t=252s",
        )

    def test_clip_post_text_separa_primer_comentario(self) -> None:
        clip = ClipCandidate(
            start=252.0,
            end=260.0,
            title="T",
            description="D",
            tags=["x"],
            reason="R",
            score=0.7,
        )
        url = "https://www.youtube.com/watch?v=abc123"
        text = clip_post_text(clip, url)
        self.assertIn("=== TÍTULO ===\nT", text)
        self.assertIn("=== DESCRIPCIÓN (TIKTOK / REELS / SHORTS) ===\nD", text)
        self.assertIn("🎥 Video completo (minuto 04:12): https://www.youtube.com/watch?v=abc123&t=252s", text)
        self.assertIn(
            "📌 Mira el contexto completo desde el minuto 04:12 aquí 👉 "
            "https://www.youtube.com/watch?v=abc123&t=252s",
            text,
        )
        metadata = clip_metadata(clip, 1, url)
        self.assertEqual(metadata["source_timestamp"], "04:12")
        self.assertEqual(metadata["source_url_timestamp"], "https://www.youtube.com/watch?v=abc123&t=252s")
        self.assertEqual(metadata["primer_comentario_fijado"], primer_comentario_fijado(clip, url))

    def test_write_clip_text_crea_archivo(self) -> None:
        _, clips = normalize_selection(
            {
                "resumen": {"general": "R"},
                "clips": [{"inicio": 0, "fin": 8, "titulo": "T", "descripcion": "D", "tags": ["x"], "score": 0.7}],
            },
            sample_transcript(),
            ClipSettings(1, 5, 20),
        )
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "clip_01.txt"
            write_clip_text(path, clips[0], "https://youtu.be/abc123")
            self.assertIn("https://youtu.be/abc123?t=0s", path.read_text(encoding="utf-8"))


class OllamaTests(unittest.TestCase):
    def test_parse_json_with_fences(self) -> None:
        from clipper.ollama import _parse_json

        self.assertEqual(_parse_json('```json\n{"a": 1}\n```'), {"a": 1})

    def test_chunking_and_prompt(self) -> None:
        transcript = sample_transcript()
        self.assertEqual(len(chunk_transcript(transcript, 12000)), 1)
        self.assertEqual(len(chunk_transcript(transcript, 100)), 2)
        prompt = selection_prompt(transcript, ClipSettings(2, 15, 60))
        self.assertIn("[0.0-8.0]", prompt)
        self.assertIn("entre 15 y 60 segundos", prompt)

    def test_style_prompts_are_differentiated(self) -> None:
        expected = {
            "humor": "remates inesperados",
            "noticias": "titular o hecho central",
            "vender": "Problema/Dolor",
            "educar": "unidad autocontenida",
            "debate": "segundo 0.0",
        }
        prompts = {
            style: selection_prompt(sample_transcript(), ClipSettings(1, 5, 20), style=style)
            for style in expected
        }
        for style, phrase in expected.items():
            self.assertIn(phrase, prompts[style])
        self.assertEqual(len(set(prompts.values())), len(expected))

    def test_vision_fallback_without_model(self) -> None:
        from clipper.vision import refine_candidates

        candidates = heuristic_candidates(sample_transcript(), ClipSettings(1, 5, 20))
        warnings: list[str] = []
        with TemporaryDirectory() as tmp:
            result = refine_candidates(
                OllamaClient("127.0.0.1:11434", "m"),
                None,
                Path(tmp) / "video.mp4",
                candidates,
                sample_transcript(),
                "auto",
                ClipSettings(1, 5, 20),
                Path(tmp),
                warnings=warnings,
            )
        self.assertEqual(result, candidates)
        self.assertTrue(any("Verificación visual desactivada" in item for item in warnings))

    def test_host_normalization(self) -> None:
        self.assertEqual(OllamaClient("127.0.0.1:11434", "m").host, "http://127.0.0.1:11434")


class AudioCleaningTests(unittest.TestCase):
    def test_command_builder(self) -> None:
        command = build_clean_audio_command(Path("entrada.mp4"), Path("salida.wav"))
        self.assertEqual(
            command,
            [
                "ffmpeg",
                "-nostdin",
                "-y",
                "-loglevel",
                "error",
                "-i",
                "entrada.mp4",
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-af",
                CLEAN_AUDIO_FILTER,
                "-c:a",
                "pcm_s16le",
                "salida.wav",
            ],
        )
        self.assertIn("highpass=f=90", CLEAN_AUDIO_FILTER)
        self.assertIn("lowpass=f=7800", CLEAN_AUDIO_FILTER)
        self.assertIn("afftdn=nf=-22", CLEAN_AUDIO_FILTER)
        self.assertIn("equalizer=f=2800:width_type=h:width=1200:g=4.5", CLEAN_AUDIO_FILTER)
        self.assertIn("acompressor=threshold=-22dB:ratio=4:attack=5:release=60:makeup=4", CLEAN_AUDIO_FILTER)
        self.assertIn("dynaudnorm=f=150:g=15:p=0.90:m=12.0", CLEAN_AUDIO_FILTER)
        self.assertEqual(CLEAN_RENDER_FILTER, VOCAL_BOOST_FILTER)
        self.assertIn("volume=6dB", VOCAL_BOOST_FILTER)
        self.assertIn("equalizer=f=2600:width_type=h:width=1400:g=5.0", VOCAL_BOOST_FILTER)
        self.assertIn("equalizer=f=4500:width_type=h:width=1500:g=3.0", VOCAL_BOOST_FILTER)
        self.assertIn("acompressor=threshold=-24dB:ratio=5:attack=4:release=50:makeup=5", VOCAL_BOOST_FILTER)
        self.assertIn("dynaudnorm=f=100:g=11:p=0.92:m=15.0", VOCAL_BOOST_FILTER)
        self.assertIn("alimiter=limit=0.95:attack=2:release=25", VOCAL_BOOST_FILTER)

    def test_render_command_boosts_vocals_by_default(self) -> None:
        command = build_render_command(
            Path("video.mp4"),
            1.0,
            3.0,
            Path("clip.mp4"),
            vertical=False,
        )
        self.assertEqual(command[command.index("-af") + 1], CLEAN_RENDER_FILTER)
        self.assertEqual(command[command.index("-ac") + 1], "2")
        self.assertEqual(command[command.index("-ar") + 1], "48000")
        self.assertNotIn("copy", command)
        self.assertEqual(command[command.index("-c:a") + 1], "aac")
        self.assertEqual(command[command.index("-b:a") + 1], "192k")

    def test_demucs_mix_keeps_voice_full_and_ambient_at_12_percent(self) -> None:
        command = build_demucs_mix_command(
            Path("vocals.wav"),
            Path("no_vocals.wav"),
            Path("mezcla.wav"),
        )
        filter_complex = command[command.index("-filter_complex") + 1]
        self.assertIn("weights=1 0.12:normalize=0", filter_complex)
        self.assertIn("alimiter=limit=0.95", filter_complex)
        self.assertEqual(command[command.index("-c:a") + 1], "pcm_s16le")

    def test_clean_audio_preserves_duration(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = root / "video.mp4"
            subprocess.run(
                [
                    "ffmpeg",
                    "-nostdin",
                    "-y",
                    "-loglevel",
                    "error",
                    "-f",
                    "lavfi",
                    "-i",
                    "testsrc=duration=2:size=320x240:rate=25",
                    "-f",
                    "lavfi",
                    "-i",
                    "sine=frequency=440:duration=2:sample_rate=44100",
                    "-shortest",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "ultrafast",
                    "-c:a",
                    "aac",
                    str(video),
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            clean = preparar_audio_limpio(video, root / "audio_limpio.wav")
            self.assertTrue(clean.is_file())
            self.assertAlmostEqual(probe_duration(clean), probe_duration(video), delta=0.02)


class ViralFxTests(unittest.TestCase):
    def test_video_filter_adds_microzoom_and_progress(self) -> None:
        on = build_video_filter(
            vertical=True,
            crop=None,
            framing="auto",
            enhance="off",
            subtitles=None,
            burn_subtitles=False,
            viral_fx=True,
            duration=20.0,
            emphasis=[5.0],
            pauses=[12.0],
        )
        off = build_video_filter(
            vertical=True,
            crop=None,
            framing="auto",
            enhance="off",
            subtitles=None,
            burn_subtitles=False,
            viral_fx=False,
            duration=20.0,
            emphasis=[5.0],
            pauses=[12.0],
        )
        self.assertIn("scale=w='trunc(iw/(1+min(0.12,", on)
        self.assertIn(":eval=frame:flags=bilinear,scale=1080:1920:flags=lanczos,setsar=1", on)
        self.assertNotIn("zoompan", on)
        self.assertEqual(PROGRESS_Y, 0)
        self.assertIn(f"y={PROGRESS_Y}:w=1080:h={PROGRESS_HEIGHT}", on)
        self.assertIn("color=0x91EBCB@0.9", on)
        self.assertNotIn("y=1420", on)
        self.assertNotIn("drawbox", off)
        self.assertNotIn("min(0.12,", off)

    def test_microzoom_combines_four_windows_with_two_arg_max(self) -> None:
        self.assertEqual(_max_pairwise(["a", "b", "c", "d"]), "max(max(a,b),max(c,d))")
        on = build_video_filter(
            vertical=True,
            crop=None,
            framing="auto",
            enhance="off",
            subtitles=None,
            burn_subtitles=False,
            viral_fx=True,
            duration=40.0,
            emphasis=[3.0, 6.0, 16.0, 33.0],
            pauses=[],
        )
        self.assertIn("max(max(", on)

    def test_render_command_reuses_full_audio_with_seek(self) -> None:
        command = build_render_command(
            Path("video.mp4"),
            12.5,
            32.5,
            Path("out.mp4"),
            False,
            audio_source=Path("audio.wav"),
            audio_seek=12.5,
            duration=20.0,
        )
        windows = [command[index : index + 6] for index in range(len(command))]
        self.assertIn(["-ss", "12.500", "-t", "20.000", "-i", "audio.wav"], windows)

    def test_sfx_schedule_is_empty_when_assets_are_missing(self) -> None:
        with TemporaryDirectory() as tmp:
            directory = Path(tmp) / "sfx"
            directory.mkdir()
            self.assertEqual(build_sfx_schedule(directory, 20.0, "humor", [5.0], [12.0]), [])

    def test_sfx_schedule_and_filter(self) -> None:
        with TemporaryDirectory() as tmp:
            directory = Path(tmp) / "sfx"
            directory.mkdir()
            for name in ("whoosh.wav", "pop.wav", "ding.wav", "bass_drop.wav"):
                (directory / name).write_bytes(b"RIFF")
            schedule = build_sfx_schedule(directory, 20.0, "humor", [5.0, 9.0], [12.0])
            self.assertEqual([round(at, 2) for _, at in schedule], [0.05, 5.0, 9.0, 11.6])

            audio_filter = build_sfx_filter(schedule, voice_input=1, vocal_boost=True)
            self.assertIn("[1:a]" + VOCAL_BOOST_FILTER + "[voice]", audio_filter)
            self.assertIn("adelay=50|50,volume=-14dB[sfx0]", audio_filter)
            self.assertIn("amix=inputs=5:duration=first:normalize=0[outa]", audio_filter)

            command = build_render_command(
                Path("video.mp4"),
                0.0,
                20.0,
                Path("clip.mp4"),
                vertical=True,
                audio_source=Path("voz.wav"),
                sfx=schedule,
            )
            self.assertIn("-filter_complex", command)
            self.assertNotIn("-af", command)
            map_indexes = [index for index, value in enumerate(command) if value == "-map"]
            self.assertEqual(command[map_indexes[0] + 1], "0:v:0")
            self.assertEqual(command[map_indexes[-1] + 1], "[outa]")

            command_off = build_render_command(
                Path("video.mp4"),
                0.0,
                20.0,
                Path("clip_off.mp4"),
                vertical=True,
                viral_fx=False,
                sfx=[],
            )
            self.assertNotIn("-filter_complex", command_off)


class DiagramZoomTests(unittest.TestCase):
    def test_parse_visual_layout(self) -> None:
        layout = parse_visual_layout(
            {
                "contiene_diagrama_o_texto": True,
                "region_diagrama": {"x_pct": 5, "y_pct": 35, "w_pct": 90, "h_pct": 35},
                "tiene_webcam_esquina": True,
                "posicion_webcam": "bottom-right",
            }
        )
        self.assertTrue(layout.has_diagram)
        self.assertEqual(layout.roi, (0.05, 0.35, 0.9, 0.35))
        self.assertEqual(layout.webcam_corner, "bottom-right")
        self.assertFalse(parse_visual_layout({"contiene_diagrama_o_texto": False}).has_diagram)

    def test_diagram_zoom_filter_uses_even_dynamic_crops(self) -> None:
        graph = build_video_filter_complex(
            vertical=True,
            enhance="off",
            subtitles=None,
            burn_subtitles=False,
            viral_fx=False,
            duration=20.0,
            emphasis=[],
            pauses=[],
            fps=30.0,
            diagram_roi=(0.05, 0.35, 0.9, 0.35),
            split_webcam=False,
        )
        self.assertIsNotNone(graph)
        assert graph is not None
        self.assertIn("crop=w='trunc(iw/(", graph)
        self.assertIn("h='trunc(ih/(", graph)
        self.assertIn("scale=1080:1920:force_original_aspect_ratio=decrease", graph)
        self.assertIn("pad=1080:1920:(ow-iw)/2:(oh-ih)/2", graph)
        self.assertIn("if(lt(t,2.500),0,if(lt(t,16.500),1,0))", graph)
        self.assertIn("min(1,max(0,(t-2.500)/14.000))", graph)
        self.assertIn(",1.75,1)", graph)
        self.assertIn("*(min(1,max(0,(t-2.500)/14.000)))/2)*2,0)", graph)
        self.assertIn("0.52500*ih", graph)
        self.assertIn("[fg]scale=1080:1920:force_original_aspect_ratio=decrease", graph)
        self.assertGreaterEqual(graph.count("/2)*2"), 6)
        self.assertIn("[vout]", graph)

    def test_split_webcam_filter_composes_top_camera_and_center_diagram(self) -> None:
        graph = build_video_filter_complex(
            vertical=True,
            enhance="off",
            subtitles=None,
            burn_subtitles=False,
            viral_fx=False,
            duration=20.0,
            emphasis=[],
            pauses=[],
            fps=30.0,
            diagram_roi=(0.05, 0.35, 0.9, 0.35),
            split_webcam=True,
            webcam_corner="bottom-right",
        )
        self.assertIsNotNone(graph)
        assert graph is not None
        self.assertIn("scale=440:260:flags=lanczos,cas=0.35", graph)
        self.assertIn("scale=1080:960:force_original_aspect_ratio=decrease", graph)
        self.assertIn("pad=1080:960:(ow-iw)/2:(oh-ih)/2", graph)
        self.assertNotIn("force_original_aspect_ratio=increase", graph)
        self.assertIn("overlay=0:440:format=auto", graph)
        self.assertIn("overlay=(W-w)/2:150:format=auto", graph)
        self.assertIn("drawbox=x=320:y=150:w=440:h=260:color=0x91EBCB@0.95:t=2", graph)
        self.assertIn("x='iw-(trunc(iw*0.28/2)*2)'", graph)
        self.assertIn("y='ih-(trunc(ih*0.24/2)*2)'", graph)

    def test_render_command_uses_single_complex_graph(self) -> None:
        command = build_render_command(
            Path("video.mp4"),
            0.0,
            20.0,
            Path("clip.mp4"),
            vertical=True,
            diagram_roi=(0.05, 0.35, 0.9, 0.35),
            split_webcam=True,
            webcam_corner="bottom-right",
        )
        self.assertIn("-filter_complex", command)
        self.assertNotIn("-vf", command)
        self.assertNotIn("-af", command)
        map_indexes = [index for index, value in enumerate(command) if value == "-map"]
        self.assertEqual(command[map_indexes[0] + 1], "[vout]")
        self.assertEqual(command[map_indexes[-1] + 1], "[outa]")

    def test_cli_flags_default_to_auto(self) -> None:
        args = build_parser().parse_args(["video.mp4"])
        self.assertEqual(args.diagram_zoom, "auto")
        self.assertEqual(args.split_webcam, "auto")


class VideoEnhancementTests(unittest.TestCase):
    def _make_video(self, path: Path, width: int, height: int) -> None:
        subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-y",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                f"testsrc=duration=1:size={width}x{height}:rate=25",
                "-c:v",
                "libx264",
                "-preset",
                "ultrafast",
                str(path),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def test_auto_enables_fast_on_low_resolution(self) -> None:
        with TemporaryDirectory() as tmp:
            video = Path(tmp) / "sd.mp4"
            self._make_video(video, 320, 240)
            self.assertEqual(resolve_enhance_mode("auto", video), "fast")
            command = build_render_command(video, 0.0, 1.0, Path(tmp) / "clip.mp4", True, enhance="fast")
            self.assertIn(ENHANCE_FAST_FILTER, command[command.index("-vf") + 1])
            self.assertIn("hqdn3d=4.0:3.0:6.0:4.5", ENHANCE_FAST_FILTER)
            self.assertIn("cas=0.45", ENHANCE_FAST_FILTER)

    def test_auto_disables_enhancement_on_720p(self) -> None:
        with TemporaryDirectory() as tmp:
            video = Path(tmp) / "hd.mp4"
            self._make_video(video, 1280, 720)
            self.assertEqual(resolve_enhance_mode("auto", video), "off")
            command = build_render_command(video, 0.0, 1.0, Path(tmp) / "clip.mp4", True, enhance="off")
            self.assertNotIn("hqdn3d", " ".join(command))


class AudioEnrichmentTests(unittest.TestCase):
    def test_inserts_pause_reaction_and_emphasis(self) -> None:
        transcript = Transcript(
            segments=[
                Segment(start=0.0, end=1.0, text="hola"),
                Segment(start=4.0, end=5.0, text="mundo"),
            ]
        )
        with TemporaryDirectory() as tmp:
            audio = Path(tmp) / "audio.wav"
            sample_rate = 16000
            samples = array("h", [0]) * (sample_rate * 5)
            samples[0:sample_rate] = array("h", [1000]) * sample_rate
            samples[sample_rate : sample_rate + sample_rate // 2] = array("h", [20000]) * (
                sample_rate // 2
            )
            samples[sample_rate + sample_rate // 2 : 4 * sample_rate] = array("h", [100]) * (
                3 * sample_rate - sample_rate // 2
            )
            samples[4 * sample_rate :] = array("h", [5000]) * sample_rate
            with wave.open(str(audio), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(sample_rate)
                handle.writeframes(samples.tobytes())

            result = enrich_transcript_with_audio(transcript, audio)

        self.assertIn("[PAUSA 3.0s]", result.transcript.segments[0].text)
        self.assertIn("[REACCIÓN/RISAS]", result.transcript.segments[0].text)
        self.assertIn("[ÉNFASIS ALTO]", result.transcript.segments[1].text)
        self.assertIn(0, result.peaks)


class FasterWhisperTests(unittest.TestCase):
    def test_vad_filter_is_enabled(self) -> None:
        from clipper.transcribe import _transcribe_faster_whisper

        captured: dict[str, object] = {}

        class FakeSegment:
            start = 0.0
            end = 1.0
            text = "hola"
            words = []

        class FakeInfo:
            language = "es"

        class FakeModel:
            def transcribe(self, audio: str, **kwargs: object) -> tuple[list[FakeSegment], FakeInfo]:
                captured.update(kwargs)
                return [FakeSegment()], FakeInfo()

        fake_module = types.ModuleType("faster_whisper")
        fake_module.WhisperModel = lambda *args, **kwargs: FakeModel()
        with patch.dict(sys.modules, {"faster_whisper": fake_module}):
            transcript = _transcribe_faster_whisper(Path("audio.wav"), "small", "es", "cpu")

        self.assertEqual(transcript.segments[0].text, "hola")
        self.assertIs(captured["vad_filter"], True)
        self.assertEqual(captured["vad_parameters"], dict(min_silence_duration_ms=600))
        self.assertEqual(captured["beam_size"], 1)


class AdaptiveAudioTests(unittest.TestCase):
    def test_detect_noise_floor_parses_volumedetect(self) -> None:
        completed = subprocess.CompletedProcess(
            [], 0, "", "mean_volume: -50.1 dB\nmax_volume: -3.0 dB\n"
        )
        with patch("clipper.media.run", return_value=completed):
            self.assertEqual(detect_noise_floor(Path("audio.wav")), -50.1)

    def test_auto_mode_threshold(self) -> None:
        self.assertEqual(resolve_audio_mode("auto", -50.0), "studio")
        self.assertEqual(resolve_audio_mode("auto", -40.0), "noisy")
        self.assertEqual(resolve_audio_mode("studio", -10.0), "studio")
        self.assertEqual(resolve_audio_mode("noisy", -90.0), "noisy")

    def test_render_command_selects_studio_chain(self) -> None:
        command = build_render_command(
            Path("video.mp4"),
            0.0,
            10.0,
            Path("clip.mp4"),
            vertical=True,
            audio_mode="studio",
            encoder="libx264",
        )
        self.assertIn(STUDIO_AUDIO_FILTER, command)
        self.assertNotIn(VOCAL_BOOST_FILTER, command)

    def test_clean_command_selects_studio_chain(self) -> None:
        command = build_clean_audio_command(Path("video.mp4"), Path("audio.wav"), audio_mode="studio")
        self.assertIn(STUDIO_AUDIO_FILTER, command)
        self.assertNotIn(CLEAN_AUDIO_FILTER, command)

    def test_studio_preparation_skips_demucs(self) -> None:
        with TemporaryDirectory() as directory:
            output = Path(directory) / "audio.wav"
            completed = subprocess.CompletedProcess([], 0, "", "")
            with (
                patch("clipper.media.probe_duration", return_value=1.0),
                patch("clipper.media.run", return_value=completed) as run_mock,
                patch("clipper.media._isolate_with_demucs") as demucs_mock,
            ):
                preparar_audio_limpio(
                    Path("video.mp4"),
                    output,
                    isolate_vocals=True,
                    audio_mode="studio",
                )
            demucs_mock.assert_not_called()
            self.assertIn(STUDIO_AUDIO_FILTER, run_mock.call_args.args[0])

    def test_audio_preparation_reuses_duration(self) -> None:
        with TemporaryDirectory() as directory:
            output = Path(directory) / "audio.wav"
            completed = subprocess.CompletedProcess([], 0, "", "")
            with (
                patch("clipper.media.probe_duration") as probe_mock,
                patch("clipper.media.run", return_value=completed) as run_mock,
            ):
                preparar_audio_limpio(Path("video.mp4"), output, duration=12.0)
            probe_mock.assert_not_called()
            self.assertIn("-t", run_mock.call_args.args[0])
            self.assertEqual(run_mock.call_args.args[0][-2:], ["-t", "12.000", str(output)][-2:])


class EncoderTests(unittest.TestCase):
    def test_vaapi_is_selected_when_device_and_encoder_exist(self) -> None:
        with (
            patch("clipper.media.VAAPI_DEVICE", Path("/dev/dri/renderD128")),
            patch.object(Path, "exists", return_value=True),
            patch("clipper.media.ffmpeg_supports_encoder", return_value=True),
        ):
            encoder, device = resolve_encoder()
        self.assertEqual(encoder, "h264_vaapi")
        self.assertEqual(device, Path("/dev/dri/renderD128"))

    def test_falls_back_to_libx264(self) -> None:
        with (
            patch("clipper.media.VAAPI_DEVICE", Path("/dev/dri/renderD128")),
            patch.object(Path, "exists", return_value=False),
        ):
            encoder, device = resolve_encoder()
        self.assertEqual(encoder, "libx264")
        self.assertIsNone(device)

    def test_encoder_options(self) -> None:
        self.assertIn("-qp", encoder_options("h264_vaapi"))
        self.assertIn("veryfast", encoder_options("libx264"))
        self.assertIn("format=nv12,hwupload", video_hw_filter("h264_vaapi"))
        self.assertEqual(video_hw_filter("libx264"), "")

    def test_render_command_uses_vaapi_graph(self) -> None:
        command = build_render_command(
            Path("video.mp4"),
            0.0,
            10.0,
            Path("clip.mp4"),
            vertical=True,
            encoder="h264_vaapi",
        )
        joined = " ".join(command)
        self.assertIn("-vaapi_device", command)
        self.assertIn("h264_vaapi", command)
        self.assertIn("format=nv12,hwupload", joined)
        self.assertNotIn("libx264", joined)

    def test_ai_restore_is_skipped_for_hd_sources(self) -> None:
        with patch("clipper.media.probe_video_height", return_value=1080):
            self.assertEqual(resolve_enhance_mode("ai", Path("video.mp4")), "off")
        with patch("clipper.media.probe_video_height", return_value=480):
            self.assertEqual(resolve_enhance_mode("ai", Path("video.mp4")), "ai")


class KeyframeBatchTests(unittest.TestCase):
    def test_single_ffmpeg_call_generates_all_low_res_frames(self) -> None:
        targets = [(1, 1, 1.0), (1, 2, 5.0), (2, 1, 11.0)]
        command = build_keyframe_command(Path("video.mp4"), targets, Path("vision"))
        self.assertEqual(command.count("ffmpeg"), 1)
        self.assertEqual(command.count("-i"), 3)
        self.assertEqual(" ".join(command).count("scale=384:-2"), 3)
        self.assertIn(str(Path("vision") / "clip_01_frame_1.png"), command)
        self.assertIn(str(Path("vision") / "clip_02_frame_1.png"), command)


class NativeVerticalAndGamingTests(unittest.TestCase):
    def test_is_vertical(self) -> None:
        with patch("clipper.media._probe_video_field", side_effect=["1080", "1920"]):
            self.assertTrue(is_vertical(Path("video.mp4")))
        with patch("clipper.media._probe_video_field", side_effect=["1920", "1080"]):
            self.assertFalse(is_vertical(Path("video.mp4")))

    def test_native_vertical_bypasses_complex_graph(self) -> None:
        video_filter = build_video_filter(
            vertical=True,
            crop=(0, 0, 100, 100),
            framing="fit",
            enhance="fast",
            subtitles=None,
            burn_subtitles=False,
            viral_fx=True,
            duration=10.0,
            emphasis=[],
            pauses=[],
            fps=30.0,
            native_vertical=True,
        )
        self.assertIn("scale=1080:1920:flags=lanczos", video_filter)
        self.assertNotIn("boxblur", video_filter)
        self.assertNotIn("crop=", video_filter)
        complex_filter = build_video_filter_complex(
            vertical=True,
            enhance="off",
            subtitles=None,
            burn_subtitles=False,
            viral_fx=True,
            duration=10.0,
            emphasis=[],
            pauses=[],
            fps=30.0,
            diagram_roi=(0.1, 0.1, 0.8, 0.8),
            split_webcam=True,
            native_vertical=True,
        )
        self.assertIsNone(complex_filter)
        command = build_render_command(
            Path("video.mp4"),
            0.0,
            10.0,
            Path("clip.mp4"),
            vertical=True,
            encoder="libx264",
            diagram_roi=(0.1, 0.1, 0.8, 0.8),
            split_webcam=True,
            native_vertical=True,
        )
        joined = " ".join(command)
        self.assertNotIn("-filter_complex", joined)
        self.assertIn("scale=1080:1920:flags=lanczos", joined)

    def test_gaming_style_prompt_and_duration_bounds(self) -> None:
        prompt = selection_prompt(sample_transcript(), ClipSettings(2, 20, 42), style="gaming")
        self.assertIn("Tensión -> Clímax -> Reacción", prompt)
        self.assertIn("entre 20 y 42 segundos", prompt)
        self.assertEqual(resolve_duration_bounds(None, None, "gaming"), (20.0, 42.0))
        self.assertEqual(resolve_duration_bounds(9.0, 33.0, "gaming"), (9.0, 33.0))
        self.assertEqual(resolve_duration_bounds(None, None, "auto"), (15.0, 60.0))
        args = build_parser().parse_args(["video.mp4", "--style", "gaming"])
        self.assertEqual(args.style, "gaming")

    def test_detect_gaming_style(self) -> None:
        transcript = Transcript.from_dict(
            {
                "idioma": "es",
                "segmentos": [
                    {
                        "inicio": 0.0,
                        "fin": 4.0,
                        "texto": "[ÉNFASIS ALTO] ¡No! ¡Morí! [REACCIÓN/RISAS] Gané la partida.",
                    }
                ],
            }
        )
        self.assertTrue(detect_gaming_style(transcript, True))
        self.assertFalse(detect_gaming_style(transcript, False))

    def test_isolated_fillers_are_not_rendered(self) -> None:
        transcript = Transcript.from_dict(
            {
                "idioma": "es",
                "segmentos": [
                    {"inicio": 0.0, "fin": 0.5, "texto": "ah", "palabras": [{"inicio": 0.0, "fin": 0.5, "texto": "ah"}]},
                    {"inicio": 0.6, "fin": 1.0, "texto": "oh", "palabras": [{"inicio": 0.6, "fin": 1.0, "texto": "oh"}]},
                    {
                        "inicio": 1.2,
                        "fin": 3.0,
                        "texto": "Gané la partida",
                        "palabras": [
                            {"inicio": 1.2, "fin": 1.7, "texto": "Gané"},
                            {"inicio": 1.8, "fin": 2.2, "texto": "la"},
                            {"inicio": 2.3, "fin": 3.0, "texto": "partida"},
                        ],
                    },
                ],
            }
        )
        srt = srt_from_transcript(transcript, 0.0, 4.0)
        self.assertNotIn("ah", srt)
        self.assertNotIn("oh", srt)
        self.assertIn("Gané", srt)
        for viral_fx in (False, True):
            ass = ass_from_transcript(transcript, 0.0, 4.0, viral_fx=viral_fx)
            dialogue_texts = [line.split(",", 9)[-1].strip() for line in ass.splitlines() if line.startswith("Dialogue:")]
            self.assertFalse(any(text in {"ah", "oh"} for text in dialogue_texts))
            self.assertIn("Gané", ass)


class MiscTests(unittest.TestCase):
    def test_is_url(self) -> None:
        self.assertTrue(is_url("https://www.youtube.com/watch?v=yAMoc5c05V4"))
        self.assertFalse(is_url("raw.mp4"))

    def test_safe_stem(self) -> None:
        self.assertEqual(safe_stem("Mi video (1).mp4"), "Mi_video_1")


if __name__ == "__main__":
    unittest.main()
