from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from clipper.media import is_url, safe_stem
from clipper.ollama import OllamaClient, chunk_transcript, selection_prompt
from clipper.schema import (
    ClipSettings,
    Transcript,
    build_output,
    dedupe_candidates,
    format_timestamp,
    heuristic_candidates,
    normalize_selection,
    srt_from_transcript,
)


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

    def test_host_normalization(self) -> None:
        self.assertEqual(OllamaClient("127.0.0.1:11434", "m").host, "http://127.0.0.1:11434")


class MiscTests(unittest.TestCase):
    def test_is_url(self) -> None:
        self.assertTrue(is_url("https://www.youtube.com/watch?v=yAMoc5c05V4"))
        self.assertFalse(is_url("raw.mp4"))

    def test_safe_stem(self) -> None:
        self.assertEqual(safe_stem("Mi video (1).mp4"), "Mi_video_1")


if __name__ == "__main__":
    unittest.main()
