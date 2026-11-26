# Comandos

```bash
python3 -m pip install -e .
ollama pull qwen2.5:7b
ollama pull qwen3-vl:8b
ollama serve

clips "https://www.youtube.com/watch?v=yAMoc5c05V4" -o clips.json
clips grabacion_obs.mp4 --source-url "https://youtu.be/yAMoc5c05V4" --render-dir clips_con_enlace
clips raw.mp4 --render-dir clips --clips 5 --min-duration 15 --max-duration 60
clips raw.mp4 --render-dir clips --framing face
clips raw.mp4 --render-dir clips --framing fit --no-subtitles
clips raw.mp4 --render-dir clips --style humor --vision
clips raw.mp4 --render-dir clips --style noticias --no-vision
clips gameplay_vertical.mp4 --render-dir clips_gaming --style gaming
clips gameplay_vertical.mp4 --render-dir clips_gaming_auto --style auto --vision
clips raw.mp4 --render-dir clips --keep-workdir
clips raw.mp4 --render-dir clips --isolate-vocals --no-vocal-boost
clips raw.mp4 --render-dir clips --isolate-vocals --clean-render-audio
clips raw.mp4 --render-dir clips --enhance-video fast
clips raw.mp4 --render-dir clips --enhance-video ai --clean-render-audio
clips raw.mp4 --render-dir clips --viral-fx on --verbose
clips raw.mp4 --render-dir clips --viral-fx off
clips raw.mp4 --render-dir clips --diagram-zoom auto --split-webcam auto
clips raw.mp4 --render-dir clips --diagram-zoom on --split-webcam on
clips raw.mp4 --render-dir clips --diagram-zoom off --split-webcam off
clips raw.mp4 --render-dir clips --audio-mode studio
clips raw.mp4 --render-dir clips --audio-mode noisy --isolate-vocals
clips raw.mp4 --no-clean-audio -o clips_sin_limpieza.json
python3 -m pytest -q
```

Con `--source-url` (o con una URL de entrada), cada `clip_XX.txt` incluye el enlace `&t=<segundos>s` / `?t=<segundos>s` y el bloque separado de primer comentario fijado. El CLI imprime los tiempos por etapa y `clips.json` guarda `tiempos_segundos` y `tiempos_render_por_clip`.

El estilo `gaming` configura 20-42 s, prioriza Tensión → Clímax → Reacción y, en fuentes verticales nativas, escala directamente a 1080x1920 sin blur, recorte facial, Pan & Scan ni split-webcam; el audio usa la cadena `studio` sin demucs para conservar el ambiente del juego. `--style auto` detecta gaming cuando el video es vertical y hay exclamaciones cortas con etiquetas de energía/reacción.
