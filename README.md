# ai-local-clipping

CLI local que convierte videos largos (pódcast, entrevistas, streams) en clips listos para TikTok, Instagram Reels y YouTube Shorts.

Pipeline:

1. Descarga el video con `yt-dlp` si se pasa una URL (mejor video + mejor audio).
2. Extrae y limpia el audio a WAV mono 16 kHz con `ffmpeg`: alto-paso 90 Hz, anti-hiss, `afftdn`, realce de presencia vocal a 2.8 kHz, compresión con makeup y `dynaudnorm` frase por frase. Si están instalados, aplica DeepFilterNet antes de FFmpeg; con `--isolate-vocals` usa demucs. Nunca elimina silencios, así que los timestamps siguen alineados con el video.
3. Transcribe con **Whisper** (palabra por palabra cuando el backend lo permite; `faster-whisper` usa VAD con 500 ms de silencio mínimo).
4. Enriquece la transcripción con pausas dramáticas (`[PAUSA x.xs]`), énfasis (`[ÉNFASIS ALTO]`) y reacciones (`[REACCIÓN/RISAS]`) calculadas sobre el WAV limpio.
5. Resume y elige clips con **Ollama** usando el `--style` editorial (`auto`, `humor`, `noticias`, `vender`, `educar`, `debate`, `gaming`) y, si hay un modelo multimodal como `qwen3-vl:8b`, valida 4 keyframes por candidato con `--vision`. El perfil `gaming` (o `auto` detectado en video vertical con exclamaciones y picos de energía) busca Tensión → Clímax → Reacción y usa un rango de 20-42 s.
6. Escribe `clips.json` con `estilo_aplicado`, ranking, `score_total`, razonamiento, evidencia acústica/visual y textos listos para publicar; junto a cada MP4 guarda un `.json` de metadatos.
 7. Renderiza MP4 verticales 9:16: si `demucs` está disponible, aísla la voz del segmento y mezcla 100 % de voz con 12 % de ambiente; aplica siempre la cadena de alto impacto vocal (`volume=6dB`, realce 2.6/4.5 kHz, compresión 5:1, `dynaudnorm` rápido y `alimiter`) y codifica AAC 192k/48 kHz estéreo. También restaura video SD con `--enhance-video fast` (o `ai` vía Vulkan), encuadra al hablante cuando la detección de rostro es confiable y, si no, muestra el cuadro completo sobre un fondo difuminado. Con `--viral-fx on` (por defecto) añade un HookBanner editorial de 0.0 a 3.8 s (tarjeta oscura semitransparente, texto blanco y últimas dos palabras en menta), subtítulos karaoke de máximo 3 palabras con pop escalado, micro-zoom de 1.0x a 1.12x en énfasis/pausas (excepto en fuentes verticales nativas), barra de progreso superior en `y=0` y SFX opcionales desde `assets/sfx`; guarda además un SRT relativo al clip.
 8. Detecta diagramas, diapositivas o código con el modelo multimodal (o OpenCV como respaldo) y aplica `--diagram-zoom auto` una vista general completa de 0 a 2.5 s, un recorrido horizontal 1.75x hasta 3.5 s antes del final y un cierre panorámico sin recortar bordes; si además se detecta una webcam en esquina, `--split-webcam auto` la muestra a 440x260 en `y=150` con borde menta sobre el diagrama centrado. El audio se mide en 1 s: con piso < -45 dB usa una cadena de estudio transparente (`loudnorm`) y sin demucs; con más ruido activa el realce vocal. El audio limpio se calcula una sola vez y se reutiliza en cada render con `-ss`/`-t`.
 El render usa `h264_vaapi` cuando hay `/dev/dri/renderD128`, con fallback a `libx264 -preset veryfast`, y procesa hasta 2 clips en paralelo.

## Requisitos

```bash
sudo apt install ffmpeg
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen2.5:7b
ollama pull qwen3-vl:8b
ollama serve
```

## Instalación del CLI

```bash
python3 -m pip install -e .
```

Esto instala el comando `clips` y `openai-whisper`. Opcionalmente puedes usar `faster-whisper` (`python3 -m pip install -e '.[faster]'`), aislamiento de voz (`python3 -m pip install -e '.[voice]'`) o `realesrgan-ncnn-vulkan` para `--enhance-video ai`; el CLI detecta automáticamente lo que esté disponible y usa FFmpeg como respaldo. Para la capa visual, `qwen3-vl:8b` se detecta solo; todas las peticiones a Ollama mantienen el modelo cargado 5 minutos (`keep_alive: "5m"`) para evitar recargas entre etapas.

## Uso

```bash
clips "https://www.youtube.com/watch?v=yAMoc5c05V4"
```

Si el video fue grabado con OBS, puedes conservar la referencia al original:

```bash
clips grabacion.mp4 --source-url "https://youtu.be/yAMoc5c05V4" --render-dir clips
```

O con un archivo local:

```bash
clips raw.mp4 -o clips.json --clips 6 --min-duration 20 --max-duration 55
```

Renderizar además los MP4 y los SRT:

```bash
clips raw.mp4 --render-dir clips --clips 5
```

Sin instalar el paquete:

```bash
python3 -m clipper raw.mp4 -o clips.json
```

Reutilizar una transcripción (evita volver a correr Whisper):

```bash
clips raw.mp4 --transcript transcripcion.json --transcript-out transcripcion.json
```

Para videos muy largos, el CLI divide la transcripción en bloques de `--chunk-chars` caracteres, propone candidatos por bloque y luego pide a Ollama que escoja los mejores.

## Opciones principales

| Opción | Descripción | Default |
| --- | --- | --- |
| `--source-url` | URL original para enlazar cada clip con su minuto exacto | detectada de la entrada |
| `--clips` | Número máximo de clips | `5` |
| `--min-duration` / `--max-duration` | Rango de duración en segundos (con `--style gaming`: `20` / `42`) | `15` / `60` |
| `--whisper-model` | Modelo de Whisper | `small` |
| `--whisper-language` | Idioma o `auto` | `es` |
| `--whisper-backend` | `auto`, `openai`, `faster` o `cli` | `auto` |
| `--ollama-model` | Modelo de Ollama | `qwen2.5:7b` |
| `--ollama-host` | URL del servidor | `http://127.0.0.1:11434` |
| `--style` | `auto`, `humor`, `noticias`, `vender`, `educar`, `debate` o `gaming` | `auto` |
| `--vision` / `--no-vision` | Forzar / desactivar verificación visual multimodal | automática |
| `--vision-model` | Modelo multimodal de Ollama | detección automática |
| `--diagram-zoom` | `auto`, `on` u `off` (Pan & Scan guiado sobre diagramas/diapositivas/código) | `auto` |
| `--split-webcam` | `auto`, `on` u `off` (webcam del presentador en recuadro superior) | `auto` |
| `--render-dir` | Carpeta para MP4 + SRT + TXT de publicación + JSON de metadatos | desactivado |
| `--framing` | `auto`, `face`, `fit` o `center` (encuadre del hablante) | `auto` |
| `--no-subtitles` | No quemar los subtítulos en el MP4 | desactivado |
| `--no-clean-audio` | No limpiar el audio antes de transcribir | desactivado (limpieza activa) |
| `--no-vocal-boost` | No aplicar la cadena de alto impacto vocal en el MP4 final | desactivado (realce activo) |
| `--isolate-vocals` / `--no-isolate-vocals` | Forzar / desactivar aislamiento con demucs en la preparación del audio | automático si demucs está disponible |
| `--audio-mode` | `auto`, `studio` o `noisy` (según piso de ruido medido) | `auto` |
| `--enhance-video` | `auto`, `off`, `fast` o `ai` (la restauración IA solo se aplica si la fuente mide < 720p) | `auto` |
| `--viral-fx` | `on` u `off` (hook, karaoke, micro-zoom, barra y SFX) | `on` |
| `--verbose` | Imprimir el comando FFmpeg final | desactivado |
| `--no-vertical` | Conservar la relación de aspecto original | desactivado |

Los SFX virales se leen de `assets/sfx/` (o de la carpeta indicada internamente). Se usan `whoosh.wav` al inicio, `pop.wav`/`ding.wav` en los picos de énfasis, `ding.wav` de respaldo y `bass_drop.wav` en pausas de estilos `humor`/`debate`; se mezclan a -14 dB sobre la voz realzada. Si falta algún archivo, el render continúa solo con los efectos disponibles.

Con `--diagram-zoom auto`, el modelo multimodal (o la detección OpenCV de respaldo) marca la región del diagrama: la diapositiva se ajusta al ancho (`force_original_aspect_ratio=decrease` + `pad`) para no perder bordes, se muestra completa de 0 a 2.5 s, se recorre de izquierda a derecha a 1.75x hasta 3.5 s antes del final y vuelve a la vista completa para cerrar. Con `--split-webcam auto`, si se detecta una webcam en alguna esquina, el video principal se escala a 1080x960 en el centro, la webcam se recorta y se muestra a 440x260 en `y=150` con borde menta. `on` fuerza la opción (usando una región central por defecto y la esquina inferior derecha si no hay detección) y `off` la desactiva.

Si la fuente ya es vertical (`height >= width`), se detecta como nativa y se escala directamente a 1080x1920: no se aplican fondo difuminado, recorte facial, Pan & Scan de diagramas ni split-webcam, para no deformar ni perder acción de gameplay. En ese modo se usa audio `studio` transparente sin demucs, se conserva el sonido ambiente del juego y el perfil `auto` puede seleccionar `gaming`.

El perfil de audio se decide con `volumedetect` sobre el primer segundo: `studio` aplica `highpass=f=75,loudnorm=I=-15:TP=-1.5:LRA=11,alimiter=limit=0.95` y omite demucs, `afftdn`, ecualización agresiva y `dynaudnorm`; `noisy` conserva el realce vocal. El audio limpio se calcula una sola vez y se reutiliza en los tres renders mediante `-ss`/`-t`, en vez de repetir demucs por clip. `faster-whisper` usa `beam_size=1` y VAD con 600 ms, los keyframes se extraen en una sola llamada FFmpeg a 384 px y las peticiones de visión mantienen el modelo cargado 5 minutos.

Cuando hay `source_url` (detectada de la entrada o indicada con `--source-url`), cada `clip_XX.txt` contiene el título, la descripción de redes, el enlace al minuto exacto (`&t=N s` en YouTube, `?t=N s` en youtu.be) y un bloque separado con el primer comentario fijado recomendado. El mismo enlace, el minuto `MM:SS` y el texto completo se guardan en `clip_XX.json` y en `clips.json`.

El CLI imprime al final los tiempos por etapa (`duracion`, `perfil`, `audio_prep`, `whisper`, `audio_enrich`, `ollama_model_check`, `ollama_texto`, `vision_llm`, `render_total`, `json_output`, `total`) y `clips.json` incluye `tiempos_segundos` y `tiempos_render_por_clip` con `face`, `layout_cv`, `subtitulos`, `ffmpeg_render` y `metadata` por clip.

## Formato de `clips.json`

```json
{
  "fuente": {"entrada": "raw.mp4", "duracion_segundos": 771.25, "idioma": "es"},
  "resumen": {
    "general": "Resumen del contenido...",
    "temas": ["tema 1", "tema 2"],
    "tono": "divulgativo",
    "audiencia": "creadores de contenido"
  },
  "clips": [
    {
      "id": "clip_01",
      "inicio": 42.5,
      "fin": 78.2,
      "duracion_segundos": 35.7,
      "inicio_tiempo": "00:00:42.500",
      "fin_tiempo": "00:01:18.200",
      "titulo": "Gancho del clip",
      "descripcion": "Texto listo para publicar",
      "tags": ["tag1", "tag2"],
      "hashtags": "#tag1 #tag2",
      "transcripcion": "Lo que se dice en el fragmento",
      "formato": {
        "relacion_aspecto": "9:16",
        "resolucion": "1080x1920",
        "encuadre": "rostro",
        "recorte": [320, 0, 405, 720],
        "subtitulos_incrustados": true
      },
      "publicacion": {
        "tiktok": {"titulo": "...", "descripcion": "..."},
        "instagram_reels": {"titulo": "...", "descripcion": "..."},
        "youtube_shorts": {"titulo": "...", "descripcion": "...", "tags": "..."}
      },
      "archivo": "clips/clip_01.mp4",
      "subtitulos": "clips/clip_01.srt"
    }
  ]
}
```

## Pruebas

```bash
python3 -m unittest discover -s tests -v
```
