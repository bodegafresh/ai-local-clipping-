# ai-local-clipping

CLI local que convierte videos largos (pódcast, entrevistas, streams) en clips listos para TikTok, Instagram Reels y YouTube Shorts.

Pipeline:

1. Descarga el video con `yt-dlp` si se pasa una URL (mejor video + mejor audio).
2. Extrae audio a 16 kHz con `ffmpeg`.
3. Transcribe con **Whisper** (palabra por palabra cuando el backend lo permite).
4. Resume y elige los mejores fragmentos con **Ollama**.
5. Escribe `clips.json` con tiempos, transcripción, título, descripción, tags y textos por plataforma.
6. Opcionalmente renderiza MP4 verticales 9:16 centrados en el rostro del hablante, con subtítulos grandes quemados en el video y un SRT relativo al clip como archivo independiente.

## Requisitos

```bash
sudo apt install ffmpeg
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen2.5:7b
ollama serve
```

## Instalación del CLI

```bash
python3 -m pip install -e .
```

Esto instala el comando `clips` y `openai-whisper`. Opcionalmente puedes usar `faster-whisper` (`python3 -m pip install faster-whisper`); el CLI lo detecta automáticamente si está disponible.

## Uso

```bash
clips "https://www.youtube.com/watch?v=yAMoc5c05V4"
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
| `--clips` | Número máximo de clips | `5` |
| `--min-duration` / `--max-duration` | Rango de duración en segundos | `15` / `60` |
| `--whisper-model` | Modelo de Whisper | `small` |
| `--whisper-language` | Idioma o `auto` | `es` |
| `--whisper-backend` | `auto`, `openai`, `faster` o `cli` | `auto` |
| `--ollama-model` | Modelo de Ollama | `qwen2.5:7b` |
| `--ollama-host` | URL del servidor | `http://127.0.0.1:11434` |
| `--render-dir` | Carpeta para MP4 + SRT | desactivado |
| `--framing` | `auto`, `face` o `center` (encuadre del hablante) | `auto` |
| `--no-subtitles` | No quemar los subtítulos en el MP4 | desactivado |
| `--no-vertical` | Conservar la relación de aspecto original | desactivado |

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
