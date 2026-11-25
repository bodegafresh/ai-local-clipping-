# Comandos

```bash
python3 -m pip install -e .
ollama pull qwen2.5:7b
ollama serve

clips "https://www.youtube.com/watch?v=yAMoc5c05V4" -o clips.json
clips raw.mp4 --render-dir clips --clips 5 --min-duration 15 --max-duration 60
python3 -m unittest discover -s tests -v
```
