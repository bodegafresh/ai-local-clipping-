# Efectos de sonido para `--viral-fx`

Coloca aquí archivos WAV con estos nombres exactos; el render los detecta automáticamente y omite los que falten:

- `whoosh.wav`: entrada del HookBanner en `t=0.05s`.
- `pop.wav`: primer pico de énfasis del clip.
- `ding.wav`: segundo pico de énfasis (o único si no hay picos y el archivo existe).
- `bass_drop.wav`: justo antes de una pausa dramática cuando el estilo es `humor` o `debate`.

Todos se mezclan a `-14 dB` respecto a la voz y con `amix=duration=first` para preservar la duración del clip.
