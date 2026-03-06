import modal

# 1. Definir la imagen con dependencias
image = (
    modal.Image.debian_slim()
    .pip_install("openai-whisper", "torch", "torchaudio", "ffmpeg-python")
    .apt_install("ffmpeg")
)

app = modal.App("audio-processor-v1", image=image)

# 2. Clase de Inferencia (se carga el modelo una sola vez)
@app.cls(gpu="T4", container_idle_timeout=30)
class Transcriber:
    @modal.enter()
    def load_model(self):
        import whisper
        self.model = whisper.load_model("base")

    @modal.method()
    def transcribe(self, audio_url: str):
        # El análisis masivo ocurre aquí
        result = self.model.transcribe(audio_url)
        text = result["text"]
        
        # Análisis adicional (puedes meter un LLM aquí mismo)
        analysis = f"Longitud: {len(text)} caracteres."
        
        return {"text": text, "analysis": analysis}

# 3. Punto de entrada para procesamiento masivo
@app.local_entrypoint()
def main(urls: list[str]):
    transcriber = Transcriber()
    # .map() ejecuta todo en paralelo en la nube de Modal
    results = list(transcriber.transcribe.map(urls))
    
    for r in results:
        print(f"Transcripción terminada: {r['analysis']}")
