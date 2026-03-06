import modal
import os
import json
from datetime import datetime

# 1. Configuración del entorno (Añadimos librerías para el análisis)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "faster-whisper", 
        "torch", 
        "torchaudio",
        "transformers" # Para el análisis de texto
    )
    .apt_install("ffmpeg")
)

volume = modal.Volume.from_name("resultados-analisis-audio", create_if_missing=True)
app = modal.App("transcriptor-analista-v1", image=image)

@app.cls(
    gpu="T4", 
    container_idle_timeout=60, 
    volumes={"/data": volume}
)
class CallAnalyst:
    @modal.enter()
    def setup(self):
        from faster_whisper import WhisperModel
        from transformers import pipeline
        
        # Modelo para transcripción
        self.transcriber = WhisperModel("base", device="cuda", compute_type="float16")
        
        # Modelo para análisis de sentimiento (multilingüe)
        self.sentiment_pipe = pipeline(
            "sentiment-analysis", 
            model="nlptown/bert-base-multilingual-uncased-sentiment",
            device=0 # Usa la GPU
        )

    @modal.method()
    def process_call(self, filename: str, content: bytes):
        print(f"--- Procesando llamada: {filename} ---")
        
        temp_path = f"/tmp/{filename}"
        with open(temp_path, "wb") as f:
            f.write(content)

        # A. Transcripción
        segments, info = self.transcriber.transcribe(temp_path, beam_size=5)
        text = " ".join([segment.text for segment in segments])

        # B. Análisis de la llamada
        # El modelo de sentimiento devuelve de 1 a 5 estrellas
        sentiment_result = self.sentiment_pipe(text[:512])[0] # Analiza el inicio de la llamada
        
        # C. Resumen y Métricas
        resultado = {
            "archivo": filename,
            "fecha_proceso": datetime.now().isoformat(),
            "idioma": info.language,
            "transcripcion": text,
            "analisis": {
                "puntuacion_sentimiento": sentiment_result['label'], # Ej: "1 star" (molesto) a "5 stars" (satisfecho)
                "confianza_analisis": f"{sentiment_result['score']:.2f}",
                "duracion_estimada_palabras": len(text.split()),
                "alerta_calidad": "REVISAR" if "1 star" in sentiment_result['label'] else "NORMAL"
            }
        }

        # D. Guardar en el Volumen
        output_path = f"/data/{filename.replace('.', '_')}_analisis.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(resultado, f, ensure_ascii=False, indent=4)

        return resultado

@app.local_entrypoint()
def main(folder_path: str):
    import os
    if not os.path.isdir(folder_path): return

    files = [f for f in os.listdir(folder_path) if f.lower().endswith((".mp3", ".wav"))]
    analyst = CallAnalyst()
    
    payloads = []
    for f in files:
        with open(os.path.join(folder_path, f), "rb") as audio:
            payloads.append((f, audio.read()))

    results = list(analyst.process_call.starmap(payloads))

    print("\n✅ ANALISIS MASIVO COMPLETADO")
    for r in results:
        status = "⚠️" if r['analisis']['alerta_calidad'] == "REVISAR" else "✅"
        print(f"{status} Archivo: {r['archivo']} | Sentimiento: {r['analisis']['puntuacion_sentimiento']}")
