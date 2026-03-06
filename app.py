import modal
import os
import json
from datetime import datetime

# 1. Configuración del entorno en la nube
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "faster-whisper", 
        "torch", 
        "torchaudio"
    )
    .apt_install("ffmpeg") # Necesario para procesar audio
)

# Creamos o conectamos un volumen persistente para guardar resultados
volume = modal.Volume.from_name("resultados-analisis-audio", create_if_missing=True)

app = modal.App("transcriptor-masivo-v1", image=image)

# 2. Clase Procesadora con GPU
@app.cls(
    gpu="T4",               # GPU económica y eficiente para Whisper
    container_idle_timeout=60, 
    timeout=600,            # Tiempo máximo por audio (10 min)
    volumes={"/data": volume}
)
class AudioProcessor:
    @modal.enter()
    def setup(self):
        from faster_whisper import WhisperModel
        # Cargamos el modelo en la GPU (usamos 'base' para velocidad, 'medium' para precisión)
        self.model = WhisperModel("base", device="cuda", compute_type="float16")

    @modal.method()
    def process_audio(self, filename: str, content: bytes):
        print(f"--- Procesando: {filename} ---")
        
        # Guardar temporalmente el audio en el contenedor para que Whisper lo lea
        temp_path = f"/tmp/{filename}"
        with open(temp_path, "wb") as f:
            f.write(content)

        # A. Transcripción
        segments, info = self.model.transcribe(temp_path, beam_size=5)
        text = " ".join([segment.text for segment in segments])

        # B. Análisis Básico (Aquí puedes integrar un LLM si lo deseas)
        word_count = len(text.split())
        
        resultado = {
            "archivo": filename,
            "timestamp": datetime.now().isoformat(),
            "idioma_detectado": info.language,
            "probabilidad_idioma": info.language_probability,
            "conteo_palabras": word_count,
            "transcripcion": text
        }

        # C. Guardar en el Volumen Persistente de Modal
        output_path = f"/data/{filename.replace('.', '_')}_res.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(resultado, f, ensure_ascii=False, indent=4)

        os.remove(temp_path) # Limpiar temporal
        return resultado

# 3. Punto de entrada local (Ejecución desde tu PC)
@app.local_entrypoint()
def main(folder_path: str):
    if not os.path.isdir(folder_path):
        print(f"Error: La ruta '{folder_path}' no es una carpeta válida.")
        return

    # Listar audios locales
    extensions = (".mp3", ".wav", ".m4a", ".ogg", ".flac")
    files = [f for f in os.listdir(folder_path) if f.lower().endswith(extensions)]
    
    if not files:
        print("No se encontraron archivos de audio.")
        return

    print(f"🚀 Iniciando procesamiento masivo de {len(files)} archivos...")
    
    processor = AudioProcessor()
    
    # Preparar datos para enviar a la nube
    payloads = []
    for f in files:
        path = os.path.join(folder_path, f)
        with open(path, "rb") as audio_file:
            payloads.append((f, audio_file.read()))

    # .starmap() ejecuta todo en paralelo en la infraestructura de Modal
    # Si envías 100 audios, Modal levantará varias GPUs al mismo tiempo
    results = list(processor.process_audio.starmap(payloads))

    print("\n✅ PROCESAMIENTO COMPLETADO")
    for r in results:
        print(f"- {r['archivo']}: {r['conteo_palabras']} palabras analizadas.")
    
    print("\n📂 Los resultados están guardados en el volumen 'resultados-analisis-audio' de Modal.")
