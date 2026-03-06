import modal
import os
import json
import pandas as pd
from datetime import datetime

# Configuración de la imagen con Streamlit y modelos de IA
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "faster-whisper", "torch", "transformers", 
        "streamlit", "pandas", "plotly"
    )
    .apt_install("ffmpeg")
)

volume = modal.Volume.from_name("resultados-analisis-audio", create_if_missing=True)
app = modal.App("dashboard-callcenter", image=image)

# --- CLASE PROCESADORA (GPU) ---
@app.cls(gpu="T4", volumes={"/data": volume})
class CallAnalyst:
    @modal.enter()
    def setup(self):
        from faster_whisper import WhisperModel
        from transformers import pipeline
        self.transcriber = WhisperModel("base", device="cuda", compute_type="float16")
        self.sentiment_pipe = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment", device=0)

    @modal.method()
    def process_call(self, filename: str, content: bytes):
        temp_path = f"/tmp/{filename}"
        with open(temp_path, "wb") as f:
            f.write(content)
        
        segments, info = self.transcriber.transcribe(temp_path, beam_size=5)
        text = " ".join([segment.text for segment in segments])
        sentiment = self.sentiment_pipe(text[:512])[0]

        res = {
            "archivo": filename,
            "idioma": info.language,
            "sentimiento": sentiment['label'], # "1 star" a "5 stars"
            "score": sentiment['score'],
            "texto": text,
            "palabras": len(text.split()),
            "fecha": datetime.now().strftime("%Y-%m-%d %H:%M")
        }
        
        # Guardar en volumen
        with open(f"/data/{filename}_res.json", "w") as f:
            json.dump(res, f)
        return res

# --- INTERFAZ WEB (STREAMLIT) ---
@app.function(volumes={"/data": volume})
@modal.wsgi_app()
def ui():
    import streamlit as st
    import plotly.express as px

    st.set_page_config(page_title="Dashboard CallCenter AI", layout="wide")
    st.title("📊 Análisis Masivo de Llamadas")

    # Sidebar: Cargar audios
    with st.sidebar:
        st.header("Cargar Audios")
        uploaded_files = st.file_uploader("Sube archivos .mp3 o .wav", accept_multiple_files=True)
        if st.button("🚀 Procesar Todo en Modal"):
            if uploaded_files:
                analyst = CallAnalyst()
                payloads = [(f.name, f.read()) for f in uploaded_files]
                with st.spinner("Procesando en paralelo con GPU..."):
                    results = list(analyst.process_call.starmap(payloads))
                st.success(f"¡{len(results)} llamadas procesadas!")

    # Cuerpo Principal: Dashboard
    st.subheader("Historial de Análisis")
    
    # Leer datos del volumen
    data = []
    for file in os.listdir("/data"):
        if file.endswith(".json"):
            with open(f"/data/{file}", "r") as f:
                data.append(json.load(f))

    if data:
        df = pd.DataFrame(data)
        
        # Métricas rápidas
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Llamadas", len(df))
        col2.metric("Promedio Palabras", int(df["palabras"].mean()))
        col3.metric("Idioma Principal", df["idioma"].mode()[0])

        # Gráfico de Sentimiento
        fig = px.pie(df, names='sentimiento', title='Distribución de Satisfacción del Cliente')
        st.plotly_chart(fig)

        # Tabla de datos
        st.dataframe(df[["fecha", "archivo", "idioma", "sentimiento", "palabras"]])
    else:
        st.info("Aún no hay datos. Sube audios en el panel de la izquierda.")
