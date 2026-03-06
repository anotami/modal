import modal
import os
import json
import pandas as pd
from datetime import datetime

# 1. Imagen con todas las dependencias
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "faster-whisper", 
        "torch", 
        "transformers", 
        "streamlit", 
        "pandas", 
        "plotly"
    )
    .apt_install("ffmpeg")
)

volume = modal.Volume.from_name("resultados-analisis-audio", create_if_missing=True)
app = modal.App("dashboard-callcenter-gabriel", image=image)

# --- PROCESADOR DE AUDIOS (GPU) ---
@app.cls(
    gpu="T4", 
    scaledown_window=60, 
    volumes={"/data": volume}
)
class CallAnalyst:
    @modal.enter()
    def setup(self):
        from faster_whisper import WhisperModel
        from transformers import pipeline
        self.transcriber = WhisperModel("base", device="cuda", compute_type="float16")
        self.sentiment_pipe = pipeline(
            "sentiment-analysis", 
            model="nlptown/bert-base-multilingual-uncased-sentiment", 
            device=0
        )

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
            "sentimiento": sentiment['label'],
            "confianza": f"{sentiment['score']:.2f}",
            "texto": text,
            "palabras": len(text.split()),
            "fecha": datetime.now().strftime("%Y-%m-%d %H:%M")
        }
        
        with open(f"/data/{filename}_res.json", "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False)
        return res

# --- INTERFAZ WEB (STREAMLIT) ---
@app.function(volumes={"/data": volume})
@modal.wsgi_app()
def ui():
    import streamlit as st
    import plotly.express as px

    # Función interna para la interfaz de Streamlit
    def main_ui():
        st.set_page_config(page_title="Analista IA CallCenter", layout="wide")
        st.title("📊 Dashboard de Análisis de Llamadas")

        with st.sidebar:
            st.header("Cargar Nuevos Audios")
            uploaded_files = st.file_uploader("Sube archivos .mp3 o .wav", accept_multiple_files=True)
            
            if st.button("🚀 Iniciar Análisis"):
                if uploaded_files:
                    analyst = CallAnalyst()
                    payloads = [(f.name, f.read()) for f in uploaded_files]
                    with st.spinner("Procesando en la nube..."):
                        list(analyst.process_call.starmap(payloads))
                    st.success("¡Completado!")
                    st.rerun()

            st.divider()
            if st.button("🗑️ Borrar Historial"):
                for f in os.listdir("/data"):
                    os.remove(os.path.join("/data", f))
                st.info("Historial limpio.")
                st.rerun()

        # Cargar datos
        data = []
        if os.path.exists("/data"):
            for file in os.listdir("/data"):
                if file.endswith(".json"):
                    try:
                        with open(os.path.join("/data", file), "r", encoding="utf-8") as f:
                            data.append(json.load(f))
                    except: continue

        if data:
            df = pd.DataFrame(data)
            # Asegurar que las columnas existan antes de mostrar métricas
            if "palabras" in df.columns:
                col1, col2, col3 = st.columns(3)
                col1.metric("Llamadas", len(df))
                col2.metric("Promedio Palabras", int(df["palabras"].mean()))
                col3.metric("Idioma", df["idioma"].mode()[0].upper())

                fig = px.pie(df, names='sentimiento', title='Satisfacción Detectada')
                st.plotly_chart(fig)
                st.dataframe(df[["fecha", "archivo", "sentimiento", "texto"]])
        else:
            st.info("Sube audios en el panel izquierdo para empezar.")

    # Esta línea es crucial para evitar el error 'NoneType'
    from streamlit.web.server.server import Server
    return main_ui()
