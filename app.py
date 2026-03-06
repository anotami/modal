import modal
import os
import json
import pandas as pd
from datetime import datetime

# 1. Configuración del entorno con todas las dependencias necesarias
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

# Definición del volumen persistente para guardar las transcripciones y análisis
volume = modal.Volume.from_name("resultados-analisis-audio", create_if_missing=True)
app = modal.App("dashboard-callcenter-gabriel", image=image)

# --- MOTOR DE INTELIGENCIA ARTIFICIAL (GPU) ---
@app.cls(
    gpu="T4", 
    scaledown_window=60, # Nombre actualizado para evitar el aviso de Deprecation
    volumes={"/data": volume}
)
class CallAnalyst:
    @modal.enter()
    def setup(self):
        from faster_whisper import WhisperModel
        from transformers import pipeline
        # Cargamos el modelo de transcripción en la GPU
        self.transcriber = WhisperModel("base", device="cuda", compute_type="float16")
        # Cargamos el analizador de sentimiento multilingüe
        self.sentiment_pipe = pipeline(
            "sentiment-analysis", 
            model="nlptown/bert-base-multilingual-uncased-sentiment", 
            device=0
        )

    @modal.method()
    def process_call(self, filename: str, content: bytes):
        print(f"--- Procesando archivo: {filename} ---")
        
        # Guardado temporal para procesamiento
        temp_path = f"/tmp/{filename}"
        with open(temp_path, "wb") as f:
            f.write(content)
        
        # Paso 1: Transcripción masiva rápida
        segments, info = self.transcriber.transcribe(temp_path, beam_size=5)
        text = " ".join([segment.text for segment in segments])
        
        # Paso 2: Análisis de sentimiento (escala 1-5 estrellas)
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
        
        # Guardado permanente en el volumen de Modal
        output_path = f"/data/{filename}_res.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False)
            
        return res

# --- INTERFAZ WEB PROFESIONAL (STREAMLIT) ---
@app.function(volumes={"/data": volume})
@modal.wsgi_app()
def ui():
    import streamlit as st
    import plotly.express as px

    st.set_page_config(page_title="Analista IA - Call Center Gabriel", layout="wide")
    st.title("📊 Dashboard de Análisis Masivo de Llamadas")

    # Panel lateral para la carga de audios
    with st.sidebar:
        st.header("Entrada de Audios")
        uploaded_files = st.file_uploader("Arrastra aquí tus archivos .mp3 o .wav", accept_multiple_files=True)
        
        if st.button("🚀 Iniciar Análisis en Nube"):
            if uploaded_files:
                analyst = CallAnalyst()
                payloads = [(f.name, f.read()) for f in uploaded_files]
                with st.spinner("Procesando en paralelo con GPU..."):
                    # Ejecución masiva en paralelo
                    list(analyst.process_call.starmap(payloads))
                st.success("¡Procesamiento masivo completado!")
                st.rerun()
            else:
                st.warning("Por favor, sube al menos un archivo.")

        st.divider()
        if st.button("🗑️ Limpiar Historial"):
            # Lógica para borrar archivos del volumen si es necesario
            for f in os.listdir("/data"):
                os.remove(os.path.join("/data", f))
            st.info("Historial eliminado.")
            st.rerun()

    # --- LECTURA Y VISUALIZACIÓN DE DATOS ---
    data = []
    if os.path.exists("/data"):
        for file in os.listdir("/data"):
            if file.endswith(".json"):
                try:
                    with open(os.path.join("/data", file), "r", encoding="utf-8") as f:
                        data.append(json.load(f))
                except Exception:
                    continue

    if data:
        df = pd.DataFrame(data)
        
        # Red de seguridad: Verificar que existan las columnas para evitar KeyError
        columnas_base = ["palabras", "idioma", "sentimiento"]
        if all(col in df.columns for col in columnas_base):
            # Fila de métricas clave (KPIs)
            m1, m2, m3 = st.columns(3)
            m1.metric("Total Llamadas", len(df))
            m2.metric("Promedio Palabras", int(df["palabras"].mean()))
            m3.metric("Idioma Predominante", df["idioma"].mode()[0].upper())

            # Gráficos dinámicos
            col_left, col_right = st.columns(2)
            with col_left:
                fig_pie = px.pie(df, names='sentimiento', title='Nivel de Satisfacción (1-5 Estrellas)',
                                 color_discrete_sequence=px.colors.qualitative.Pastel)
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col_right:
                st.subheader("Detalle de las Interacciones")
                st.dataframe(df[["fecha", "archivo", "sentimiento", "texto"]], height=400)
        else:
            st.warning("⚠️ Se detectaron archivos con formato antiguo. Por favor, limpia el historial y procesa audios nuevos.")
    else:
        st.info("👋 Bienvenido, Gabriel. Sube los audios del Call Center en el panel izquierdo para generar el análisis.")
