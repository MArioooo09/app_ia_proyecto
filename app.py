import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 1. Configuración de la página
st.set_page_config(page_title="IA Rendimiento Pro", page_icon="🎓", layout="wide")
st.title("🎓 IA de Análisis de Rendimiento Estudiantil")
st.write("Este sistema predice si un alumno aprobará basándose en su historial académico y datos sociales.")

# 2. Cargar modelo y columnas
@st.cache_resource
def load_assets():
    model = joblib.load('modelo_arbol.joblib')
    columns = joblib.load('model_columns.joblib')
    return model, columns

try:
    model, model_columns = load_assets()
    st.sidebar.success("✅ Modelo cargado correctamente")
except Exception as e:
    st.sidebar.error("❌ Error al cargar archivos .joblib")
    st.stop()

# 3. Formulario de entrada de datos
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("📊 Notas Actuales")
    g1 = st.slider("Nota Periodo 1 (G1)", 0, 20, 10)
    g2 = st.slider("Nota Periodo 2 (G2)", 0, 20, 10)
    g3 = st.slider("Nota Examen Final (G3)", 0, 20, 10)

with col2:
    st.subheader("🏫 Datos Académicos")
    failures = st.number_input("Fracasos anteriores", 0, 4, 0)
    absences = st.slider("Ausencias totales", 0, 93, 5)
    studytime = st.selectbox("Tiempo estudio semanal", [1, 2, 3, 4], format_func=lambda x: f"{x} (Horas: {['<2','2-5','5-10','>10'][x-1]})")

with col3:
    st.subheader("👨‍👩‍👧 Entorno Familiar")
    medu = st.slider("Educación Madre (0-4)", 0, 4, 2)
    fedu = st.slider("Educación Padre (0-4)", 0, 4, 2)
    higher = st.radio("¿Quiere estudios superiores?", ["Sí", "No"])

# 4. Lógica de Predicción
if st.button("🚀 Realizar Análisis"):
    # Creamos un DataFrame con una fila llena de ceros
    entrada = pd.DataFrame(np.zeros((1, len(model_columns))), columns=model_columns)
    
    # Rellenamos las variables que tenemos
    entrada['G1'] = g1
    entrada['G2'] = g2
    entrada['G3'] = g3
    entrada['failures'] = failures
    entrada['absences'] = absences
    entrada['studytime'] = studytime
    entrada['Medu'] = medu
    entrada['Fedu'] = fedu
    
    # Manejo de la variable categórica 'higher'
    if higher == "Sí":
        if 'higher_yes' in entrada.columns: entrada['higher_yes'] = 1
    else:
        if 'higher_no' in entrada.columns: entrada['higher_no'] = 1

    # Predicción
    pred = model.predict(entrada)
    prob = model.predict_proba(entrada)

    st.markdown("---")
    if pred[0] == 1:
        st.balloons()
        st.success(f"### 🎉 RESULTADO: PROBABLE APROBADO")
        st.write(f"Confianza de la IA: {prob[0][1]*100:.1f}%")
    else:
        st.error(f"### 📉 RESULTADO: RIESGO DE SUSPENSO")
        st.write(f"Confianza de la IA: {prob[0][0]*100:.1f}%")
    
    # Mensaje de ayuda basado en notas
    if (g1 + g2 + g3) < 30:
        st.warning("Aviso: La suma de notas es baja. La IA detecta que el esfuerzo actual no es suficiente para el aprobado.")

st.sidebar.markdown("---")
st.sidebar.info("Este modelo analiza G1, G2 y G3 como factores críticos de éxito.")
