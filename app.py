import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# --- Configuración básica de la página ---
st.set_page_config(page_title="Makeup Style Classifier", layout="centered")
st.title("🔍 Clasificador de estilos de maquillaje (TFM)")

# Las 7 clases definidas en el modelo
CLASSES = ['natural','soft_glam','glam_dia','glam_noche','artistico','tematico','editorial']

@st.cache_resource
def load_model():
    # Carga el modelo SavedModel desde la carpeta ./model
    return tf.keras.models.load_model("model")

st.write("Sube una imagen de un rostro maquillado y el modelo intentará clasificar el estilo.")

file = st.file_uploader("📂 Subir imagen (.jpg / .jpeg / .png)", type=["jpg","jpeg","png"])

if file is not None:
    # Mostrar imagen original
    img = Image.open(file).convert("RGB")
    st.image(img, caption="Imagen subida", use_column_width=True)

    # Preprocesar para el modelo
    img_resized = img.resize((224, 224))
    x = np.array(img_resized) / 255.0
    x = np.expand_dims(x, axis=0)

    # Predicción
    try:
        model = load_model()
        probs = model.predict(x, verbose=0)[0]
        pred_idx = int(np.argmax(probs))
        pred_class = CLASSES[pred_idx]

        st.subheader(f"🎯 Predicción: **{pred_class}**")
        st.write("Distribución de probabilidades por clase:")
        st.bar_chart(probs)
    except Exception as e:
        st.error("No se pudo cargar el modelo. Asegúrate de que la carpeta `model` está en el mismo directorio que `app.py`.")
        st.exception(e)
