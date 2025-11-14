import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import time

# ------------------ CONFIGURACIÓN DE LA PÁGINA ------------------
st.set_page_config(
    page_title="Clasificador de estilos de maquillaje (TFM)",
    page_icon="💄",
    layout="wide"
)

# ------------------ CONSTANTES ------------------
CLASSES = ['natural','soft_glam','glam_dia','glam_noche','artistico','tematico','editorial']

# ------------------ CARGA DEL MODELO ------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("model.h5", compile=False)
    return model

# ------------------ SIDEBAR ------------------
st.sidebar.title("💄 Clasificador TFM")
st.sidebar.markdown(
    """
    Esta demo forma parte del **TFM** sobre reconocimiento automático 
    de **estilos de maquillaje** mediante *deep learning*.

    **Cómo usarla:**
    1. Sube una foto de un rostro maquillado (frontal o 3/4).
    2. La red neuronal (EfficientNet-B0 fine-tuned) 
       clasificará la imagen en una de las **7 categorías**.
    3. Se muestran las probabilidades y el *top-3* de estilos más probables.
    """
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Clases del modelo:**")
for c in CLASSES:
    st.sidebar.write(f"- `{c}`")

st.sidebar.markdown("---")
st.sidebar.caption("Modelo ejecutado en Streamlit • Prototipo académico, no uso comercial.")

# ------------------ CONTENIDO PRINCIPAL ------------------
st.title("🔍 Clasificador de estilos de maquillaje (TFM)")

st.write("Sube una imagen de un rostro maquillado y el modelo intentará clasificar el estilo.")

uploaded_file = st.file_uploader(
    "📂 Subir imagen (.jpg / .jpeg / .png)",
    type=["jpg", "jpeg", "png"]
)

col_left, col_right = st.columns([1.1, 1])

if uploaded_file is not None:
    # --------- Columna izquierda: imagen ---------
    with col_left:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Imagen subida", use_column_width=True)

    # --------- Columna derecha: resultados ---------
    with col_right:
        with st.spinner("Clasificando estilo de maquillaje..."):
            t0 = time.time()
            model = load_model()

            img_resized = image.resize((224, 224))
            x = np.array(img_resized) / 255.0
            x = np.expand_dims(x, axis=0)

            probs = model.predict(x, verbose=0)[0]
            t1 = time.time()

        # Índice de clase predicha
        pred_idx = int(np.argmax(probs))
        pred_class = CLASSES[pred_idx]
        pred_conf = float(probs[pred_idx])

        st.subheader(f"🎯 Predicción principal: **{pred_class}**")
        st.caption(f"Confianza aproximada: **{pred_conf:.2%}**  •  Tiempo de inferencia: {(t1 - t0):.2f} s")

        # Top-3 clases
        st.markdown("**🏅 Top-3 estilos más probables:**")
        top3_idx = np.argsort(probs)[-3:][::-1]
        for rank, idx in enumerate(top3_idx, start=1):
            st.write(f"{rank}. **{CLASSES[idx]}** — {probs[idx]:.2%}")

        st.markdown("---")
        st.markdown("**Distribución de probabilidades por clase:**")

        # Probabilidades en forma de tabla + gráfico
        prob_table = { "clase": CLASSES, "probabilidad": probs }
        st.bar_chart(data=probs)

        st.caption(
            "Nota: esta herramienta es un **prototipo experimental**. "
            "El rendimiento depende del tamaño y calidad del dataset usado para el entrenamiento."
        )

else:
    st.info("📌 Sube una imagen en la parte superior para realizar una predicción.")
