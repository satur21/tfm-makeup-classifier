import os
import json
import tempfile

import cv2
import numpy as np
import pandas as pd
from PIL import Image

import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import mediapipe as mp

# =====================================================
# 1. CONFIGURACIÓN GLOBAL MEDIAPIPE / FACE MESH
# =====================================================

mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    refine_landmarks=True,
    max_num_faces=1,
    min_detection_confidence=0.5,
)


# =====================================================
# 2. FUNCIONES AUXILIARES VISIÓN POR COMPUTADOR
# =====================================================

def load_image_bgr(path: str):
    img = cv2.imread(path)
    if img is None:
        return None
    return img


def get_face_landmarks(image_bgr):
    """
    Devuelve lista de (x, y) en píxeles de los 468 landmarks.
    Si no detecta rostro, devuelve None.
    """
    h, w = image_bgr.shape[:2]
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(img_rgb)

    if not results.multi_face_landmarks:
        return None

    face_landmarks = results.multi_face_landmarks[0]
    pts = []
    for lm in face_landmarks.landmark:
        x_px = int(lm.x * w)
        y_px = int(lm.y * h)
        pts.append((x_px, y_px))
    return np.array(pts, dtype=np.int32)


def get_face_bbox(landmarks, image_shape):
    """
    Devuelve bounding box [x0, y0, x1, y1] del rostro,
    expandida ligeramente.
    """
    h, w = image_shape[:2]
    xs = landmarks[:, 0]
    ys = landmarks[:, 1]
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()

    # Expandir un poco
    dx = int(0.05 * (x1 - x0))
    dy = int(0.10 * (y1 - y0))

    x0 = max(0, x0 - dx)
    y0 = max(0, y0 - dy)
    x1 = min(w - 1, x1 + dx)
    y1 = min(h - 1, y1 + dy)

    return x0, y0, x1, y1


def build_region_masks(img_shape, bbox):
    """
    Genera máscaras aproximadas:
      - face
      - skin ( = face - ojos - labios )
      - eyes
      - lips
      - cheeks
    No usa landmarks finos, solo proporciones dentro del bbox.
    """
    h, w = img_shape[:2]
    x0, y0, x1, y1 = bbox
    bw = x1 - x0
    bh = y1 - y0

    masks = {}
    masks["face"] = np.zeros((h, w), dtype=np.uint8)

    # Rostro como rectángulo principal
    masks["face"][y0:y1, x0:x1] = 1

    # Zona ojos: franja superior del bbox
    eyes_y0 = y0 + int(0.20 * bh)
    eyes_y1 = y0 + int(0.45 * bh)
    eyes_x0 = x0 + int(0.10 * bw)
    eyes_x1 = x0 + int(0.90 * bw)

    masks["eyes"] = np.zeros((h, w), dtype=np.uint8)
    masks["eyes"][eyes_y0:eyes_y1, eyes_x0:eyes_x1] = 1

    # Zona labios: franja inferior del bbox
    lips_y0 = y0 + int(0.60 * bh)
    lips_y1 = y0 + int(0.90 * bh)
    lips_x0 = x0 + int(0.25 * bw)
    lips_x1 = x0 + int(0.75 * bw)

    masks["lips"] = np.zeros((h, w), dtype=np.uint8)
    masks["lips"][lips_y0:lips_y1, lips_x0:lips_x1] = 1

    # Zona pómulos: franja media, lados izquierdo y derecho
    cheeks_y0 = y0 + int(0.45 * bh)
    cheeks_y1 = y0 + int(0.70 * bh)

    cheeks_mask = np.zeros((h, w), dtype=np.uint8)
    # mejilla izquierda
    cheeks_mask[cheeks_y0:cheeks_y1, x0 + int(0.05 * bw): x0 + int(0.35 * bw)] = 1
    # mejilla derecha
    cheeks_mask[cheeks_y0:cheeks_y1, x0 + int(0.65 * bw): x0 + int(0.95 * bw)] = 1
    masks["cheeks"] = cheeks_mask

    # Skin aproximado: cara menos ojos y labios
    skin_mask = masks["face"].copy()
    skin_mask[masks["eyes"] == 1] = 0
    skin_mask[masks["lips"] == 1] = 0
    masks["skin"] = skin_mask

    return masks


def region_stats_hsv(img_hsv, mask):
    """
    Calcula H_mean, S_mean, V_mean, V_std sobre una máscara booleana.
    Escala H a [0, 1], S y V a [0, 1].
    """
    if mask.sum() < 30:
        return None

    H = img_hsv[:, :, 0][mask]
    S = img_hsv[:, :, 1][mask]
    V = img_hsv[:, :, 2][mask]

    H_norm = H / 179.0
    S_norm = S / 255.0
    V_norm = V / 255.0

    return {
        "H_mean": float(np.mean(H_norm)),
        "S_mean": float(np.mean(S_norm)),
        "V_mean": float(np.mean(V_norm)),
        "V_std": float(np.std(V_norm)),
    }


def edge_density(img_gray, mask):
    """
    Densidad de bordes Canny dentro de la máscara.
    """
    if mask.sum() < 30:
        return 0.0
    edges = cv2.Canny(img_gray, 100, 200)
    # Normalizar 0–1
    return float((edges[mask] > 0).mean())


# =====================================================
# 3. EXTRACTOR DE FEATURES (VERSIÓN COLAB ADAPTADA)
# =====================================================

def extract_makeup_features(image_path):
    """
    Versión extendida:
      - HSV en piel, ojos, labios, pómulos
      - Contrastes ojos/piel, labios/piel, pómulos/piel
      - Densidad de bordes en ojos
      - Índice global de intensidad de color
      - Features globales temático:
          global_skin_shift, non_skin_color_ratio,
          extreme_saturation_ratio, masklike_pattern_score, fx_ratio
      - Aliases: S_skin, S_eyes, S_cheeks, deltaH_ojos_piel, deltaV_ojos_piel
      - NUEVOS FEATURES NOCHE:
          eyes_dark_ratio, lips_dark_ratio, eyes_color_concentration
    """
    img_bgr = load_image_bgr(image_path)
    if img_bgr is None:
        print("No se pudo cargar la imagen:", image_path)
        return None

    landmarks = get_face_landmarks(img_bgr)
    if landmarks is None:
        print("No se detectó rostro en la imagen:", image_path)
        return None

    bbox = get_face_bbox(landmarks, img_bgr.shape)
    masks = build_region_masks(img_bgr.shape, bbox)

    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    H_norm = img_hsv[:, :, 0] / 179.0
    S_norm = img_hsv[:, :, 1] / 255.0
    V_norm = img_hsv[:, :, 2] / 255.0

    mask_face = masks["face"].astype(bool)
    mask_skin = masks["skin"].astype(bool)
    mask_eyes = masks["eyes"].astype(bool)
    mask_lips = masks["lips"].astype(bool)
    mask_cheeks = masks["cheeks"].astype(bool)

    stats_skin = region_stats_hsv(img_hsv, mask_skin)
    stats_eyes = region_stats_hsv(img_hsv, mask_eyes)
    stats_lips = region_stats_hsv(img_hsv, mask_lips)
    stats_cheeks = region_stats_hsv(img_hsv, mask_cheeks)

    if stats_skin is None:
        print("No se pudo obtener región de piel base en la imagen:", image_path)
        return None

    feats = {}

    # --- PIEL ---
    feats["skin_H_mean"] = stats_skin["H_mean"]
    feats["skin_S_mean"] = stats_skin["S_mean"]
    feats["skin_V_mean"] = stats_skin["V_mean"]
    feats["skin_V_std"] = stats_skin["V_std"]
    feats["S_skin"] = stats_skin["S_mean"]

    # --- LABIOS ---
    if stats_lips is not None:
        feats["lips_H_mean"] = stats_lips["H_mean"]
        feats["lips_S_mean"] = stats_lips["S_mean"]
        feats["lips_V_mean"] = stats_lips["V_mean"]
        feats["lips_V_std"] = stats_lips["V_std"]

        feats["contrast_lips_skin_S"] = stats_lips["S_mean"] - stats_skin["S_mean"]
        feats["contrast_lips_skin_V"] = stats_lips["V_mean"] - stats_skin["V_mean"]
    else:
        feats["lips_H_mean"] = np.nan
        feats["lips_S_mean"] = np.nan
        feats["lips_V_mean"] = np.nan
        feats["lips_V_std"] = np.nan
        feats["contrast_lips_skin_S"] = np.nan
        feats["contrast_lips_skin_V"] = np.nan

    # --- OJOS ---
    if stats_eyes is not None:
        feats["eyes_H_mean"] = stats_eyes["H_mean"]
        feats["eyes_S_mean"] = stats_eyes["S_mean"]
        feats["eyes_V_mean"] = stats_eyes["V_mean"]
        feats["eyes_V_std"] = stats_eyes["V_std"]

        feats["contrast_eye_skin_S"] = stats_eyes["S_mean"] - stats_skin["S_mean"]
        feats["contrast_eye_skin_V"] = stats_eyes["V_mean"] - stats_skin["V_mean"]

        feats["eyes_edge_density"] = edge_density(img_gray, mask_eyes)

        feats["S_eyes"] = stats_eyes["S_mean"]
        feats["deltaH_ojos_piel"] = stats_eyes["H_mean"] - stats_skin["H_mean"]
        feats["deltaV_ojos_piel"] = stats_eyes["V_mean"] - stats_skin["V_mean"]
    else:
        feats["eyes_H_mean"] = np.nan
        feats["eyes_S_mean"] = np.nan
        feats["eyes_V_mean"] = np.nan
        feats["eyes_V_std"] = np.nan
        feats["contrast_eye_skin_S"] = np.nan
        feats["contrast_eye_skin_V"] = np.nan
        feats["eyes_edge_density"] = np.nan
        feats["S_eyes"] = np.nan
        feats["deltaH_ojos_piel"] = 0.0
        feats["deltaV_ojos_piel"] = 0.0

    # --- PÓMULOS ---
    if stats_cheeks is not None:
        feats["cheeks_H_mean"] = stats_cheeks["H_mean"]
        feats["cheeks_S_mean"] = stats_cheeks["S_mean"]
        feats["cheeks_V_mean"] = stats_cheeks["V_mean"]
        feats["cheeks_V_std"] = stats_cheeks["V_std"]

        feats["contrast_cheeks_skin_V"] = stats_cheeks["V_mean"] - stats_skin["V_mean"]
        feats["contrast_cheeks_skin_S"] = stats_cheeks["S_mean"] - stats_skin["S_mean"]

        feats["S_cheeks"] = stats_cheeks["S_mean"]
    else:
        feats["cheeks_H_mean"] = np.nan
        feats["cheeks_S_mean"] = np.nan
        feats["cheeks_V_mean"] = np.nan
        feats["cheeks_V_std"] = np.nan
        feats["contrast_cheeks_skin_V"] = np.nan
        feats["contrast_cheeks_skin_S"] = np.nan
        feats["S_cheeks"] = np.nan

    # --- ÍNDICE GLOBAL DE INTENSIDAD DE COLOR ---
    diffs_S = []
    for key in ["lips_S_mean", "eyes_S_mean", "cheeks_S_mean"]:
        val = feats.get(key, np.nan)
        if not (val is None or np.isnan(val)):
            diffs_S.append(val - stats_skin["S_mean"])
    feats["global_color_intensity_index"] = float(np.mean(diffs_S)) if diffs_S else np.nan

    # ======================================================
    # FEATURES GLOBALES TEMÁTICO
    # ======================================================
    face_area = max(int(mask_face.sum()), 1)

    ref_H, ref_S, ref_V = 0.07, 0.35, 0.65
    global_skin_shift = np.sqrt(
        (stats_skin["H_mean"] - ref_H) ** 2
        + (stats_skin["S_mean"] - ref_S) ** 2
        + (stats_skin["V_mean"] - ref_V) ** 2
    )
    feats["global_skin_shift"] = float(global_skin_shift)

    skin_H = stats_skin["H_mean"]
    skin_S = stats_skin["S_mean"]
    skin_V = stats_skin["V_mean"]

    delta_skin = np.sqrt(
        (H_norm - skin_H) ** 2
        + (S_norm - skin_S) ** 2
        + (V_norm - skin_V) ** 2
    )

    non_skin_base = mask_face & (~mask_skin)
    high_color_mask = non_skin_base & (delta_skin > 0.25) & (S_norm > skin_S + 0.10)

    non_skin_color_ratio = float(high_color_mask.sum() / face_area)
    feats["non_skin_color_ratio"] = non_skin_color_ratio

    extreme_sat_mask = mask_face & (S_norm > 0.80) & (V_norm > 0.35)
    extreme_saturation_ratio = float(extreme_sat_mask.sum() / face_area)
    feats["extreme_saturation_ratio"] = extreme_saturation_ratio

    high_color_uint8 = (high_color_mask.astype(np.uint8)) * 255
    num_labels, labels_cc, stats_cc, _ = cv2.connectedComponentsWithStats(
        high_color_uint8, connectivity=8
    )
    largest_area_ratio = 0.0
    if num_labels > 1:
        areas = stats_cc[1:, cv2.CC_STAT_AREA]
        largest_area_ratio = float(areas.max() / face_area)
    feats["masklike_pattern_score"] = largest_area_ratio
    feats["non_skin_large_zones"] = largest_area_ratio

    fx_mask = high_color_mask & (~mask_eyes) & (~mask_lips) & (~mask_cheeks)
    fx_ratio = float(fx_mask.sum() / face_area)
    feats["fx_ratio"] = fx_ratio

    # ======================================================
    # NUEVOS FEATURES GLAM NOCHE
    # ======================================================

    # 1) Proporción de píxeles muy oscuros en ojos
    if stats_eyes is not None and mask_eyes.sum() > 0:
        eyes_V = V_norm[mask_eyes]
        feats["eyes_dark_ratio"] = float(np.mean(eyes_V < 0.30))
    else:
        feats["eyes_dark_ratio"] = np.nan

    # 2) Proporción de píxeles oscuros en labios
    if stats_lips is not None and mask_lips.sum() > 0:
        lips_V = V_norm[mask_lips]
        feats["lips_dark_ratio"] = float(np.mean(lips_V < 0.40))
    else:
        feats["lips_dark_ratio"] = np.nan

    # 3) Concentración de color “no piel” en la zona de ojos
    face_high = high_color_mask[mask_face]
    eyes_high = high_color_mask[mask_eyes] if mask_eyes.sum() > 0 else None

    if eyes_high is not None and face_high.size > 0 and np.any(face_high):
        eye_high_ratio = float(np.mean(eyes_high))
        face_high_ratio = float(np.mean(face_high))
        feats["eyes_color_concentration"] = eye_high_ratio / (face_high_ratio + 1e-6)
    else:
        feats["eyes_color_concentration"] = np.nan

    return feats


# =====================================================
# 4. CARGA DE ARTEFACTOS (MODELO + SCALER + PORTFOLIO)
# =====================================================

@st.cache_resource(show_spinner=True)
def load_artifacts():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    models_dir = os.path.join(base_dir, "models")
    data_dir = os.path.join(base_dir, "data")

    rf_path = os.path.join(models_dir, "rf_model.pkl")
    scaler_path = os.path.join(models_dir, "scaler.pkl")
    le_path = os.path.join(models_dir, "label_encoder.pkl")
    feature_cols_path = os.path.join(data_dir, "feature_cols.json")
    portfolio_path = os.path.join(data_dir, "portfolio_features_final.csv")

    import joblib

    rf_model = joblib.load(rf_path)
    scaler = joblib.load(scaler_path)
    label_encoder = joblib.load(le_path)

    with open(feature_cols_path, "r", encoding="utf-8") as f:
        feature_cols = json.load(f)

    portfolio_df = pd.read_csv(portfolio_path)

    # Por si falta artist
    if "artist" not in portfolio_df.columns:
        portfolio_df["artist"] = "desconocido"

    # Calcular medias de cada feature (para imputar faltantes en nuevas imágenes)
    feature_means = portfolio_df[feature_cols].mean(axis=0)

    # Matriz escalada del portfolio (para recomendación)
    feature_data = portfolio_df[feature_cols].astype(float).fillna(feature_means)
    X_portfolio_scaled = scaler.transform(feature_data.values)

    return rf_model, scaler, label_encoder, feature_cols, feature_means, portfolio_df, X_portfolio_scaled


# =====================================================
# 5. PREPARAR FEATURES Y RECOMENDACIÓN
# =====================================================

def prepare_features_for_model(feats: dict,
                               feature_cols: list,
                               feature_means: pd.Series,
                               scaler) -> np.ndarray:
    """
    Ordena las features, rellena NaN con medias y aplica scaler.
    """
    row = []
    for col in feature_cols:
        v = feats.get(col, np.nan)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            v = feature_means[col]
        row.append(v)

    X = pd.DataFrame([row], columns=feature_cols)
    X_scaled = scaler.transform(X)
    return X_scaled


def recommend_similar_looks(X_query_scaled: np.ndarray,
                            X_portfolio_scaled: np.ndarray,
                            portfolio_df: pd.DataFrame,
                            top_k: int = 3) -> pd.DataFrame:
    """
    Similitud coseno entre la imagen subida y todo el portfolio.
    """
    sims = cosine_similarity(X_query_scaled, X_portfolio_scaled)[0]
    best_idx = np.argsort(sims)[::-1][:top_k]
    df_rec = portfolio_df.iloc[best_idx].copy()
    df_rec["similarity"] = sims[best_idx]
    return df_rec


# =====================================================
# 6. INTERFAZ STREAMLIT
# =====================================================

def main():
    st.set_page_config(
        page_title="Clasificador de Maquillaje + Recomendador",
        layout="centered"
    )

    st.title("Clasificador de maquillaje y recomendador de maquilladores")
    st.write(
        "Sube la foto de un maquillaje. La herramienta:\n"
        "1) Clasifica el look en **glam día, glam noche o temático**.\n"
        "2) Busca en el portfolio los maquillajes más similares y muestra a los artistas."
    )

    rf_model, scaler, label_encoder, feature_cols, feature_means, portfolio_df, X_portfolio_scaled = load_artifacts()

    uploaded_file = st.file_uploader("Sube una imagen (jpg o png)", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Imagen subida", use_container_width=True)

        # Guardar en temporal
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(uploaded_file.getbuffer())
            tmp_path = tmp.name

        st.write("Extrayendo características de maquillaje...")
        feats = extract_makeup_features(tmp_path)

        if feats is None:
            st.error("No se detectó rostro o no se pudieron extraer características.")
            return

        # Vector + predicción
        X_scaled = prepare_features_for_model(feats, feature_cols, feature_means, scaler)
        pred_idx = rf_model.predict(X_scaled)[0]
        proba = rf_model.predict_proba(X_scaled)[0]
        pred_label = label_encoder.inverse_transform([pred_idx])[0]

        st.subheader("Clasificación de estilo")
        st.write(f"**Estilo predicho:** {pred_label}")
        st.write("Probabilidades:")

        prob_df = pd.DataFrame(
            [proba],
            columns=label_encoder.inverse_transform(np.arange(len(proba)))
        ).T.rename(columns={0: "probabilidad"})
        prob_df["probabilidad"] = prob_df["probabilidad"].round(3)
        st.table(prob_df)

        st.subheader("Recomendación de maquillajes similares")
        top_k = st.slider("Número de recomendaciones", 1, 5, 3)

        rec_df = recommend_similar_looks(X_scaled, X_portfolio_scaled, portfolio_df, top_k=top_k)

        base_dir = os.path.dirname(os.path.abspath(__file__))
        images_base = os.path.join(base_dir, "images")

        for _, row in rec_df.iterrows():
            rel_path = row["relative_path"]
            artist = row.get("artist", "desconocido")
            similarity = row["similarity"]
            cat = row.get("category", "")

            img_path = os.path.join(images_base, rel_path)

            if os.path.exists(img_path):
                st.image(
                    img_path,
                    caption=f"{cat} | {artist} | similitud: {similarity:.2f}",
                    use_container_width=True,
                )
            else:
                st.write(f"(No se encontró la imagen: {img_path})")


if __name__ == "__main__":
    main()
