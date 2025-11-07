import streamlit as st
import numpy as np
import cv2
import joblib
from tensorflow import keras
from PIL import Image
import os

# ---------------------- Load Models ----------------------
import gdown
import os

# Define model directory
# ✅ Works both locally & on Streamlit Cloud
import tempfile

BASE_DIR = tempfile.gettempdir()
MODEL_DIR = os.path.join(BASE_DIR, "saved_models")
os.makedirs(MODEL_DIR, exist_ok=True)

#https://drive.google.com/file/d/1ZfgP53sorzuiSD3n0KbL3yAuG7uO1lyH/view?usp=sharing
# --- Download large models from Google Drive if missing ---
drive_files = {
    "breast_cancer_model_last.h5": "12Y5ju8ZyAifCAHiQiqQ635eSnifxdTJt",  # ✅ Breast model
    "Brain_Tumor_Classification_model.h5": "1ZfgP53sorzuiSD3n0KbL3yAuG7uO1lyH"  # ✅ Brain model
}
#1TVhY0DEDbehA3A-t8GzbiSDY5G0LAlXc

for name, file_id in drive_files.items():
    path = os.path.join(MODEL_DIR, name)
    if not os.path.exists(path):
        print(f"⬇️ Downloading {name} from Google Drive...")
        gdown.download(f"https://drive.google.com/uc?id={file_id}", path, quiet=False)
    else:
        print(f"✅ {name} already exists, skipping download.")

# --- Load models (after ensuring they're downloaded) ---
breast_model = keras.models.load_model(os.path.join(MODEL_DIR, "breast_cancer_model_last.h5"))
brain_model = keras.models.load_model(os.path.join(MODEL_DIR, "Brain_Tumor_Classification_model.h5"))
diabetes_model = joblib.load(os.path.join(MODEL_DIR, "diabetes_model.sav"))
heart_model = joblib.load(os.path.join(MODEL_DIR, "heart_disease_model.sav"))
parkinsons_model = joblib.load(os.path.join(MODEL_DIR, "parkinsons_model.sav"))


# ---------------------- App Theme ----------------------
st.set_page_config(page_title="AI Health Diagnosis Suite", page_icon="🏥", layout="wide")

# Custom CSS
st.markdown("""
    <style>
    .stApp {
        background: radial-gradient(circle at top left, #0f2027, #203a43, #2c5364);
        color: #f0f0f0;
    }

    h1, h2, h3 {
        text-align: center;
        color: #00e6e6;
        text-shadow: 0 0 20px rgba(0, 230, 230, 0.6);
        font-weight: 700;
    }

    h2 span {
        color: #ff80bf;
        text-shadow: 0 0 15px rgba(255, 128, 191, 0.6);
    }

    .subtitle {
        text-align: center;
        color: #d1d1d1;
        font-size: 1.1rem;
        margin-top: -10px;
        margin-bottom: 30px;
    }

    .prediction-box {
        border-radius: 15px;
        padding: 25px;
        margin-top: 30px;
        text-align: center;
        font-size: 1.5em;
        font-weight: 700;
        letter-spacing: 0.5px;
        width: 80%;
        margin-left: auto;
        margin-right: auto;
        box-shadow: 0 0 25px rgba(0,0,0,0.3);
        transition: all 0.4s ease-in-out;
    }

    .success {
        background: linear-gradient(135deg, rgba(0,255,170,0.2), rgba(0,255,200,0.05));
        border: 2px solid #00ffaa;
        color: #00ffcc;
        text-shadow: 0 0 10px rgba(0,255,200,0.6);
        box-shadow: 0 0 30px rgba(0,255,170,0.2);
    }

    .error {
        background: linear-gradient(135deg, rgba(255,77,77,0.2), rgba(255,0,100,0.05));
        border: 2px solid #ff4d4d;
        color: #ff6666;
        text-shadow: 0 0 10px rgba(255,77,77,0.6);
        box-shadow: 0 0 30px rgba(255,0,80,0.25);
    }

    .stButton>button {
        background: linear-gradient(90deg, #00c6ff, #0072ff);
        color: white;
        border-radius: 10px;
        padding: 0.6rem 1.2rem;
        border: none;
        font-size: 1rem;
        transition: 0.3s;
    }

    .stButton>button:hover {
        background: linear-gradient(90deg, #ff80bf, #ff0066);
        transform: scale(1.05);
    }

    div[data-testid="stSidebar"] {
        background: linear-gradient(to bottom, #14213d, #000000);
    }

    input, select, textarea {
        border-radius: 10px !important;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------- Header ----------------------
st.markdown("<h1>🏥 AI Health Diagnosis Suite</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Your all-in-one intelligent platform for predicting diseases using AI and medical data.</p>", unsafe_allow_html=True)

# ---------------------- Helper Function ----------------------
def safe_image_read(uploaded_file, target_size):
    uploaded_file.seek(0)
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("Invalid image file")

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    elif img.shape[2] > 3:
        img = img[:, :, :3]

    img = cv2.resize(img, target_size)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img_rgb

# ---------------------- BEAUTIFIED IMAGE PREDICTIONS ----------------------
def predict_breast_cancer(uploaded_file):
    class_labels = ["Benign", "Malignant", "Normal"]
    img = Image.open(uploaded_file).convert("RGB").resize((224, 224))
    img_array = np.expand_dims(np.asarray(img, dtype=np.float32), axis=0)
    img_array = keras.applications.efficientnet.preprocess_input(img_array)

    preds = breast_model.predict(img_array)
    idx = np.argmax(preds[0])
    conf = float(np.max(preds[0])) * 100
    label = class_labels[idx]

    st.image(img, caption="", use_container_width=False, width=300)

    color = "#00ffaa" if label.lower() in ["benign", "normal"] else "#ff4d4d"
    emoji = "💖" if label.lower() in ["benign", "normal"] else "⚠️"

    st.markdown(f"""
        <div style="
            text-align:center;
            margin-top:25px;
            background:rgba(255,255,255,0.08);
            border-radius:15px;
            padding:25px;
            border:2px solid {color};
            color:{color};
            font-size:1.8em;
            font-weight:700;
            box-shadow:0 0 25px {color}55;
            animation: fadeIn 1s ease-in-out;
        ">
            {emoji} <b>Diagnosis:</b> {label}<br>
            <span style="font-size:0.8em;color:#ccc;">Confidence: {conf:.2f}%</span><br><br>
            <span style="font-size:0.85em;color:#ddd;">
                The uploaded mammogram indicates <b>{label}</b> features with {conf:.2f}% confidence.<br>
                ⚕️ Please consult a certified radiologist for medical verification.
            </span>
        </div>

        <style>
        @keyframes fadeIn {{
            from {{opacity:0; transform:translateY(10px);}}
            to {{opacity:1; transform:translateY(0);}}
        }}
        </style>
    """, unsafe_allow_html=True)


def predict_brain_tumor(uploaded_file):
    class_labels = ['Pituitary', 'Meningioma', 'No Tumor', 'Glioma']
    img_rgb = safe_image_read(uploaded_file, (128, 128))
    img_array = np.array(img_rgb).astype("float32") / 255.0
    x = np.expand_dims(img_array, axis=0)
    preds = brain_model.predict(x)
    idx = np.argmax(preds[0])
    conf = float(np.max(preds[0])) * 100
    result = class_labels[idx]

    st.image(img_rgb, caption="", use_container_width=False, width=300)

    is_tumor = result.lower() not in ["no tumor"]
    color = "#ff4d4d" if is_tumor else "#00ffaa"
    emoji = "⚠️" if is_tumor else "💚"

    st.markdown(f"""
        <div style="
            text-align:center;
            margin-top:25px;
            background:rgba(255,255,255,0.08);
            border-radius:15px;
            padding:25px;
            border:2px solid {color};
            color:{color};
            font-size:1.8em;
            font-weight:700;
            box-shadow:0 0 25px {color}55;
            animation: fadeIn 1s ease-in-out;
        ">
            {emoji} <b>Diagnosis:</b> {result}<br>
            <span style="font-size:0.8em;color:#ccc;">Confidence: {conf:.2f}%</span><br><br>
            <span style="font-size:0.85em;color:#ddd;">
                The MRI scan suggests <b>{result}</b> features with {conf:.2f}% confidence.<br>
                ⚕️ Please consult a neurologist for detailed evaluation.
            </span>
        </div>
    """, unsafe_allow_html=True)


# ---------------------- Remaining Prediction Functions ----------------------
def predict_diabetes(inputs):
    inputs = np.asarray(inputs, dtype=float).reshape(1, -1)
    pred = diabetes_model.predict(inputs)[0]
    if pred == 1:
        st.markdown('<div class="prediction-box error">⚠️ You are likely to have Diabetes.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="prediction-box success">✅ You are not likely to have Diabetes.</div>', unsafe_allow_html=True)


def predict_heart(inputs):
    inputs = np.asarray(inputs, dtype=float).reshape(1, -1)
    pred = heart_model.predict(inputs)[0]
    if pred == 1:
        st.markdown('<div class="prediction-box error">❤️ Heart Disease Detected!</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="prediction-box success">💖 Your Heart is Healthy!</div>', unsafe_allow_html=True)


def predict_parkinsons(inputs):
    inputs = np.asarray(inputs, dtype=float).reshape(1, -1)
    pred = parkinsons_model.predict(inputs)[0]
    if pred == 1:
        st.markdown('<div class="prediction-box error">🧠 Parkinson\'s symptoms detected.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="prediction-box success">✨ No Parkinson\'s symptoms detected.</div>', unsafe_allow_html=True)


# ---------------------- Sidebar ----------------------
menu = st.sidebar.radio(
    "Select Disease",
    ("🩷 Breast Cancer", "🧠 Brain Tumor", "🍬 Diabetes", "❤️ Heart Disease", "🗣️ Parkinson’s")
)

# ---------------------- Layout ----------------------
if "Breast" in menu:
    st.markdown("<h2><span>🩷 Breast Cancer Detection</span></h2>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload a Mammogram Image", type=["jpg", "jpeg", "png", "webp"])
    if uploaded_file:
        predict_breast_cancer(uploaded_file)

elif "Brain" in menu:
    st.markdown("<h2><span>🧠 Brain Tumor Detection</span></h2>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload a Brain MRI Image", type=["jpg", "jpeg", "png", "webp"])
    if uploaded_file:
        predict_brain_tumor(uploaded_file)

elif "Diabetes" in menu:
    st.markdown("<h2><span>🍬 Diabetes Prediction</span></h2>", unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    with col1: pregnancies = st.number_input("Pregnancies", min_value=0)
    with col2: glucose = st.number_input("Glucose Level")
    with col3: bp = st.number_input("Blood Pressure")
    with col4: skin = st.number_input("Skin Thickness")
    col5, col6, col7, col8 = st.columns(4)
    with col5: insulin = st.number_input("Insulin Level")
    with col6: bmi = st.number_input("BMI")
    with col7: dpf = st.number_input("Diabetes Pedigree Function")
    with col8: age = st.number_input("Age")
    if st.button("🔍 Predict Diabetes"):
        predict_diabetes([pregnancies, glucose, bp, skin, insulin, bmi, dpf, age])

elif "Heart" in menu:
    st.markdown("<h2><span>❤️ Heart Disease Prediction</span></h2>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.number_input("Age")
        cp = st.number_input("Chest Pain Type (0–3)")
        chol = st.number_input("Cholesterol")
        fbs = st.number_input("Fasting Blood Sugar > 120 mg/dl (1/0)")
    with c2:
        sex = st.selectbox("Sex", ["Male", "Female"])
        trestbps = st.number_input("Resting BP")
        restecg = st.number_input("Rest ECG Results (0–2)")
        thalach = st.number_input("Max Heart Rate")
    with c3:
        exang = st.number_input("Exercise Induced Angina (1/0)")
        oldpeak = st.number_input("Oldpeak")
        slope = st.number_input("Slope (0–2)")
        ca = st.number_input("No. of major vessels (0–3)")
        thal = st.number_input("Thal (1–3)")

    sex_val = 1 if sex == "Male" else 0
    if st.button("🔍 Predict Heart Disease"):
        predict_heart([age, sex_val, cp, trestbps, chol, fbs, restecg,
                       thalach, exang, oldpeak, slope, ca, thal])

elif "Parkinson" in menu:
    st.markdown("<h2><span>🗣️ Parkinson’s Disease Prediction</span></h2>", unsafe_allow_html=True)
    cols = st.columns(3)
    inputs = []
    for i in range(22):
        col = cols[i % 3]
        with col:
            inputs.append(st.number_input(f"Feature {i+1}"))
    if st.button("🔍 Predict Parkinson’s"):
        predict_parkinsons(inputs)

