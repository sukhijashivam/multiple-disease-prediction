# 🏥 AI Health Diagnosis Suite  
### 🤖 Multi-Disease Prediction using Deep Learning & Machine Learning  

> 🚀 *An advanced AI-powered platform capable of detecting multiple diseases — including Breast Cancer, Brain Tumor, Diabetes, Heart Disease, and Parkinson’s — all in one place!*  

---

## 🌟 Overview  

This project brings together **Machine Learning (ML)** and **Deep Learning (DL)** models in a single **Streamlit web app**, enabling users to predict critical diseases using medical images or patient data.  

Each model has been trained and fine-tuned on real medical datasets to ensure **accuracy, performance, and usability**.  

---

## 🧠 Diseases Predicted  

| Disease | Model | Input Type | Description |
|----------|--------|-------------|--------------|
| 🩷 **Breast Cancer** | CNN (EfficientNet) | Mammogram Image | Classifies as *Benign*, *Malignant*, or *Normal*. |
| 🧠 **Brain Tumor** | CNN | MRI Image | Detects *Glioma*, *Meningioma*, *Pituitary*, or *No Tumor*. |
| 🍬 **Diabetes** | Random Forest | Numeric Data | Predicts likelihood of diabetes. |
| ❤️ **Heart Disease** | Logistic Regression | Numeric Data | Predicts heart disease risk. |
| 🗣️ **Parkinson’s** | SVM | Voice Data | Detects Parkinson’s symptoms. |

---

## 🧩 Tech Stack  

- **Frontend:** Streamlit 🎨  
- **Backend:** TensorFlow & Scikit-learn 🧠  
- **Language:** Python 3.12  
- **Deployment:** Streamlit Cloud ☁️  
- **Storage:** Google Drive for `.h5` models (>100 MB)  

---

## ⚙️ How It Works  

1. **Upload** an image or enter patient data.  
2. The app automatically **downloads models** from Google Drive.  
3. Data is **preprocessed and analyzed** using trained AI models.  
4. The model predicts the **disease type & confidence score**.  
5. Result is displayed beautifully with detailed health suggestions.  

---

## 🧬 Model Management via Google Drive  

Large `.h5` models are stored on **Google Drive** (since GitHub limits files >100MB).  
Models are automatically downloaded using `gdown`.

```python
drive_files = {
    "breast_cancer_model_last.h5": "12Y5ju8ZyAifCAHiQiqQ635eSnifxdTJt",
    "Brain_Tumor_Classification_model_fixed.h5": "1TVhY0DEDbehA3A-t8GzbiSDY5G0LAlXc"
}
