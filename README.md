# 🏥 AI Health Diagnosis Suite  

### 🤖 Multi-Disease Prediction using Deep Learning & Machine Learning  
🚀 An advanced **AI-powered healthcare platform** that predicts multiple diseases — **Breast Cancer, Brain Tumor, Diabetes, Heart Disease, and Parkinson’s** — all within a single interactive web app!

---

## 🌟 Overview  
This project unifies **Machine Learning (ML)** and **Deep Learning (DL)** techniques into one **Streamlit-based web application**, enabling users to predict critical diseases using either **medical images** or **numerical patient data**.  
Each model has been **trained, validated, and fine-tuned** on trusted, real-world medical datasets to deliver **high accuracy**, **robustness**, and **ease of use**.  

---

## 🧠 Diseases Predicted  
| 🩺 Disease | 🧠 Model | 📊 Input Type | 💬 Description |
|-------------|----------|---------------|----------------|
| 🩷 **Breast Cancer** | CNN (EfficientNet) | Mammogram Image | Classifies as *Benign*, *Malignant*, or *Normal*. |
| 🧠 **Brain Tumor** | CNN | MRI Image | Detects *Glioma*, *Meningioma*, *Pituitary*, or *No Tumor*. |
| 🍬 **Diabetes** | Random Forest | Numeric Data | Predicts the likelihood of diabetes. |
| ❤️ **Heart Disease** | Logistic Regression | Numeric Data | Predicts the risk of heart disease. |
| 🗣️ **Parkinson’s** | SVM | Voice Data | Detects early signs of Parkinson’s. |

---

## 🧩 Tech Stack  
- **Frontend:** Streamlit 🎨  
- **Backend:** TensorFlow, Scikit-learn 🧠  
- **Language:** Python 3.12 🐍  
- **Deployment:** Streamlit Cloud ☁️  
- **Model Storage:** Google Drive (for `.h5` models >100 MB)  

---

## ⚙️ How It Works  
1. 🧾 **Upload** an image or **enter** patient data.  
2. ☁️ App automatically **downloads pre-trained models** from Google Drive.  
3. ⚙️ Data is **preprocessed & analyzed** using advanced ML/DL models.  
4. 🔮 Model predicts the disease type & confidence score.  
5. 📊 Results are displayed with **interactive visuals & health recommendations.**

---

## 🧬 Model Management via Google Drive  
Large `.h5` models (>100 MB) are hosted on **Google Drive** because GitHub limits large files.  
The app uses the `gdown` library to automatically download models into the app’s runtime.  

```python
drive_files = {
    "breast_cancer_model_last.h5": "12Y5ju8ZyAifCAHiQiqQ635eSnifxdTJt",
    "Brain_Tumor_Classification_model_fixed.h5": "1TVhY0DEDbehA3A-t8GzbiSDY5G0LAlXc",
    "diabetes_model.sav": "1yRAWrjY3B2K6s5X87ZdWduHChvgOu__V",
    "heart_disease_model.sav": "12-9QP7AvEBHoEbbEEjFAG2JUkOYZwKJw",
    "parkinsons_model.sav": "1sW9oZsmBVcpfPc0HVWozGpzgDasI1JWe"
}
```


🧠 Installation & Running the App

Follow these steps to run the project on your local system 🖥️ or Google Colab ☁️

🏡 Running Locally
# 1️⃣ Clone the Repository
git clone https://github.com/yourusername/multiple-disease-prediction.git
cd multiple-disease-prediction

# 2️⃣ Create a Virtual Environment
python -m venv venv

# 3️⃣ Activate the Virtual Environment
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

# 4️⃣ Install Dependencies
pip install -r requirements.txt

# 5️⃣ Run the Streamlit App
streamlit run app.py


☁️ Running on Google Colab

If you prefer not to install locally, you can easily run it on Colab

# 1️⃣ Install dependencies
!pip install streamlit gdown pyngrok

# 2️⃣ Clone the repository
!git clone https://github.com/yourusername/multiple-disease-prediction.git
%cd multiple-disease-prediction

# 3️⃣ Run the app
!streamlit run app.py & npx localtunnel --port 8501

💡 Future Enhancements

Integration with real-time image capture (via webcam or DICOM upload).

Add Explainability (XAI) using Grad-CAM for image-based predictions.

Deploy via Docker + Streamlit Cloud for scalable hosting.

Add disease probability charts & lifestyle recommendations.

🧑‍💻 Author
Shivam Sukhija
💼 AI & Data Science Enthusiast
📧 shivamsukhija002@gmail.com
🌐 GitHub Profile- https://github.com/sukhijashivam

⭐ Support & Contributions

If you like this project, please ⭐ the repository and share it!
Pull requests and suggestions for improvements are always welcome 🙌



}
