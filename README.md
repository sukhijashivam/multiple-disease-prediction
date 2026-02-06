# 🏥 Multiple Disease Prediction System

AI-powered Streamlit app for multiple disease prediction using Machine Learning and Deep Learning

## 📋 About

This project is a Streamlit-based web application that predicts multiple diseases using trained machine learning and deep learning models. The app can predict:

- 🩷 **Breast Cancer** (using CNN - EfficientNet)
- 🧠 **Brain Tumor** (using CNN)
- 🍬 **Diabetes** (using Random Forest)
- ❤️ **Heart Disease** (using Logistic Regression)
- 🗣️ **Parkinson's Disease** (using SVM)

## 🛠️ Tech Stack

- **Python 3.12**
- **Streamlit** - Web framework
- **TensorFlow/Keras** - Deep learning models
- **Scikit-learn** - Machine learning models
- **Google Drive** - Model storage (for large .h5 files)

## 📁 Project Structure

```
multiple-disease-prediction/
│
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Docker configuration
│
├── saved_models/              # Pre-trained models
│   ├── breast_cancer_model_last.h5
│   ├── Brain_Tumor_Classification_model_fixed.h5
│   ├── diabetes_model.sav
│   ├── heart_disease_model.sav
│   └── parkinsons_model.sav
│
└── temp_backup/               # Backup files
```

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Steps

1. **Clone the repository**
```bash
git clone https://github.com/sukhijashivam/multiple-disease-prediction.git
cd multiple-disease-prediction
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the application**
```bash
streamlit run app.py
```

4. **Access the app**
- Open your browser and go to `http://localhost:8501`
- Models will auto-download from Google Drive on first run

## 💻 How to Use

### For Image-Based Predictions (Breast Cancer, Brain Tumor)

1. Select the disease from the sidebar
2. Upload the medical image (JPG, JPEG, PNG)
   - Mammogram for Breast Cancer
   - MRI scan for Brain Tumor
3. Click "Predict"
4. View results with confidence scores

### For Data-Based Predictions (Diabetes, Heart Disease, Parkinson's)

1. Select the disease from the sidebar
2. Enter the required medical parameters
3. Click "Predict"
4. View results and recommendations

## 🧠 Models

| Disease | Model Type | Input Type |
|---------|-----------|------------|
| Breast Cancer | CNN (EfficientNet) | Mammogram Image |
| Brain Tumor | Custom CNN | MRI Image |
| Diabetes | Random Forest | Clinical Data (8 features) |
| Heart Disease | Logistic Regression | Clinical Data (13 features) |
| Parkinson's | SVM | Voice Data (22 features) |

## 📦 Model Storage

Large `.h5` model files (>100 MB) are stored on Google Drive and automatically downloaded when you first run the app using the `gdown` library.

## ⚠️ Medical Disclaimer

**IMPORTANT**: This application is for **educational purposes only**. It is NOT a substitute for professional medical advice, diagnosis, or treatment. Always consult qualified healthcare providers for medical decisions.

## 👨‍💻 Author

**Shivam Sukhija**

- GitHub: [@sukhijashivam](https://github.com/sukhijashivam)
- Email: shivamsukhija002@gmail.com

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- Dataset sources from Kaggle
- Streamlit for the web framework
- TensorFlow and Scikit-learn communities

---

⭐ If you find this project helpful, please star the repository!
