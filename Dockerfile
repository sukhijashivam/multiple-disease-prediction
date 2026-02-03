FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# 🔥 Install exact compatible stack (no resolver fights)
RUN pip install \
    numpy==1.23.5 \
    protobuf==3.19.6 \
    h5py==3.8.0 \
    tensorflow==2.10.0 \
    keras==2.10.0 \
    streamlit==1.25.0 \
    pandas==1.5.3 \
    scikit-learn==1.1.3 \
    joblib==1.2.0 \
    matplotlib==3.6.3 \
    gdown==4.7.1 \
    opencv-python-headless==4.7.0.72 \
    Pillow==9.5.0

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
