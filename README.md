# Kidney Disease Classification

A deep learning pipeline for classifying kidney disease from CT scan images, built with TensorFlow/Keras, tracked with MLflow & DVC, and deployed via Docker + AWS CI/CD.

## Features

- CNN-based image classification for kidney CT scans
- Experiment tracking and model versioning with MLflow (via DagsHub)
- Reproducible pipelines with DVC
- Flask web app for inference (`app.py`)
- Dockerized deployment with GitHub Actions CI/CD to AWS

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/Dippy2003/Kidney_Disease_Classification
```

### 2. Create a conda environment

```bash
conda create -n cnncls python=3.8 -y
conda activate cnncls
```

### 3. Install the requirements

```bash
pip install -r requirements.txt
```

### 4. Run the app

```bash
python app.py
```

Then open the local host and port shown in the terminal.
