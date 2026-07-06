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

## MLflow & DVC

- **MLflow** — production-grade experiment tracking, logging and tagging models. [Docs](https://mlflow.org/docs/latest/index.html)
- **DVC** — lightweight pipeline orchestration and experiment tracking, good for POCs.

### MLflow (via DagsHub)

Set the following as environment variables (get credentials from your DagsHub account — do not hardcode them):

```bash
export MLFLOW_TRACKING_URI=https://dagshub.com/Dippy2003/Kidney_Disease_Classification.mlflow
export MLFLOW_TRACKING_USERNAME=<your-dagshub-username>
export MLFLOW_TRACKING_PASSWORD=<your-dagshub-token>
```

```bash
mlflow ui
```

### DVC commands

```bash
dvc init
dvc repro
dvc dag
```

## AWS CI/CD Deployment (GitHub Actions)

### 1. Login to AWS console

### 2. Create an IAM user for deployment

With specific access:
- **EC2** — virtual machine access
- **ECR** — Elastic Container Registry, to store the Docker image

Deployment flow:
1. Build a Docker image of the source code
2. Push the Docker image to ECR
3. Launch an EC2 instance
4. Pull the image from ECR onto EC2
5. Run the Docker container on EC2

Required policies:
- `AmazonEC2ContainerRegistryFullAccess`
- `AmazonEC2FullAccess`

### 3. Create an ECR repo to store the Docker image

Save the repository URI for later use in GitHub secrets.

### 4. Create an EC2 machine (Ubuntu)

### 5. Install Docker on the EC2 machine

```bash
sudo apt-get update -y
sudo apt-get upgrade

curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker ubuntu
newgrp docker
```

### 6. Configure EC2 as a self-hosted GitHub Actions runner

Settings → Actions → Runners → New self-hosted runner → choose OS → run the setup commands.

### 7. Set up GitHub secrets

```
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
AWS_REGION=us-east-1
AWS_ECR_LOGIN_URI=<account-id>.dkr.ecr.<region>.amazonaws.com
ECR_REPOSITORY_NAME=<your-repo-name>
```
