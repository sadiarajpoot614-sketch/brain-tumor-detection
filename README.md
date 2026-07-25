
# Brain Tumor Detection API

## Overview
This project is a FastAPI based web service that uses a ResNet18 deep learning model to classify brain MRI images. The API can detect and predict 4 types: glioma, meningioma, pituitary tumor, and no tumor, along with a confidence score for each prediction.

## Key Features
- Upload MRI images and get instant predictions
- Built with PyTorch and FastAPI for high performance
- Clean and interactive API documentation with Swagger UI
- Model is automatically downloaded from Google Drive on first run to keep the repository lightweight

## Technology Stack
Backend: FastAPI
Model: ResNet18 with PyTorch
Server: Uvicorn
Image Processing: PIL and Torchvision

## How to Run Locally
First, install all dependencies using pip install -r http://requirements.txt.
After that, start the server by running uvicorn app:app --reload.
Once the server is running, open http://127.0.0.1:8000/docs in your browser. Here you can test the /predict/upload endpoint by uploading an MRI image.

## API Usage
Main Endpoint: POST /predict/upload
Input: MRI image file
Output: JSON response with predicted tumor type and confidence score

## Notes
The model file tumor_detection_resnet18.pth is not included in the repo. It will be downloaded automatically from Google Drive when the app starts for the first time.
