from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List
import torch
from torchvision import transforms, models
from PIL import Image
import io
import base64
import os
import gdown # <-- naya add

# Initialize FastAPI app
app = FastAPI()

# --- Google Drive se Model Download ---
MODEL_PATH = 'tumor_detection_resnet18.pth'
GDRIVE_URL = "https://drive.google.com/uc?id=1qi_IuS5RPjpx5gl7qknwq3CQ2i2b2SOu" # aapka link

if not os.path.exists(MODEL_PATH):
    print("Downloading model from Google Drive...")
    gdown.download(GDRIVE_URL, MODEL_PATH, quiet=False)
    print("Download complete!")


# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Model Loading ---
num_classes = 4 
class_names = ['glioma', 'meningioma', 'no_tumor', 'pituitary'] 

# Load the pre-trained ResNet18 model
model = models.resnet18(pretrained=False)

# Replace the final fully connected layer (fc)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, num_classes)

# Load the saved state dictionary
try:
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    raise HTTPException(status_code=500, detail="Model could not be loaded.")

# --- Image Transformation ---
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- Prediction Function ---
async def predict_image(image_bytes: bytes):
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_tensor = preprocess(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            _, predicted_idx = torch.max(outputs, 1)

        predicted_class = class_names[predicted_idx.item()]
        confidence = probabilities[0][predicted_idx.item()].item()

        return {"prediction": predicted_class, "confidence": round(confidence, 4)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image processing or prediction failed: {e}")

# --- API Endpoints ---
@app.get("/", tags=["Root"])
async def read_root():
    return {"message": "Welcome to the Brain Tumor Detection API!"}

@app.post("/predict/upload", tags=["Prediction"])
async def predict_with_upload(file: UploadFile = File(...)):
    image_bytes = await file.read()
    return await predict_image(image_bytes)

@app.post("/predict/base64", tags=["Prediction"])
async def predict_with_base64(image_data: dict):
    try:
        base64_string = image_data.get("image_base64")
        if not base64_string:
            raise HTTPException(status_code=400, detail="'image_base64' field is missing.")
        image_bytes = base64.b64decode(base64_string)
        return await predict_image(image_bytes)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Base64 decoding or prediction failed: {e}")
