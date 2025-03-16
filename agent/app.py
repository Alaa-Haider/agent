
from fastapi import FastAPI, File, UploadFile
import torch
import torchxrayvision as xrv
import torchvision.transforms as transforms
from PIL import Image
import io

app = FastAPI()

# Load Pre-trained Model
model = xrv.models.DenseNet(weights="densenet121-res224-all")

# Preprocessing Function
def preprocess_image(image_bytes):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = transform(image).unsqueeze(0)
    return image

# X-ray Analysis Function
def analyze_xray(image_bytes):
    image_tensor = preprocess_image(image_bytes)
    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
    labels = xrv.datasets.default_pathologies
    predictions = {labels[i]: float(output[0][i]) for i in range(len(labels))}
    return predictions

# API Endpoint for X-ray Analysis
@app.post("/analyze/")
async def analyze(file: UploadFile = File(...)):
    image_bytes = await file.read()
    predictions = analyze_xray(image_bytes)
    return {"predictions": predictions}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

