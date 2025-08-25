from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
import torch.nn.functional as F
import io, os

from app.models.sign_model import SignLanguageModel

app = FastAPI(title="Sign Language API")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = os.getenv("MODEL_PATH", "model/mymodel_statedict.pth")

model = SignLanguageModel(input_size=784, output_size=24)
state = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(state)
model.to(DEVICE)
model.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

@app.post("/detection")
async def detection(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("L")
    if image.size != (28, 28):
        image = image.resize((28, 28))

    tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1)
        conf, pred = torch.max(probs, dim=1)

    return JSONResponse(content={
        "predicted_label": int(pred.item()),
        "confidence": round(float(conf.item()), 4)
    })
