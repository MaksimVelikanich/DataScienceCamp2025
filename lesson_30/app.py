from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
import torch
import io
from torchvision import transforms

from model import load_model 
app = FastAPI()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model("mymodel_statedict.pth", device)

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

    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(image)
        predicted_label = pred.argmax(1).item()
        probabilities = torch.nn.functional.softmax(pred, dim=1)
        top_prob = probabilities.max().item()

    return JSONResponse(content={
        "predicted_label": predicted_label,
        "confidence": f"{top_prob:.4f}"
    })
