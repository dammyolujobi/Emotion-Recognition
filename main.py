from fastapi import APIRouter,File,UploadFile,FastAPI
from transformers import  AutoModelForImageClassification,AutoImageProcessor
import torch
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import uvicorn
import io

extractor = AutoImageProcessor.from_pretrained("dima806/facial_emotions_image_detection",use_fast = True)
model = AutoModelForImageClassification.from_pretrained("dima806/facial_emotions_image_detection")

app = FastAPI(
    title = "Emotion Recognizer"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
async def upload_image(file:UploadFile = File(...)):
    
    img_bytes = await file.read()
    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    inputs = extractor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits,dim =1)
        predicted_class = probs.argmax().item()
        label = model.config.id2label[predicted_class]

    return f"Predicted emotion: {label}"

if __name__ == "__main__":
    uvicorn.run(app="main:app",reload=True,host="0.0.0.0",port=8000)