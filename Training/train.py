from transformers import  AutoModelForImageClassification,AutoImageProcessor
import torch
from PIL import Image

extractor = AutoImageProcessor.from_pretrained("dima806/facial_emotions_image_detection")
model = AutoModelForImageClassification.from_pretrained("dima806/facial_emotions_image_detection")

image = Image.open("unnamed.jpg").convert("RGB")

inputs = extractor(images=image, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits,dim =1)
    predicted_class = probs.argmax().item()
    label = model.config.id2label[predicted_class]

print(f"Predicted emotion: {label}")