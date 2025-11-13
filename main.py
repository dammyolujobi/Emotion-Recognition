from flask import Flask, render_template_string, request
from transformers import AutoModelForImageClassification, AutoImageProcessor
from PIL import Image
import torch
import io

# Load model and processor
extractor = AutoImageProcessor.from_pretrained("dima806/facial_emotions_image_detection", use_fast=True)
model = AutoModelForImageClassification.from_pretrained("dima806/facial_emotions_image_detection")

app = Flask(__name__)

import base64

@app.route("/", methods=["GET", "POST"])
def index():
    label = None
    image_data = None

    if request.method == "POST":
        file = request.files["file"]
        if file:
            image = Image.open(io.BytesIO(file.read())).convert("RGB")
            inputs = extractor(images=image, return_tensors="pt")

            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=1)
                predicted_class = probs.argmax().item()
                label = model.config.id2label[predicted_class]

            # Convert image to base64 for inline HTML display
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG")
            image_data = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return render_template_string(template = "Template\main.html", label=label, image_data=image_data)

if __name__ == "__main__":
    app.run(debug=True)
