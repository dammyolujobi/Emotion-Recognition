from flask import Flask, render_template_string, request
from transformers import AutoModelForImageClassification, AutoImageProcessor
from PIL import Image
import torch
import io

# Load model and processor
extractor = AutoImageProcessor.from_pretrained("dima806/facial_emotions_image_detection", use_fast=True)
model = AutoModelForImageClassification.from_pretrained("dima806/facial_emotions_image_detection")

app = Flask(__name__)

# HTML + CSS (Apple-style minimalism)
template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Emotion Recognizer</title>
    <style>
        /* Apple-style design: simplicity, white space, elegance */
        body {
            font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "Helvetica Neue", sans-serif;
            background-color: #f5f5f7;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            height: 100vh;
            margin: 0;
        }

        .container {
            background: white;
            border-radius: 24px;
            box-shadow: 0 8px 24px rgba(0,0,0,0.08);
            padding: 40px;
            width: 400px;
            text-align: center;
            transition: transform 0.3s ease;
        }

        .container:hover {
            transform: translateY(-5px);
        }

        h1 {
            color: #1d1d1f;
            font-size: 28px;
            margin-bottom: 20px;
            letter-spacing: -0.5px;
        }

        input[type="file"] {
            margin: 20px 0;
            padding: 10px;
            border-radius: 12px;
            border: 1px solid #d2d2d7;
            background-color: #f9f9fb;
            cursor: pointer;
            width: 100%;
        }

        button {
            background-color: #007aff;
            color: white;
            border: none;
            padding: 12px 20px;
            border-radius: 12px;
            cursor: pointer;
            font-size: 16px;
            transition: background-color 0.3s ease;
            width: 100%;
        }

        button:hover {
            background-color: #005ecb;
        }

        .result {
            margin-top: 20px;
            font-size: 18px;
            color: #1d1d1f;
            background-color: #f5f5f7;
            border-radius: 12px;
            padding: 15px;
            font-weight: 500;
        }

        img {
            max-width: 100%;
            border-radius: 16px;
            margin-top: 15px;
        }

        footer {
            position: fixed;
            bottom: 20px;
            color: #86868b;
            font-size: 14px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Emotion Recognizer</h1>
        <form method="POST" enctype="multipart/form-data">
            <input type="file" name="file" accept="image/*" required>
            <button type="submit">Analyze Emotion</button>
        </form>

        {% if label %}
        <div class="result">
            Predicted Emotion: <strong>{{ label }}</strong>
        </div>
        <img src="data:image/jpeg;base64,{{ image_data }}" alt="Uploaded Image">
        {% endif %}
    </div>

    <footer>Designed with 🍎 Apple Design Principles</footer>
</body>
</html>
"""

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

    return render_template_string(template, label=label, image_data=image_data)

if __name__ == "__main__":
    app.run(debug=True)
