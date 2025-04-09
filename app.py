from flask import Flask, render_template, request
from PIL import Image
from io import BytesIO
import base64
from predict import predict_potato, predict_tomato
from model import tomato_model
import torch

app = Flask(__name__)

import zipfile

# Unzip file
with zipfile.ZipFile("tomato_model_statedict_f.pth.zip", 'r') as zip_ref:
    zip_ref.extractall("tomato_model_statedict_f")  # or specify any target directory


# Load models
tomato_model.load_state_dict(torch.load("tomato_model_statedict_f/tomato_model_statedict_f.pth", map_location=torch.device('cpu')))

@app.route('/')
def home():
    # Default to potato model
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Get the image file from the request
    file = request.files['file']

    # Make prediction
    class_name, probability, image = predict_tomato(file, tomato_model)

    # Set background image (empty in this case)
    background_image = r''

    # Convert image to base64 format
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    print("probability", probability)  # Debug print

    # Pass the raw probability value (as a float) to the template
    return render_template('index.html',
                          image=img_str,
                          class_name=class_name,
                          probability=probability,  # Send as raw float, not formatted string
                          background_image=background_image)

if __name__ == '__main__':
    app.run(debug=True)
