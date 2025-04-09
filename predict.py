from torchvision import transforms as tt
from PIL import Image
import torch
from torchvision import transforms as tt
from PIL import Image
import cv2


def predict_potato(image_path, model):

    # Define the pre-processing transform
    transforms = tt.Compose([
    tt.Resize((224, 224)),
    tt.ToTensor()
])
    image = Image.open(image_path).convert("RGB")
    # Pre-process the image
    image_tensor = transforms(image).unsqueeze(0)
    # Set the model to evaluation mode
    model.eval()

    # Make a prediction
    with torch.no_grad():
        output = model(image_tensor)

    # Convert the output to probabilities using softmax
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    # Get the predicted class
    predicted_class = torch.argmax(probabilities).item()
    # Get the probability for the predicted class
    predicted_probability = probabilities[predicted_class].item()
    # Define class labels
    class_labels = ['Potato Early Blight', 'Potato Late Blight', 'Potato Healthy']

    return class_labels[predicted_class], predicted_probability, image


def predict_tomato(image_file, model):
    # Define the pre-processing transform
    transforms = tt.Compose([
        tt.Resize((224, 224)),
        tt.ToTensor()
    ])

    # Load and preprocess the image
    image = Image.open(image_file).convert("RGB")
    image_tensor = transforms(image).unsqueeze(0)

    # Set the model to evaluation mode
    model.eval()

    # Make a prediction
    with torch.no_grad():
        output = model(image_tensor)

    # Convert the output to probabilities using softmax
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    # Get the predicted class
    predicted_class = torch.argmax(probabilities).item()
    # Get the probability for the predicted class
    predicted_probability = probabilities[predicted_class].item()
    # Define class labels for tomato
    class_labels = ['Tomato Early Blight', 'Tomato Late Blight', 'Tomato Healthy']

    return class_labels[predicted_class], predicted_probability, image
