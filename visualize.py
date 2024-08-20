from __future__ import print_function
import argparse
import os
import cv2
import pandas as pd
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from deep_emotion import Deep_Emotion
from data_loaders import PlainDataset, eval_data_dataloader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Argument parsing
parser = argparse.ArgumentParser(description="Configuration of testing process")
parser.add_argument('-d', '--data', type=str, required=True, help='Folder that contains finaltest.csv and test images')
parser.add_argument('-m', '--model', type=str, required=True, help='Path to pretrained model')
parser.add_argument('-t', '--test_acc', action='store_true', help='Show test accuracy')
parser.add_argument('-c', '--cam', action='store_true', help='Test the model in real-time with a webcam')
args = parser.parse_args()

# Transformations for the test images
transformation = transforms.Compose([
    transforms.Resize((48, 48)),  # Resize images to 48x48 as expected by the model
    transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Dataset and DataLoader
test_csv = os.path.join(args.data, 'val.csv')
test_img_dir = os.path.join(args.data, 'val')
if not os.path.isfile(test_csv):
    raise FileNotFoundError(f"CSV file not found: {test_csv}")
dataset = PlainDataset(csv_file=test_csv, img_dir=test_img_dir, data_type='val', transform=transformation)
test_loader = DataLoader(dataset, batch_size=64, num_workers=0)

# Load the model
net = Deep_Emotion()
print("Deep Emotion Model:", net)
if not os.path.isfile(args.model):
    raise FileNotFoundError(f"Model file not found: {args.model}")
# Load model weights securely
model_weights = torch.load(args.model, map_location=device, weights_only=True)
net.load_state_dict(model_weights)
net.to(device)
net.eval()

# Model Evaluation on test data
classes = ('Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral')
total_accuracy = []

if args.test_acc:
    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)
            outputs = net(data)
            pred_probs = F.softmax(outputs, dim=1)
            predictions = torch.argmax(pred_probs, dim=1)
            correct_predictions = torch.sum(predictions == labels)
            accuracy = correct_predictions.item() / data.size(0)
            total_accuracy.append(accuracy)

    avg_accuracy = np.mean(total_accuracy)
    print(f'Accuracy of the network on the test images: {avg_accuracy * 100:.2f}%')

# Helper function for real-time testing
def load_img(path):
    img = Image.open(path)
    img = transformation(img).float()
    img = torch.autograd.Variable(img, requires_grad=False)
    img = img.unsqueeze(0)
    return img.to(device)

if args.cam:
    # Load the face detection model
    face_cascade = cv2.CascadeClassifier('cascade_model/haarcascade_frontalface_default.xml')
    if face_cascade.empty():
        raise RuntimeError("Cascade classifier XML file not found or is empty.")

    # Capture video from webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot access the webcam.")

    while True:
        ret, img = cap.read()
        if not ret:
            print("Failed to capture image from webcam.")
            break

        # Convert to grayscale and detect faces
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        # Draw rectangle around faces and make predictions
        for (x, y, w, h) in faces:
            roi = img[y:y+h, x:x+w]
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            roi_resized = cv2.resize(roi_gray, (48, 48))
            roi_path = "roi.jpg"
            cv2.imwrite(roi_path, roi_resized)
            
            imgg = load_img(roi_path)
            with torch.no_grad():
                out = net(imgg)
                pred_probs = F.softmax(out, dim=1)
                class_idx = torch.argmax(pred_probs, dim=1).item()
                prediction = classes[class_idx]

            # Display the prediction on the image
            font = cv2.FONT_HERSHEY_SIMPLEX
            org = (50, 50)
            fontScale = 1
            color = (255, 0, 0)
            thickness = 2
            img = cv2.putText(img, prediction, org, font, fontScale, color, thickness, cv2.LINE_AA)
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

        cv2.imshow('Real-time Emotion Recognition', img)
        
        # Exit on 'q' key press
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    # Release the video capture object
    cap.release()
    cv2.destroyAllWindows()
