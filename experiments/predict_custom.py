import numpy as np
import sys
import os
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from engine.network import NeuralNetwork

def predict_my_digit(image_path):
    # 1. පින්තූරය Load කර Grayscale කරගන්න
    img = Image.open(image_path).convert('L')

    # 2. ප්‍රමාණය 28x28 ද කියා නැවත තහවුරු කරගන්න
    img = img.resize((28, 28))

    # 3. පින්තූරය Array එකක් බවට හරවා Normalize කරන්න
    img_arr = np.array(img).astype('float32')/ 255.0

    # 4. Shape එක (1, 784) ලෙස සකසන්න
    input_data = img_arr.reshape(1,784)

    # 5. Model එක Load කර Predict කරන්න
    net = NeuralNetwork.load('mnist_deep_model.pkl')
    output = net.predict(input_data)

    prediction = np.argmax(output)
    confidence = np.max(output) * 100

    print(f"\nTarget Image: {image_path}")
    print(f"AI Prediction: {prediction}")
    print(f"Confidence: {confidence:.2f}%")


if __name__ == "__main__":
    predict_my_digit('digit-04.png')
