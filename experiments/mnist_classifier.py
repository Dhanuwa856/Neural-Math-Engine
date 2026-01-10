import numpy as np
import sys
import os

# Path setup
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from engine.network import NeuralNetwork
from engine.layers import DenseLayer, ActivationLayer
from engine.activations import Activations
from engine.loss import Loss
from mnist_loader import load_mnist_data

# 1. දත්ත Load කරගනිමු (පින්තූර 2000ක් ගමු දැන් accuracy එක වැඩි වෙන්න)
(x_train, y_train), (x_test, y_test), (y_train_raw, y_test_raw) = load_mnist_data(limit=2000)

# 2. Network Architecture
net = NeuralNetwork()

# Input (784) -> Hidden (128) - Neurons ගණන 128 දක්වා වැඩි කළා
net.add(DenseLayer(784, 128))
net.add(ActivationLayer(Activations.sigmoid, Activations.sigmoid_derivative))

# Hidden (128) -> Output (10)
net.add(DenseLayer(128, 10))
# මෙන්න වෙනස: අන්තිමට Softmax පාවිච්චි කරනවා
net.add(ActivationLayer(Activations.softmax, Activations.softmax_derivative))

# මෙන්න වෙනස: Loss එක Cross-Entropy කරනවා
net.set_loss(Loss.cross_entropy, Loss.cross_entropy_derivative)

# 3. Train කරමු
print("Training Smart MNIST Classifier...")
# Learning rate එක 0.1 සහ epochs 50ක් දෙමු
net.train(x_train, y_train, epochs=50, learning_rate=0.1)

# 4. Final Testing
print("\n" + "="*30)
print("RESULTS AFTER UPGRADE")
print("="*30)
for i in range(10): # ඉලක්කම් 10ක් බලමු
    output = net.predict(x_test[i])
    prediction = np.argmax(output)
    actual = y_test_raw[i]
    status = "✅" if prediction == actual else "❌"
    print(f"Actual: {actual} | Predicted: {prediction} {status}")