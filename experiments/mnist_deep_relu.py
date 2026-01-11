import numpy as np
import sys
import os

# Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from engine.network import NeuralNetwork
from engine.layers import DenseLayer, ActivationLayer
from engine.activations import Activations
from engine.loss import Loss
from mnist_loader import load_mnist_data

# 1. දත්ත 5,000ක් ලෝඩ් කරමු (දැන් අපේ engine එකට ලොකු දත්ත දිරවන්න පුළුවන්!)

(x_train, y_train), (x_test, y_test), (y_train_raw, y_test_raw) = load_mnist_data(limit=5000)

net = NeuralNetwork()

# Layer 1: Input (784) -> Hidden 1 (128)
net.add(DenseLayer(784, 128))
net.add(ActivationLayer(Activations.relu, Activations.relu_derivative))

# Layer 2: Hidden 1 (128) -> Hidden 2 (64) - අලුත් Deep Layer එක
net.add(DenseLayer(128, 64))
net.add(ActivationLayer(Activations.relu, Activations.relu_derivative))

# Layer 3: Hidden 2 (64) -> Output (10)
net.add(DenseLayer(64, 10))
net.add(ActivationLayer(Activations.softmax, Activations.softmax_derivative))

net.set_loss(Loss.cross_entropy, Loss.cross_entropy_derivative)

# 2. Training (Learning rate එක 0.01 වගේ පොඩි අගයක් තබන්න ReLU වලදී)
print("🚀 Training Deep ReLU Network...")
net.train(x_train, y_train, epochs=100, learning_rate=0.0001)

# 3. Save the best model
net.save('mnist_deep_model.pkl')


