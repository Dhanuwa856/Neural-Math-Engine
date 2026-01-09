import numpy as np
import sys
import os

# අපේ engine එක පාවිච්චි කිරීමට path එක සකස් කරමු
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from engine.network import NeuralNetwork
from engine.layers import DenseLayer, ActivationLayer # ActivationLayer එකත් ගමු
from engine.activations import Activations
from engine.loss import Loss

# 1. දත්ත සකස් කරමු (XOR Data)
x_train = np.array([[[0,0]], [[0,1]], [[1,0]], [[1,1]]])
y_train = np.array([[[0]], [[1]], [[1]], [[0]]])

net = NeuralNetwork()

# Layer 1: Input (2) -> Hidden (3)
net.add(DenseLayer(2, 4))
net.add(ActivationLayer(Activations.sigmoid, Activations.sigmoid_derivative))

# Layer 2: Hidden (3) -> Output (1)
net.add(DenseLayer(4, 1))
net.add(ActivationLayer(Activations.sigmoid, Activations.sigmoid_derivative))

net.set_loss(Loss.mse, Loss.mse_derivative)

# Train
net.train(x_train, y_train, epochs=5000, learning_rate=0.2)

# --- 4. Testing Phase ---
print("\n" + "=" * 30)
print("FINAL TESTING RESULTS")
print("=" * 30)

for x, y in zip(x_train, y_train):
    # Model එකෙන් output එක ලබා ගැනීම
    output = net.predict(x)

    # ප්‍රතිඵලය ලස්සනට පෙන්වීම
    # np.round පාවිච්චි කරන්නේ 0.002 වගේ අගයන් 0 ටත්, 0.99 වගේ ඒවා 1 ටත් සමාන කර බලන්නයි
    print(f"Input: {x[0]} | Target: {y[0]} | Prediction: {output[0]} | Result: {np.round(output[0])}")

print("=" * 30)