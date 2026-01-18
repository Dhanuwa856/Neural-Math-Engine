import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from PIL import Image

# Engine එකට path එක ලබා දීම
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from engine.network import NeuralNetwork
from engine.layers import DenseLayer, ActivationLayer
from engine.activations import Activations
from engine.loss import Loss
from experiments.mnist_loader import load_mnist_data


def run_xor_experiment():
    print("\n🚀 Running XOR Problem Experiment...")
    x_train = np.array([[[0, 0]], [[0, 1]], [[1, 0]], [[1, 1]]])
    y_train = np.array([[[0]], [[1]], [[1]], [[0]]])

    net = NeuralNetwork()
    net.add(DenseLayer(2, 4))
    net.add(ActivationLayer(Activations.sigmoid, Activations.sigmoid_derivative))
    net.add(DenseLayer(4, 1))
    net.add(ActivationLayer(Activations.sigmoid, Activations.sigmoid_derivative))
    net.set_loss(Loss.mse, Loss.mse_derivative)

    print("--- Training Started ---")
    errors = net.train(x_train, y_train, epochs=5000, learning_rate=0.2)
    print("--- Training Completed ---")

    # Final Testing Output
    print("\n" + "=" * 45)
    print(f"{'INPUT':<15} | {'TARGET':<10} | {'PREDICTION':<12}")
    print("-" * 45)
    for x, y in zip(x_train, y_train):
        output = net.predict(x)
        pred_val = output[0][0]
        result = np.round(pred_val)
        print(f"{str(x[0]):<15} | {str(y[0]):<10} | {pred_val:.4f} (-> {int(result)})")
    print("=" * 45)

    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(errors, color='blue', linewidth=2)
    plt.xlabel('Epochs')
    plt.ylabel('Mean Squared Error')
    plt.title('Learning Curve (XOR Problem)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()


def run_mnist_training():
    print("\n🚀 Starting MNIST Deep Training...")
    (x_train, y_train), (x_test, y_test), (y_train_raw, y_test_raw) = load_mnist_data(limit=5000)

    net = NeuralNetwork()
    net.add(DenseLayer(784, 128))
    net.add(ActivationLayer(Activations.relu, Activations.relu_derivative))
    net.add(DenseLayer(128, 64))
    net.add(ActivationLayer(Activations.relu, Activations.relu_derivative))
    net.add(DenseLayer(64, 10))
    net.add(ActivationLayer(Activations.softmax, Activations.softmax_derivative))

    net.set_loss(Loss.cross_entropy, Loss.cross_entropy_derivative)

    # NaN පාලනය සඳහා learning_rate=0.0001 පාවිච්චි වේ
    net.train(x_train, y_train, epochs=70, learning_rate=0.0001)
    net.save('mnist_deep_model.pkl')



def run_custom_prediction():
    image_path = input("\nEnter the image path (e.g., digit-04.png): ")
    if not os.path.exists(image_path):
        print(f"❌ Error: {image_path} not found!")
        return

    print(f"🚀 Predicting for: {image_path}...")
    img = Image.open(image_path).convert('L').resize((28, 28))
    img_arr = np.array(img).astype('float32') / 255.0
    input_data = img_arr.reshape(1, 784)

    try:
        net = NeuralNetwork.load('mnist_deep_model.pkl')
        output = net.predict(input_data)
        prediction = np.argmax(output)
        confidence = np.max(output) * 100
        print(f"\nAI Prediction: {prediction}")
        print(f"Confidence: {confidence:.2f}%")
    except FileNotFoundError:
        print("❌ Error: 'mnist_deep_model.pkl' not found. Train the model first!")


def main_menu():
    while True:
        print("\n" + "=" * 50)
        print("      🧠 NEURAL-MATH-ENGINE (From Scratch) 🧠")
        print("=" * 50)
        print("1. Solve XOR Problem (Logic Test)")
        print("2. Train MNIST Deep Network (5,000 Samples)")
        print("3. Predict Your Hand-drawn Digit")
        print("4. Exit")

        choice = input("\nEnter your choice (1-4): ")

        if choice == '1':
            run_xor_experiment()
        elif choice == '2':
            run_mnist_training()
        elif choice == '3':
            run_custom_prediction()
        elif choice == '4':
            print("Happy Coding! 💪❤️");
            break
        else:
            print("❌ Invalid choice!")


if __name__ == "__main__":
    main_menu()