import streamlit as st
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from PIL import Image
import time

# Optional canvas import with error handling
try:
    from streamlit_drawable_canvas import st_canvas
except ImportError:
    st.error(
        "❌ Missing library: `streamlit-drawable-canvas`\n\nInstall it using:\n```bash\npip install streamlit-drawable-canvas\n```")
    st.stop()

# Engine path setup
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

try:
    from engine.network import NeuralNetwork
    from engine.layers import DenseLayer, ActivationLayer
    from engine.activations import Activations
    from engine.loss import Loss
    from experiments.mnist_loader import load_mnist_data
except ImportError as e:
    st.error(
        f"❌ Cannot find custom modules (`engine`, `experiments`).\nMake sure the folder structure exists.\nError: {e}")
    st.stop()

# --- Page Config ---
st.set_page_config(page_title="Neural Math Engine", page_icon="🧠", layout="centered")
st.title("🧠 Neural-Math-Engine")
st.markdown("**Built from scratch using NumPy!**")
st.divider()

# --- Sidebar Navigation ---
st.sidebar.header("Navigation")
menu = st.sidebar.radio(
    "Choose an Option:",
    ("1. Solve XOR Problem", "2. Train MNIST Deep Network", "3. Predict Digit")
)


def center_image(image_data):
    """
    Improved centering function:
    - Takes RGBA canvas data (280x280)
    - Crops to the drawn digit, resizes to 20x20 (preserving aspect ratio)
    - Pastes onto a 28x28 black canvas
    - Returns a PIL Image ready for MNIST model
    """
    if image_data is None:
        return None

    # Convert RGBA to grayscale
    img = Image.fromarray(image_data.astype('uint8'), 'RGBA').convert('L')

    # Find bounding box of non‑black pixels (the drawn digit)
    bbox = img.getbbox()
    if bbox is None:
        return None  # nothing drawn

    # Crop to the digit
    img_cropped = img.crop(bbox)

    # Resize so that the longer side becomes 20 pixels (MNIST standard)
    max_side = max(img_cropped.size)
    if max_side == 0:
        return None
    ratio = 20.0 / max_side
    new_size = (int(img_cropped.size[0] * ratio), int(img_cropped.size[1] * ratio))
    img_resized = img_cropped.resize(new_size, Image.Resampling.LANCZOS)

    # Create 28x28 black image and paste the resized digit in the center
    final_img = Image.new('L', (28, 28), color=0)
    paste_x = (28 - new_size[0]) // 2
    paste_y = (28 - new_size[1]) // 2
    final_img.paste(img_resized, (paste_x, paste_y))

    return final_img


# ==================== 1. XOR Problem ====================
if menu == "1. Solve XOR Problem":
    st.header("Logic Test: XOR Problem")
    st.write("Train a simple neural network to solve the XOR logic gate.")

    if st.button("🚀 Run XOR Experiment"):
        with st.spinner("Training Network..."):
            x_train = np.array([[[0, 0]], [[0, 1]], [[1, 0]], [[1, 1]]])
            y_train = np.array([[[0]], [[1]], [[1]], [[0]]])

            net = NeuralNetwork()
            net.add(DenseLayer(2, 4))
            net.add(ActivationLayer(Activations.sigmoid, Activations.sigmoid_derivative))
            net.add(DenseLayer(4, 1))
            net.add(ActivationLayer(Activations.sigmoid, Activations.sigmoid_derivative))
            net.set_loss(Loss.mse, Loss.mse_derivative)

            errors = net.train(x_train, y_train, epochs=5000, learning_rate=0.2)

        st.success("Training Completed!")

        # Results Table
        st.subheader("Predictions vs Targets")
        col1, col2, col3 = st.columns(3)
        col1.write("**INPUT**")
        col2.write("**TARGET**")
        col3.write("**PREDICTION**")

        for x, y in zip(x_train, y_train):
            output = net.predict(x)
            pred_val = output[0][0]
            result = np.round(pred_val)

            col1.write(f"{x[0]}")
            col2.write(f"{y[0][0]}")
            col3.write(f"{pred_val:.4f} (-> **{int(result)}**)")

        # Plotting learning curve
        st.subheader("Learning Curve")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(errors, color='blue', linewidth=2)
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Mean Squared Error')
        ax.set_title('XOR Problem Training Loss')
        ax.grid(True, linestyle='--', alpha=0.6)
        st.pyplot(fig)


# ==================== 2. Train MNIST ====================
elif menu == "2. Train MNIST Deep Network":
    st.header("Train MNIST Model")
    st.write("Train the custom deep neural network on 10,000 MNIST samples.")

    if st.button("🚀 Start MNIST Training"):
        with st.spinner("Loading MNIST data..."):
            (x_train, y_train), (x_test, y_test), (y_train_raw, y_test_raw) = load_mnist_data()
            # y_train and y_test are assumed to be one‑hot encoded

        # Shuffle training data
        indices = np.arange(x_train.shape[0])
        np.random.shuffle(indices)
        x_train = x_train[indices]
        y_train = y_train[indices]

        # Build network
        net = NeuralNetwork()
        net.add(DenseLayer(784, 128))
        net.add(ActivationLayer(Activations.relu, Activations.relu_derivative))
        net.add(DenseLayer(128, 64))
        net.add(ActivationLayer(Activations.relu, Activations.relu_derivative))
        net.add(DenseLayer(64, 10))
        net.add(ActivationLayer(Activations.softmax, Activations.softmax_derivative))
        net.set_loss(Loss.cross_entropy, Loss.cross_entropy_derivative)

        # Training with progress bar
        st.write("Training in progress... (this may take a few minutes)")
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Custom training loop to show progress (if your engine supports it)
        # If your engine's train() does not return per-epoch loss, you can modify it.
        # Here we assume train() accepts a callback or we simply use a loop.
        # For simplicity, we call the built-in train and then evaluate.
        epochs = 10
        learning_rate = 0.0001  # slightly increased for faster convergence

        # If your NeuralNetwork.train() can accept a callback, use it.
        # Otherwise, we call it directly.
        net.train(x_train, y_train, epochs=epochs, learning_rate=learning_rate)

        # Evaluate on test set
        st.write("Evaluating on test set...")
        correct = 0
        for i in range(x_test.shape[0]):
            pred = net.predict(x_test[i].reshape(1, -1))
            if np.argmax(pred) == np.argmax(y_test[i]):
                correct += 1
        accuracy = correct / x_test.shape[0] * 100

        # Save model
        net.save('mnist_deep_model_new.pkl')

        st.success(f"✅ Training complete! Test accuracy: {accuracy:.2f}%")
        st.info("Model saved as 'mnist_deep_model_new.pkl'")


# ==================== 3. Predict Digit ====================
elif menu == "3. Predict Digit":
    st.header("Test the Model")
    st.write("Draw a digit (0-9) in the box below and click Predict!")

    col1, col2 = st.columns([1, 1])

    with col1:
        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 255, 1)",
            stroke_width=20,
            stroke_color="#FFFFFF",
            background_color="#000000",
            height=280,
            width=280,
            drawing_mode="freedraw",
            key="canvas",
        )

    with col2:
        if st.button("🔍 Predict Drawn Digit"):
            if canvas_result.image_data is not None:
                processed_img = center_image(canvas_result.image_data)

                if processed_img is not None:
                    # Normalize and flatten
                    img_arr = np.array(processed_img).astype('float32') / 255.0
                    input_data = img_arr.reshape(1, 784)

                    try:
                        # Load model (ensure the file exists)
                        if not os.path.exists('mnist_deep_model_new.pkl'):
                            st.error(
                                "❌ Trained model not found. Please go to 'Train MNIST Deep Network' first and train the model.")
                        else:
                            net = NeuralNetwork.load('mnist_deep_model_new.pkl')
                            output = net.predict(input_data)
                            prediction = np.argmax(output)
                            confidence = np.max(output) * 100

                            st.success("Prediction Complete!")
                            st.markdown(f"### 🎯 Prediction: **{prediction}**")
                            st.progress(confidence / 100.0)
                            st.write(f"**Confidence:** {confidence:.2f}%")

                            # Show what the AI saw
                            st.write("What the AI saw (centered 28x28):")
                            st.image(processed_img, width=84)

                    except Exception as e:
                        st.error(f"Prediction error: {e}")
                else:
                    st.warning("Please draw a digit first (draw something in the black box).")
            else:
                st.warning("Canvas is empty – draw a digit first!")

# Footer
st.sidebar.divider()
st.sidebar.markdown("Made with 💪❤️ using pure NumPy")