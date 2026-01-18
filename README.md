# Neural-Math-Engine 🧠🔥
### Building Deep Learning from Scratch with Pure Mathematics

This project is a high-level implementation of a Deep Learning engine built using only **Python** and **NumPy**. It is designed to showcase the fundamental mathematics of Artificial Intelligence—specifically **Linear Algebra**, **Calculus**, and **Probability**—without the use of high-level frameworks like TensorFlow or PyTorch.

---

## 🚀 Key Achievements

### 1. Solving the XOR Problem (The Proof of Concept)
The engine successfully solved the non-linearly separable XOR logic gate.
- **Architecture**: 2 Input → 4 Hidden (Sigmoid) → 1 Output (Sigmoid)
- **Accuracy**: ~99%

### 2. MNIST Handwritten Digit Recognition (The Real Test)
The engine was upgraded to a 3-layer deep architecture to classify handwritten digits (0-9).
- **Test Accuracy**: **89.60%**
- **Real-world Inference**: Achieved **95.60% confidence** on custom hand-drawn digits.

---

## 📊 Visualizing Results

### Confusion Matrix
This chart illustrates where the model performed perfectly and where it faced challenges (e.g., distinguishing between 3 and 9).

![Confusion Matrix](/assets/img.png) 


### Real-world Testing (Inference)
Testing the model with digits drawn in MS Paint:
- **Digit 5**: 95.60% Confidence ✅
- **Digit 2**: 89.98% Confidence ✅

### Real-world Testing Results
|                  Target Digit: 5 (95.60%)                   |               Target Digit: 2 (89.98%)               |
|:-----------------------------------------------------------:|:----------------------------------------------------:|
|       ![Digit 5 MS Paint](/experiments/digit-04.png)        |   ![Digit 2 MS Paint ](/experiments/digit-05.png)    |
|    ![Digit 5](/assets/Screenshot05.png)     | ![Digit 2](/assets/Screenshot02.png) |

---

## 🛠️ Challenges & Debugging (The Learning Journey)

Building an engine from scratch isn't easy. Here are the major hurdles I overcame:

### 1. The "NaN" Mystery (Exploding Gradients)
During the transition to Deep ReLU networks, the loss often became `NaN`.
- **Cause**: Gradients were exploding due to improper weight initialization and high learning rates.
- **Solution**: 
    - Implemented **He Initialization** ($W = \text{randn} \cdot \sqrt{2/n}$) to keep weight variance stable.
    - Introduced **Gradient Clipping** to prevent weight updates from exceeding a safe threshold.
    - Optimized the **Learning Rate** (decreased from 0.01 to 0.0001 as data limit increased).

### 2. Data-Centric AI (Why 8% Accuracy?)
Initially, my custom hand-drawn digits failed (Accuracy ~8.60%). 
- **The Lesson**: The model is highly sensitive to the data distribution.
- **Fix**: Ensuring hand-drawn digits were **bold (thick strokes)** and **centered**, matching the MNIST training set.

---

## 🧮 Mathematical Foundations
- **Linear Algebra**: Implementing vectorized Dot Products for Forward Propagation ($Z = W \cdot X + b$).
- **Calculus**: Manual implementation of **Backpropagation** using the **Chain Rule** to update weights and biases.
- **Activation Functions**: Manual code for Sigmoid, Tanh, Softmax, and **ReLU** with their respective derivatives.
- **Loss Functions**: Categorical Cross-Entropy for multi-class classification and Mean Squared Error (MSE) for regression.

---

## 📂 Project Structure
```text
Neural-Math-Engine/
│
├── engine/             # Core Engine Logic
│   ├── activations.py  # ReLU, Softmax, Sigmoid
│   ├── layers.py       # DenseLayer with He Init & Gradient Clipping
│   ├── loss.py         # Cross-Entropy & MSE
│   └── network.py      # Model Persistence (Save/Load) & Training Loop
│
├── experiments/        # Practical Applications
│   ├── xor_problem.py  
│   ├── mnist_deep_relu.py
│   └── predict_custom.py # Real-world inference script
│
└── mnist_model.pkl     # Pre-trained 89% accuracy model
````
---
## 🕹️ How to Use (Main Entry Point)
The engine features a built-in CLI for easy navigation between different experiments.

1. **Clone the repository**:
   ```bash
   git clone [https://github.com/Dhanuwa856/Neural-Math-Engine.git](https://github.com/Dhanuwa856/Neural-Math-Engine.git)
   cd Neural-math-engine
   ```
2. **Run the Main Control Center:**
```bash
python main.py
```
3. **Choose an Option:**
- Option 1 (XOR Problem): Demonstrates the engine's ability to solve non-linear logic gates using the Sigmoid activation function.
- Option 2 (MNIST Training): Trains a 3-layer Deep ReLU network on 5,000 samples. It uses He Initialization and Gradient Clipping to ensure a stable 89%+ accuracy.
- Option 3 (Custom Prediction): Test the engine with your own drawings! Place a 28x28 .png file in the root directory and see the AI identify it with high confidence (e.g., 95.60% for a bold '5').
   

---
## 📦 Requirements
- Python 3.x
- NumPy
- Matplotlib (for visualization)
- Pillow (for custom image inference)
---
## 🏗️ Future Scope
- Implement **Convolutional Neural Networks (CNN)** layers for better image feature extraction.
- Add support for **Momentum** and **Adam** optimizers.
---
## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Developed by**: Dhanushka Rathnayaka | IT Student at ITUM
