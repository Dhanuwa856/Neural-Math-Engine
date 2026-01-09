# Calculus in Neural Networks

To minimize the error (Loss), the network must update its weights. We use **Derivatives** to find the direction of the steepest descent.

## 1. The Derivative
The derivative $f'(x)$ tells us how much $f(x)$ changes when $x$ changes by a tiny amount.

## 2. Chain Rule (The Core of Backpropagation)
Since a neural network is a composition of functions (Layer 1 -> Activation -> Layer 2), we use the **Chain Rule** to calculate gradients for earlier layers:
$$\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx}$$

This allows us to pass the "blame" for an error backward through the network.