# NeuroSecure-Framework

## 📋 Overview
NeuroSecure is a research-oriented framework developed in Java, designed to quantify the mathematical vulnerabilities of neural networks. By implementing gradient-based perturbation algorithms, this project explores the "brittleness" of modern machine learning models and proposes defensive mechanisms through adversarial training.

This project was developed by a team of three to bridge the gap between **Multivariable Calculus** and **Robust Software Engineering**.

## 👥 The Team
* **Ethan Hahn (@godehy26)**: Lead Technical Architect. Responsible for the Core Engine, JVM optimization, and Matrix logic.
* **[Member 2]**: Theoretical Lead. Responsible for mathematical proofs, gradient derivations, and LaTeX documentation.
* **[Member 3]**: Research Analyst. Responsible for benchmarking, data visualization, and saliency mapping.

## 🚀 Key Features
* **Custom Linear Algebra Engine:** High-performance matrix operations built from scratch in Java to ensure total transparency of the backpropagation process.
* **Adversarial Suite:** Implementation of the **Fast Gradient Sign Method (FGSM)** and **Projected Gradient Descent (PGD)**.
* **Security Benchmarking:** Tools to calculate the $L_\infty$ and $L_2$ perturbation limits of trained models.
* **Adversarial Training:** A defensive module designed to increase model resilience against gradient-based "white-box" attacks.

## 🧪 Mathematical Foundation
NeuroSecure operates on the principle of maximizing the loss function $J(\theta, x, y)$ with respect to the input pixels $x$ rather than the weights $\theta$. We utilize the sign of the gradient to generate perturbations:

$$x_{adv} = x + \epsilon \cdot \text{sign}(\nabla_x J(\theta, x, y))$$

This approach allows us to identify the minimum noise threshold required to induce misclassification in a target model.

## 🛠️ Getting Started
### Prerequisites
* JDK 17 or higher
* Maven 3.6+

### Installation
1. Clone the repository:
   ```bash
   git clone [https://github.com/godehy26/neurosecure-framework.git](https://github.com/godehy26/neurosecure-framework.git)
