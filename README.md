# Learning Probability Density Functions using GANs

## Overview
This project focuses on estimating the probability density function (PDF) of an unknown random variable using only observed samples. Instead of assuming any analytical or parametric distribution (such as Gaussian or exponential), a Generative Adversarial Network (GAN) is trained to learn the distribution directly from data.

The experiment is performed on NO₂ concentration data, which is first transformed using a nonlinear function and then modeled using a GAN.

---

## Dataset
- Feature: NO₂ concentration  
- Source: India Air Quality Dataset (Kaggle)  

### Preprocessing
- Selected only the NO₂ column  
- Removed missing values  
- Converted values to numerical format  
- Used transformed samples for training  

---

## Data Transformation

Each NO₂ value \( x \) is transformed using the nonlinear mapping:

\[
z = x + a_r \sin(b_r x)
\]

where:

- \( r \) = university roll number  
- \( a_r = 0.5 \times (r \bmod 7) \)  
- \( b_r = 0.3 \times ((r \bmod 5) + 1) \)

### Parameters Used (Roll No: 102317272)

- \( a_r = 3.0 \)
- \( b_r = 0.9 \)

Final transformation:

\[
z = x + 3.0 \sin(0.9x)
\]

This introduces non-linearity and produces an unknown probability distribution.

---

## Distribution of Transformed Data

The empirical distribution of the transformed variable is shown below.

![Real Distribution](real_histogram.png)

---

## GAN Architecture

### Generator
- Input: Gaussian noise sampled from \( \mathcal{N}(0,1) \)  
- Output: Synthetic samples of transformed variable  
- Architecture: Fully connected neural network with ReLU activations  

### Discriminator
- Input: Real or generated samples  
- Output: Probability of the sample being real  
- Architecture: Fully connected neural network with Sigmoid output  

---

## Training Details
- Loss function: Binary Cross Entropy  
- Optimizer: Adam  
- Learning rate: 0.001  
- Epochs: 3000  
- Batch size: 256  

No parametric probability distribution is assumed during training.

---

## PDF Estimation

After training the GAN:

1. A large number of samples are generated using the trained generator  
2. Histogram density estimation is applied  
3. The probability density function is approximated  

---

## PDF Approximation Results

### Generated PDF from GAN
![GAN PDF](gan_pdf.png)

### Comparison: Real vs Generated Distribution
![Comparison](gan_comparison.png)

---

## Observations
- Generator successfully captures the overall shape of the distribution  
- Good mode coverage is observed  
- Training remains stable after initial epochs  
- No significant mode collapse detected  
- Minor deviations appear at extreme tails due to limited sampling  

---

## Conclusion
This project demonstrates that Generative Adversarial Networks can effectively learn unknown probability density functions directly from sample data. The generator implicitly models the distribution by transforming random noise into realistic samples, allowing the PDF to be approximated without assuming any predefined analytical form.

---

## Tools and Libraries
- Python  
- NumPy  
- Pandas  
- Matplotlib  
- PyTorch  

