# When Models Don't Know What They Don't Know
## Domain Shift and Uncertainty Estimation in Medical Image Classification

---

A deep learning model that performs well in the lab can fail silently in the clinic.  
The problem is not just that the model makes mistakes — it is that it makes mistakes **confidently**.

This repository demonstrates the domain shift problem in medical image classification and shows how uncertainty estimation can detect it.

---

## The Problem

When a model is deployed on data from a different distribution than it was trained on — different hospital, different scanner, different staining protocol — standard softmax confidence scores give **no warning**. The model can be 95% confident and completely wrong.

This project asks: *if you deployed a model trained on colorectal cancer tissue on kidney tissue data, would you know it was failing?*

---

## What This Notebook Demonstrates

| Step | What We Show |
|------|-------------|
| 1 | Domain gap between source and target tissue images |
| 2 | Performance drop under domain shift |
| 3 | Softmax confidence fails to flag OOD samples |
| 4 | MC Dropout uncertainty correctly signals distribution shift |
| 5 | Uncertainty tracks prediction errors |
| 6 | Selective prediction: abstaining on uncertain samples improves accuracy |
| 7 | Grad-CAM reveals what the model attends to in each domain |

---

## Datasets

Both from [MedMNIST](https://medmnist.com/) — downloaded automatically.

| Dataset | Domain | Description | Classes |
|---------|--------|-------------|---------|
| **PathMNIST** | Source (train) | H&E stained colorectal cancer histology patches | 9 tissue types |
| **TissueMNIST** | Target (OOD) | Human kidney cortex microscopy | 8 cell types |

The domain shift is realistic: different organ, different staining protocol, different imaging modality — exactly the kind of shift that occurs in clinical deployment.

---

## Methods

- **Model**: ResNet-18 fine-tuned on PathMNIST
- **Uncertainty**: Monte Carlo Dropout (Gal & Ghahramani, 2016) — dropout kept active at inference, T=30 stochastic forward passes
- **Interpretability**: Grad-CAM on the last convolutional layer
- **Evaluation**: Confidence distributions, uncertainty distributions, selective prediction curves

---

## Setup

```bash
git clone https://github.com/Rajaeereza/domain-shift-uncertainty-medical-imaging
cd domain-shift-uncertainty-medical-imaging
pip install -r requirements.txt
jupyter notebook domain_shift_uncertainty.ipynb
```

---

## Requirements

```
torch>=2.0.0
torchvision>=0.15.0
medmnist>=3.0.0
numpy
matplotlib
scikit-learn
opencv-python
jupyter
```

---

## References

- Gal, Y. & Ghahramani, Z. (2016). Dropout as a Bayesian Approximation. *ICML*.
- Selvaraju, R. et al. (2017). Grad-CAM. *ICCV*.
- Yang, J. et al. (2023). MedMNIST v2. *Scientific Data*.
- Ovadia, Y. et al. (2019). Can You Trust Your Model's Uncertainty? *NeurIPS*.

---

MIT License
