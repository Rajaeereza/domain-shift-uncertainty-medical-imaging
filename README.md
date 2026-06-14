# When Models Don't Know What They Don't Know
## Domain Shift, Uncertainty Estimation, and Domain Adaptation in Medical Image Classification

---

This project grew out of a student tutoring engagement focused on trustworthy AI for medical imaging. While supervising a project on model reliability in clinical deployment, I worked through these problems in depth — from detecting distribution shift to fixing it. The result is a two-notebook tutorial covering the full arc: problem, detection, and solution.

The core observation that motivated this work: a model trained on cancer tissue images from one organ fails silently when deployed on tissue images from a different organ. Standard softmax confidence scores give no warning. This is not a hypothetical concern — it is a fundamental challenge for clinical AI deployment.

---

## Project Structure

```
├── domain_shift_uncertainty.ipynb     # Notebook 1: The Problem
├── domain_adaptation.ipynb            # Notebook 2: The Solution
├── requirements.txt
└── README.md
```

---

## Notebook 1 — The Problem: When Models Don't Know What They Don't Know

Trains a ResNet-18 on colorectal cancer histology patches (PathMNIST) and evaluates it on kidney tissue microscopy images (TissueMNIST). Demonstrates:

| What | Finding |
|------|---------|
| Source accuracy | 84.3% — model works well in training distribution |
| OOD samples with confidence > 90% | 51.8% — confidently wrong predictions |
| MC Dropout uncertainty ratio (OOD/source) | 3.4x higher on OOD samples |
| Uncertainty ratio (incorrect/correct) | 5.22x — uncertainty tracks errors |
| Selective prediction | Abstaining on uncertain samples improves accuracy |
| Grad-CAM | OOD attention falls on arbitrary regions, not tissue structure |

**Key finding:** Softmax confidence is not a reliable OOD detector. MC Dropout uncertainty is.

---

## Notebook 2 — The Solution: From Detection to Adaptation

Loads the saved model from Notebook 1 and goes further:

**Entropy-based OOD detection:**
- Prediction entropy H = -Σ p·log(p) as a single-pass alternative to MC Dropout
- Higher entropy correctly flags OOD samples
- Compared head-to-head with MC Dropout using AUROC

**MC Dropout vs Entropy — head-to-head:**
- Both evaluated as binary OOD classifiers (source=0, target=1)
- AUROC comparison: which method better separates the distributions?
- Entropy advantage: single forward pass vs T=30 stochastic passes

**Domain adaptation:**
- Fine-tune on 500 target domain samples (5 epochs, lr=1e-5)
- Realistic few-shot adaptation scenario
- Before/after: accuracy, uncertainty, entropy, Grad-CAM attention

**Key finding:** Fine-tuning on a small target subset substantially recovers performance — and uncertainty decreases, showing the model genuinely understands the new distribution.

---

## Datasets

Both from [MedMNIST](https://medmnist.com/) — downloaded automatically, no setup required.

| Dataset | Role | Description | Classes |
|---------|------|-------------|---------|
| **PathMNIST** | Source (train + test) | H&E stained colorectal cancer histology | 9 tissue types |
| **TissueMNIST** | Target (adapt + test) | Human kidney cortex microscopy | 8 cell types |

The domain shift is realistic: different organ, different staining, different imaging modality.

---

## Methods

- **Model:** ResNet-18 fine-tuned on source domain
- **MC Dropout:** T=30 stochastic forward passes, dropout kept active at inference
- **Entropy:** Single-pass prediction entropy H = -Σ p·log(p)
- **OOD evaluation:** AUROC, ROC curves
- **Domain adaptation:** Few-shot fine-tuning with low learning rate (1e-5)
- **Interpretability:** Grad-CAM on last convolutional layer, before and after adaptation

---

## Setup

```bash
git clone https://github.com/Rajaeereza/domain-shift-uncertainty-medical-imaging
cd domain-shift-uncertainty-medical-imaging
pip install -r requirements.txt

# Run Notebook 1 first — saves model_weights.pth
jupyter notebook domain_shift_uncertainty.ipynb

# Then run Notebook 2
jupyter notebook domain_adaptation.ipynb
```

---

## The Broader Principle

Trustworthy deployment of medical AI is not a one-time event. It is a process:

1. **Detect** when a model encounters data outside its training distribution
2. **Understand** what the model attends to and why — using interpretability tools
3. **Adapt** with available target domain data before deployment
4. **Monitor** uncertainty continuously after deployment

No single technique solves all of this. But uncertainty estimation, interpretability, and domain adaptation together provide a practical framework for deploying models that know their own limits.

---

## References

- Gal, Y. & Ghahramani, Z. (2016). Dropout as a Bayesian Approximation. *ICML*.
- Selvaraju, R. et al. (2017). Grad-CAM. *ICCV*.
- Yang, J. et al. (2023). MedMNIST v2. *Scientific Data*.
- Ovadia, Y. et al. (2019). Can You Trust Your Model's Uncertainty? *NeurIPS*.
- Wang, M. & Deng, W. (2018). Deep Visual Domain Adaptation: A Survey. *Neurocomputing*.

