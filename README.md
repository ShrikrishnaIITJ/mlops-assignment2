# MLOps Assignment 2

Text Classification using DistilBERT with Hugging Face, Kaggle, and Weights & Biases (W&B).

---

## Project Overview

This project demonstrates a complete MLOps workflow for training and deploying a transformer-based NLP model. The model was fine-tuned using the Hugging Face Transformers library on a Kaggle GPU environment and tracked using Weights & Biases (W&B).

The final trained model and tokenizer were uploaded to Hugging Face Hub for public access.

---

## Model Used

* Model: `distilbert-base-cased`
* Task: Text Classification
* Framework: Hugging Face Transformers

DistilBERT was selected because it is lightweight, faster than BERT, and performs well for NLP classification tasks while requiring fewer computational resources.

---

## Training Configuration

| Parameter     | Value           |
| ------------- | --------------- |
| Platform      | Kaggle Notebook |
| GPU           | Tesla T4        |
| Epochs        | 3               |
| Batch Size    | 8               |
| Learning Rate | 3e-5            |

---

## Evaluation Results

| Metric    | Score  |
| --------- | ------ |
| Accuracy  | 0.5981 |
| F1 Score  | 0.6023 |
| Precision | 0.6105 |
| Recall    | 0.5981 |
| Eval Loss | 2.3864 |

---

## Project Links

### GitHub Repository

PASTE_YOUR_GITHUB_LINK

### Kaggle Notebook

PASTE_YOUR_KAGGLE_NOTEBOOK_LINK

### Hugging Face Model

https://huggingface.co/ShrikrishnaIITJ/distilbert-goodreads-genres

### W&B Dashboard

https://wandb.ai/g25ait2103-indian-institute-of-technology-jodhpur/huggingface

---

## Installation

```bash id="3u6wgt"
pip install transformers datasets evaluate accelerate wandb huggingface_hub scikit-learn torch
```

---

## Running the Project

1. Open the notebook in Kaggle or Google Colab.
2. Enable GPU acceleration.
3. Install required dependencies.
4. Run all notebook cells sequentially.
5. Training logs and metrics will appear in W&B.
6. The trained model will be uploaded to Hugging Face Hub.

---

## Files Included

* Training Notebook (`.ipynb`)
* README.md
* requirements.txt
* Evaluation Report

---

## What I Learned

This assignment helped me understand:

* GPU-based model training using Kaggle
* Experiment tracking with W&B
* Model hosting using Hugging Face Hub
* Managing reproducible ML workflows
* Basic MLOps pipeline integration

---

## Author

Shrikrishna Tripathi

