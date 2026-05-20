# MLOps Assignment 2

Text classification project using DistilBERT, Hugging Face Transformers, Kaggle GPU, and Weights & Biases (W&B).

---

## Model

* Model: `distilbert-base-cased`
* Task: Multi-class text classification
* Classes:

  * children
  * comics_graphic
  * fantasy_paranormal
  * history_biography
  * mystery_thriller_crime
  * poetry
  * romance
  * young_adult

---

## Training Setup

| Parameter     | Value    |
| ------------- | -------- |
| Platform      | Kaggle   |
| GPU           | Tesla T4 |
| Epochs        | 3        |
| Batch Size    | 8        |
| Learning Rate | 3e-5     |

---

## Results

| Metric            | Score  |
| ----------------- | ------ |
| Accuracy          | 0.5981 |
| Weighted F1 Score | 0.6023 |
| Precision         | 0.6105 |
| Recall            | 0.5981 |
| Eval Loss         | 2.3864 |

---

## Classification Summary

| Genre                  | F1 Score |
| ---------------------- | -------- |
| children               | 0.65     |
| comics_graphic         | 0.82     |
| fantasy_paranormal     | 0.41     |
| history_biography      | 0.60     |
| mystery_thriller_crime | 0.54     |
| poetry                 | 0.80     |
| romance                | 0.57     |
| young_adult            | 0.43     |

---

## Tools & Libraries

* Python
* Hugging Face Transformers
* PyTorch
* Scikit-learn
* W&B
* Kaggle Notebook

---

## Project Links

### Hugging Face Model

https://huggingface.co/ShrikrishnaIITJ/distilbert-goodreads-genres

### W&B Dashboard

https://wandb.ai/g25ait2103-indian-institute-of-technology-jodhpur/huggingface

### Kaggle Notebook

PASTE_YOUR_KAGGLE_NOTEBOOK_LINK

### GitHub Repository

PASTE_YOUR_GITHUB_REPOSITORY_LINK

---

## Installation

```bash id="yq7yx0"
pip install transformers datasets evaluate accelerate wandb huggingface_hub scikit-learn torch
```

---

## Running the Project

1. Open the notebook in Kaggle or Colab.
2. Enable GPU acceleration.
3. Install dependencies.
4. Run all notebook cells sequentially.
5. Training metrics are tracked using W&B.
6. Final model is pushed to Hugging Face Hub.

---

## Author

Shrikrishna Tripathi
