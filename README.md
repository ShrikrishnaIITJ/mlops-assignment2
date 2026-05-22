# MLOps Assignment 2 — DistilBERT Goodreads Genre Classifier

**Student:** Shrikrishna Tripathi
**Course:** MLOps, IIT Jodhpur
**Roll Number:** G25AIT2103

---

## Overview

This project fine-tunes a **DistilBERT** (`distilbert-base-cased`) model to classify Goodreads book reviews into 8 genres: *poetry, children, comics & graphic, fantasy & paranormal, history & biography, mystery/thriller/crime, romance,* and *young adult*. The dataset comes from the [UCSD Book Graph](https://mengtingwan.github.io/data/goodreads.html). For each genre, 10,000 reviews are streamed and 2,000 are randomly sampled; then 1,000 per genre are used (800 train / 200 test), giving 6,400 training and 1,600 test samples. Training was performed on a **Kaggle Notebook** with GPU acceleration enabled. Experiment tracking was managed using **Weights & Biases (W&B)** (`report_to="wandb"`), and the trained model was uploaded publicly to **Hugging Face Hub**.

---

## Setup Instructions

The primary training was done in the Kaggle Notebook (linked below). The Python scripts below are extracted from the notebook for standalone use.

```bash
# 1. Clone the repository
git clone https://github.com/ShrikrishnaIITJ/mlops-assignment2

# 2. Open the project folder
cd mlops-assignment2

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set environment variables for API keys
export WANDB_API_KEY="your_wandb_api_key"
export HF_TOKEN="your_huggingface_token"

# 5. Run training
python train.py

# 6. Run evaluation
python evaluate.py

# 7. Run inference
python inference.py --text "A thrilling mystery with unexpected twists and a dark ending."
```

---

## Training Platform

Training was performed on **Kaggle Notebook** with GPU accelerator.
Kaggle Secrets were used to securely store `WANDB_API_KEY` and `HF_TOKEN` using `kaggle_secrets.UserSecretsClient`.

**Kaggle Notebook:**
https://www.kaggle.com/code/shrikrishnatripathi/mlops-assignment2-g25ait2103

---

## Results

These results are from the notebook's `trainer.evaluate()` output:

| Metric    | Value  |
| --------- | ------ |
| Accuracy  | 0.6000 |
| F1 Score  | 0.5983 |
| Precision | 0.5989 |
| Recall    | 0.6000 |
| Eval Loss | 2.4034 |

---

## Per-class Performance (DistilBERT)

| Genre                  | Precision | Recall | F1-Score |
| ---------------------- | --------- | ------ | -------- |
| children               | 0.69      | 0.68   | 0.68     |
| comics_graphic         | 0.83      | 0.78   | 0.80     |
| fantasy_paranormal     | 0.44      | 0.47   | 0.45     |
| history_biography      | 0.57      | 0.57   | 0.57     |
| mystery_thriller_crime | 0.52      | 0.60   | 0.56     |
| poetry                 | 0.72      | 0.78   | 0.75     |
| romance                | 0.62      | 0.59   | 0.61     |
| young_adult            | 0.41      | 0.34   | 0.37     |

---

## Links

| Resource           | Link                                                                                         |
| ------------------ | -------------------------------------------------------------------------------------------- |
| Hugging Face Model | https://huggingface.co/ShrikrishnaIITJ/distilbert-goodreads-genres                           |
| W&B Dashboard      | https://wandb.ai/g25ait2103-indian-institute-of-technology-jodhpur/huggingface/runs/4905vq6r |
| Kaggle Notebook    | https://www.kaggle.com/code/shrikrishnatripathi/mlops-assignment2-g25ait2103                 |
| GitHub Repository  | https://github.com/ShrikrishnaIITJ/mlops-assignment2                                         |

---

## Project Structure

```text
├── train.py
├── evaluate.py
├── inference.py
├── requirements.txt
├── README.md
├── notebook1ee14113a1.ipynb
└── MLOps_Assignment2_Report.pdf
```

---

## Tools & Libraries

* Python
* PyTorch
* Hugging Face Transformers
* Datasets
* Scikit-learn
* Weights & Biases (W&B)
* Kaggle Notebook
* Hugging Face Hub

---

## Conclusion

This project successfully demonstrated a complete MLOps workflow using modern NLP and cloud-based ML tools. The DistilBERT model was trained and evaluated using Kaggle GPU infrastructure while experiment tracking was managed using W&B. The trained model was uploaded to Hugging Face Hub and the entire workflow was version-controlled using GitHub for reproducibility and accessibility.
