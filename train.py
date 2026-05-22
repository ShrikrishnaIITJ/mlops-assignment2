"""
Train and fine-tune DistilBERT for Goodreads genre classification.

This script:
1. Downloads Goodreads review datasets
2. Preprocesses and tokenizes text
3. Fine-tunes DistilBERT
4. Tracks experiments with W&B
5. Evaluates model performance
6. Saves classification report
7. Pushes trained model to Hugging Face Hub

Usage:
    python train.py
"""

import os
import json
import gzip
import random
import warnings
import requests
import pickle

import torch
import wandb
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    TrainingArguments,
    Trainer,
)

warnings.filterwarnings("ignore")

# =============================================================================
# Configuration
# =============================================================================

MODEL_NAME = "distilbert-base-cased"
HF_REPO = "ShrikrishnaIITJ/distilbert-goodreads-genres"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MAX_LENGTH = 128
HEAD = 10000
SAMPLE_SIZE = 2000
TOTAL_PER_GENRE = 1000
TRAIN_PER_GENRE = 800

OUTPUT_DIR = "./results"

GENRE_URLS = {
    "poetry":
        "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_poetry.json.gz",

    "children":
        "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_children.json.gz",

    "comics_graphic":
        "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_comics_graphic.json.gz",

    "fantasy_paranormal":
        "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_fantasy_paranormal.json.gz",

    "history_biography":
        "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_history_biography.json.gz",

    "mystery_thriller_crime":
        "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_mystery_thriller_crime.json.gz",

    "romance":
        "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_romance.json.gz",

    "young_adult":
        "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_young_adult.json.gz",
}

# =============================================================================
# Dataset Class
# =============================================================================

class GenreDataset(torch.utils.data.Dataset):
    """
    Custom PyTorch dataset for Hugging Face Trainer.
    """

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {
            key: torch.tensor(value[idx])
            for key, value in self.encodings.items()
        }

        item["labels"] = torch.tensor(self.labels[idx])

        return item

    def __len__(self):
        return len(self.labels)


# =============================================================================
# Helper Functions
# =============================================================================

def load_reviews(url, head=10000, sample_size=2000):
    """
    Download and sample Goodreads reviews from compressed JSON.
    """

    reviews = []

    response = requests.get(url, stream=True)

    with gzip.open(response.raw, "rt", encoding="utf-8") as file:

        for idx, line in enumerate(file):

            if idx >= head:
                break

            try:
                data = json.loads(line)

                if "review_text" in data:
                    reviews.append(data["review_text"])

            except Exception:
                continue

    reviews = random.sample(reviews, min(sample_size, len(reviews)))

    return reviews


def compute_metrics(pred):
    """
    Compute evaluation metrics.
    """

    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    accuracy = accuracy_score(labels, preds)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        preds,
        average="weighted"
    )

    return {
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }


# =============================================================================
# Main Training Pipeline
# =============================================================================

def main():

    # -------------------------------------------------------------------------
    # Initialize W&B
    # -------------------------------------------------------------------------

    wandb.init(
        project="huggingface",
        name="distilbert-run-1",
        config={
            "model": MODEL_NAME,
            "epochs": 3,
            "batch_size": 10,
            "learning_rate": 5e-5,
            "max_length": MAX_LENGTH,
            "dataset": "UCSD Goodreads",
            "platform": "Kaggle",
        }
    )

    print("=" * 80)
    print("Loading Goodreads Reviews")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # Load Dataset
    # -------------------------------------------------------------------------

    genre_reviews_dict = {}

    for genre, url in GENRE_URLS.items():

        print(f"Loading genre: {genre}")

        genre_reviews_dict[genre] = load_reviews(
            url,
            head=HEAD,
            sample_size=SAMPLE_SIZE,
        )

    with open("genre_reviews_dict.pickle", "wb") as f:
        pickle.dump(genre_reviews_dict, f)

    # -------------------------------------------------------------------------
    # Create Train/Test Split
    # -------------------------------------------------------------------------

    train_texts = []
    train_labels = []

    test_texts = []
    test_labels = []

    for genre, reviews in genre_reviews_dict.items():

        reviews = random.sample(
            reviews,
            TOTAL_PER_GENRE
        )

        # Training Data
        for review in reviews[:TRAIN_PER_GENRE]:
            train_texts.append(review)
            train_labels.append(genre)

        # Testing Data
        for review in reviews[TRAIN_PER_GENRE:]:
            test_texts.append(review)
            test_labels.append(genre)

    print(f"Training Samples : {len(train_texts)}")
    print(f"Testing Samples  : {len(test_texts)}")

    # -------------------------------------------------------------------------
    # Baseline Model (TF-IDF + Logistic Regression)
    # -------------------------------------------------------------------------

    print("\nRunning Baseline Logistic Regression...\n")

    vectorizer = TfidfVectorizer()

    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)

    baseline_model = LogisticRegression(max_iter=1000)

    baseline_model.fit(X_train, train_labels)

    baseline_predictions = baseline_model.predict(X_test)

    print(classification_report(test_labels, baseline_predictions))

    # -------------------------------------------------------------------------
    # Tokenization
    # -------------------------------------------------------------------------

    print("\nLoading Tokenizer...\n")

    tokenizer = DistilBertTokenizerFast.from_pretrained(
        MODEL_NAME
    )

    unique_labels = sorted(list(set(train_labels)))

    label2id = {
        label: idx
        for idx, label in enumerate(unique_labels)
    }

    id2label = {
        idx: label
        for label, idx in label2id.items()
    }

    train_encodings = tokenizer(
        train_texts,
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
    )

    test_encodings = tokenizer(
        test_texts,
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
    )

    train_labels_encoded = [
        label2id[label]
        for label in train_labels
    ]

    test_labels_encoded = [
        label2id[label]
        for label in test_labels
    ]

    train_dataset = GenreDataset(
        train_encodings,
        train_labels_encoded,
    )

    test_dataset = GenreDataset(
        test_encodings,
        test_labels_encoded,
    )

    # -------------------------------------------------------------------------
    # Load Pre-trained DistilBERT
    # -------------------------------------------------------------------------

    print("\nLoading DistilBERT Model...\n")

    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
    ).to(DEVICE)

    # -------------------------------------------------------------------------
    # Training Arguments
    # -------------------------------------------------------------------------

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,

        num_train_epochs=3,

        per_device_train_batch_size=10,
        per_device_eval_batch_size=16,

        learning_rate=5e-5,

        warmup_steps=100,

        weight_decay=0.01,

        logging_dir="./logs",

        logging_steps=100,

        eval_strategy="steps",

        save_strategy="steps",

        load_best_model_at_end=True,

        report_to="wandb",

        run_name="distilbert-run-1",
    )

    # -------------------------------------------------------------------------
    # Trainer
    # -------------------------------------------------------------------------

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    # -------------------------------------------------------------------------
    # Train Model
    # -------------------------------------------------------------------------

    print("\nStarting Fine-Tuning...\n")

    trainer.train()

    # -------------------------------------------------------------------------
    # Save Model
    # -------------------------------------------------------------------------

    trainer.save_model("distilbert-goodreads-model")

    tokenizer.save_pretrained("distilbert-goodreads-model")

    print("\nModel Saved Successfully.\n")

    # -------------------------------------------------------------------------
    # Evaluation
    # -------------------------------------------------------------------------

    print("\nEvaluating Model...\n")

    eval_results = trainer.evaluate()

    print(eval_results)

    # -------------------------------------------------------------------------
    # Predictions
    # -------------------------------------------------------------------------

    predictions = trainer.predict(test_dataset)

    predicted_ids = predictions.predictions.argmax(-1)

    predicted_labels = [
        id2label[idx]
        for idx in predicted_ids
    ]

    # -------------------------------------------------------------------------
    # Classification Report
    # -------------------------------------------------------------------------

    print("\nClassification Report\n")

    print(
        classification_report(
            test_labels,
            predicted_labels
        )
    )

    report = classification_report(
        test_labels,
        predicted_labels,
        output_dict=True,
    )

    with open("eval_report.json", "w") as f:
        json.dump(report, f, indent=2)

    # -------------------------------------------------------------------------
    # Log Final Metrics to W&B
    # -------------------------------------------------------------------------

    wandb.log({
        "final/loss": eval_results["eval_loss"],
        "final/accuracy": eval_results["eval_accuracy"],
        "final/f1": eval_results["eval_f1"],
    })

    # -------------------------------------------------------------------------
    # Upload Artifact to W&B
    # -------------------------------------------------------------------------

    artifact = wandb.Artifact(
        "eval-report",
        type="evaluation"
    )

    artifact.add_file("eval_report.json")

    wandb.log_artifact(artifact)

    # -------------------------------------------------------------------------
    # Push Model to Hugging Face Hub
    # -------------------------------------------------------------------------

    hf_token = os.environ.get("HF_TOKEN")

    if hf_token:

        from huggingface_hub import login

        login(token=hf_token)

        model.push_to_hub(HF_REPO)

        tokenizer.push_to_hub(HF_REPO)

        wandb.run.summary["huggingface_model"] = (
            f"https://huggingface.co/{HF_REPO}"
        )

        print(f"\nModel pushed to Hugging Face Hub:")
        print(f"https://huggingface.co/{HF_REPO}")

    # -------------------------------------------------------------------------
    # Finish W&B
    # -------------------------------------------------------------------------

    wandb.finish()

    print("\nTraining Pipeline Completed Successfully.")


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    main()