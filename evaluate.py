"""
Evaluate the fine-tuned DistilBERT Goodreads genre classifier.

This script:
1. Loads saved Goodreads review data
2. Recreates the test dataset
3. Loads the trained DistilBERT model
4. Evaluates classification performance
5. Generates classification reports
6. Creates confusion matrix visualizations

Usage:
    python evaluate.py
"""

import pickle
import random
import warnings
from collections import defaultdict

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
)

from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
)

warnings.filterwarnings("ignore")

# =============================================================================
# Configuration
# =============================================================================

MODEL_NAME = "distilbert-base-cased"

MODEL_DIRECTORY = "distilbert-goodreads-model"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MAX_LENGTH = 128

TOTAL_PER_GENRE = 1000

TRAIN_PER_GENRE = 800


# =============================================================================
# Dataset Class
# =============================================================================

class GenreDataset(torch.utils.data.Dataset):

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
# Metrics Function
# =============================================================================

def compute_metrics(pred):

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
# Main Evaluation Pipeline
# =============================================================================

def main():

    print("=" * 80)
    print("Loading Saved Goodreads Dataset")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # Load Saved Dataset
    # -------------------------------------------------------------------------

    with open("genre_reviews_dict.pickle", "rb") as f:
        genre_reviews_dict = pickle.load(f)

    # -------------------------------------------------------------------------
    # Reconstruct Train/Test Split
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

        # Train split
        for review in reviews[:TRAIN_PER_GENRE]:
            train_texts.append(review)
            train_labels.append(genre)

        # Test split
        for review in reviews[TRAIN_PER_GENRE:]:
            test_texts.append(review)
            test_labels.append(genre)

    # -------------------------------------------------------------------------
    # Label Mapping
    # -------------------------------------------------------------------------

    unique_labels = sorted(list(set(train_labels)))

    label2id = {
        label: idx
        for idx, label in enumerate(unique_labels)
    }

    id2label = {
        idx: label
        for label, idx in label2id.items()
    }

    # -------------------------------------------------------------------------
    # Tokenization
    # -------------------------------------------------------------------------

    print("\nLoading Tokenizer...\n")

    tokenizer = DistilBertTokenizerFast.from_pretrained(
        MODEL_NAME
    )

    test_encodings = tokenizer(
        test_texts,
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
    )

    test_labels_encoded = [
        label2id[label]
        for label in test_labels
    ]

    test_dataset = GenreDataset(
        test_encodings,
        test_labels_encoded
    )

    # -------------------------------------------------------------------------
    # Load Fine-Tuned Model
    # -------------------------------------------------------------------------

    print(f"\nLoading Model from: {MODEL_DIRECTORY}\n")

    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_DIRECTORY
    ).to(DEVICE)

    # -------------------------------------------------------------------------
    # Trainer
    # -------------------------------------------------------------------------

    training_args = TrainingArguments(
        output_dir="./evaluation_results",
        per_device_eval_batch_size=16,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
    )

    # -------------------------------------------------------------------------
    # Evaluate Model
    # -------------------------------------------------------------------------

    print("\nEvaluating Model...\n")

    evaluation_results = trainer.evaluate(test_dataset)

    print(evaluation_results)

    # -------------------------------------------------------------------------
    # Generate Predictions
    # -------------------------------------------------------------------------

    prediction_results = trainer.predict(test_dataset)

    predicted_ids = prediction_results.predictions.argmax(-1)

    predicted_labels = [
        id2label[idx]
        for idx in predicted_ids
    ]

    # -------------------------------------------------------------------------
    # Classification Report
    # -------------------------------------------------------------------------

    print("\nClassification Report\n")

    report_text = classification_report(
        test_labels,
        predicted_labels
    )

    print(report_text)

    # Save report
    with open("classification_report.txt", "w") as f:
        f.write(report_text)

    print("\nSaved classification_report.txt")

    # -------------------------------------------------------------------------
    # Confusion Matrix
    # -------------------------------------------------------------------------

    print("\nGenerating Confusion Matrix...\n")

    confusion_dict = defaultdict(int)

    for true_label, predicted_label in zip(
        test_labels,
        predicted_labels
    ):

        confusion_dict[(true_label, predicted_label)] += 1

    plot_rows = []

    for (true_genre, predicted_genre), count in confusion_dict.items():

        plot_rows.append({
            "True Genre": true_genre,
            "Predicted Genre": predicted_genre,
            "Count": count,
        })

    confusion_df = pd.DataFrame(plot_rows)

    confusion_matrix_df = confusion_df.pivot_table(
        index="True Genre",
        columns="Predicted Genre",
        values="Count",
        fill_value=0,
    )

    plt.figure(figsize=(10, 8))

    sns.set(style="ticks", font_scale=1.1)

    sns.heatmap(
        confusion_matrix_df,
        linewidths=1,
        cmap="Purples",
        annot=True,
        fmt=".0f"
    )

    plt.title("Confusion Matrix")

    plt.xticks(rotation=45, ha="right")

    plt.tight_layout()

    plt.savefig(
        "confusion_matrix.png",
        dpi=150
    )

    plt.close()

    print("Saved confusion_matrix.png")

    # -------------------------------------------------------------------------
    # Misclassification Matrix
    # -------------------------------------------------------------------------

    print("\nGenerating Misclassification Matrix...\n")

    misclassification_dict = defaultdict(int)

    for true_label, predicted_label in zip(
        test_labels,
        predicted_labels
    ):

        if true_label != predicted_label:

            misclassification_dict[
                (true_label, predicted_label)
            ] += 1

    plot_rows = []

    for (true_genre, predicted_genre), count in misclassification_dict.items():

        plot_rows.append({
            "True Genre": true_genre,
            "Predicted Genre": predicted_genre,
            "Count": count,
        })

    misclassification_df = pd.DataFrame(plot_rows)

    misclassification_matrix_df = misclassification_df.pivot_table(
        index="True Genre",
        columns="Predicted Genre",
        values="Count",
        fill_value=0,
    )

    plt.figure(figsize=(10, 8))

    sns.set(style="ticks", font_scale=1.1)

    sns.heatmap(
        misclassification_matrix_df,
        linewidths=1,
        cmap="Purples",
        annot=True,
        fmt=".0f"
    )

    plt.title("Misclassification Matrix")

    plt.xticks(rotation=45, ha="right")

    plt.tight_layout()

    plt.savefig(
        "misclassification_matrix.png",
        dpi=150
    )

    plt.close()

    print("Saved misclassification_matrix.png")

    print("\nEvaluation Completed Successfully.")


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    main()