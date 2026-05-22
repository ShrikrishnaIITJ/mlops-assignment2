"""
Run inference using the fine-tuned DistilBERT Goodreads genre classifier.

This script loads the trained model from Hugging Face Hub and predicts
the genre of a Goodreads review text.

Usage:
    python inference.py --text "This book was a thrilling mystery with unexpected twists."

Interactive Mode:
    python inference.py
"""

import argparse
import torch

from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
)

# =============================================================================
# Configuration
# =============================================================================

MODEL_NAME = "ShrikrishnaIITJ/distilbert-goodreads-genres"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MAX_LENGTH = 128


# =============================================================================
# Prediction Function
# =============================================================================

def predict_genre(text, model, tokenizer, id2label):

    # Tokenize input text
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
    ).to(DEVICE)

    # Disable gradient calculation for inference
    with torch.no_grad():
        outputs = model(**inputs)

    # Convert logits to probabilities
    probabilities = torch.nn.functional.softmax(
        outputs.logits,
        dim=-1
    )

    # Get highest probability class
    predicted_index = probabilities.argmax(-1).item()

    predicted_label = id2label[predicted_index]

    confidence = probabilities[0][predicted_index].item()

    return predicted_label, confidence


# =============================================================================
# Main
# =============================================================================

def main():

    parser = argparse.ArgumentParser(
        description="Predict Goodreads review genre using DistilBERT."
    )

    parser.add_argument(
        "--text",
        type=str,
        help="Review text to classify."
    )

    parser.add_argument(
        "--model",
        type=str,
        default=MODEL_NAME,
        help="Hugging Face model repository or local model path."
    )

    args = parser.parse_args()

    print("=" * 80)
    print(f"Loading model from: {args.model}")
    print("=" * 80)

    # Load tokenizer
    tokenizer = DistilBertTokenizerFast.from_pretrained(
        args.model
    )

    # Load trained model
    model = DistilBertForSequenceClassification.from_pretrained(
        args.model
    ).to(DEVICE)

    model.eval()

    # Read label mapping from model config
    id2label = model.config.id2label

    # -------------------------------------------------------------------------
    # Single Prediction Mode
    # -------------------------------------------------------------------------

    if args.text:

        genre, confidence = predict_genre(
            args.text,
            model,
            tokenizer,
            id2label
        )

        print("\nPrediction Result")
        print("-" * 40)

        print(f"Input Review : {args.text}")
        print(f"Predicted Genre : {genre}")
        print(f"Confidence : {confidence:.2%}")

    # -------------------------------------------------------------------------
    # Interactive Mode
    # -------------------------------------------------------------------------

    else:

        print("\nInteractive Mode")
        print("Type a review and press Enter.")
        print("Type 'quit' to exit.\n")

        while True:

            text = input("Review > ").strip()

            if text.lower() in ["quit", "exit", "q"]:
                print("Exiting...")
                break

            if not text:
                continue

            genre, confidence = predict_genre(
                text,
                model,
                tokenizer,
                id2label
            )

            print(f"\nPredicted Genre : {genre}")
            print(f"Confidence      : {confidence:.2%}\n")


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    main()