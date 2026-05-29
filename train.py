"""
train.py — Run this script to train MedAI on your data.

Usage:
    python train.py

Steps:
    1. Reads all .txt / .pdf files from data/
    2. Parses disease → medicines / side_effects / alternatives
    3. Trains the TensorFlow/Keras model
    4. Saves the trained model to models/
    5. Runs sample predictions
"""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(__file__))

from utils.data_parser import load_all_data, save_processed_data
from models.medai_model import MedAIModel


DATA_DIR    = os.path.join(os.path.dirname(__file__), "data")
OUTPUT_DIR  = os.path.join(os.path.dirname(__file__), "outputs")
MODEL_DIR   = os.path.join(os.path.dirname(__file__), "models", "saved")


def print_prediction(result):
    print(f"\n{'═'*55}")
    print(f"  Disease : {result['disease'].upper()}")
    print(f"{'═'*55}")

    print("\n  💊 Recommended Medicines:")
    for name, conf in result["medicines"]:
        bar = "█" * int(conf * 20)
        print(f"     {name:<25} {bar:<20} {conf*100:.1f}%")

    print("\n  ⚠️  Side Effects:")
    for name, conf in result["side_effects"]:
        print(f"     • {name}")

    print("\n  🔄 Alternative Medicines:")
    for name, conf in result["alternatives"]:
        print(f"     • {name}")
    print()


def main():
    print("=" * 55)
    print("  MedAI — Custom Medical NLP Model Trainer")
    print("=" * 55)

    # ── Step 1: Load data ──────────────────────────────────
    print(f"\n[1/4] Loading data from: {DATA_DIR}")
    records = load_all_data(DATA_DIR)

    if not records:
        print("ERROR: No records found. Add .txt or .pdf files to the data/ folder.")
        print("       Format each entry as:\n")
        print("       DISEASE: <name>")
        print("       MEDICINES: <med1>, <med2>, ...")
        print("       SIDE_EFFECTS: <effect1>, <effect2>, ...")
        print("       ALTERNATIVES: <alt1>, <alt2>, ...")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_processed_data(records, os.path.join(OUTPUT_DIR, "parsed_data.json"))

    # ── Step 2: Train ──────────────────────────────────────
    print(f"\n[2/4] Training model on {len(records)} records ...")
    model = MedAIModel()
    history = model.train(records)

    # ── Step 3: Save model ─────────────────────────────────
    print(f"\n[3/4] Saving model to: {MODEL_DIR}")
    model.save(MODEL_DIR)

    # Save training metrics
    metrics = {
        "epochs_run": len(history.history["loss"]),
        "final_loss":            history.history["loss"][-1],
        "final_medicines_acc":   history.history.get("medicines_accuracy", [None])[-1],
        "final_side_effects_acc":history.history.get("side_effects_accuracy", [None])[-1],
        "final_alternatives_acc":history.history.get("alternatives_accuracy", [None])[-1],
    }
    metrics_path = os.path.join(OUTPUT_DIR, "training_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Training metrics saved → {metrics_path}")

    # ── Step 4: Sample predictions ─────────────────────────
    print(f"\n[4/4] Sample Predictions")
    test_diseases = ["Diabetes Type 2", "Hypertension", "Asthma", "Malaria"]
    for disease in test_diseases:
        result = model.predict(disease)
        print_prediction(result)

    print("─" * 55)
    print("  ✅  Training complete! Model is ready to use.")
    print("  Run  predict.py  to query any disease.")
    print("─" * 55)


if __name__ == "__main__":
    main()
