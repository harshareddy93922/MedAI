"""
predict.py — Query the trained MedAI model interactively.

Usage:
    python predict.py
    python predict.py --disease "Diabetes"
    python predict.py --disease "Hypertension" --threshold 0.2 --top_k 5
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(__file__))

from models.medai_model import MedAIModel

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models", "saved")


def print_result(result):
    print(f"\n{'═'*55}")
    print(f"  DISEASE : {result['disease'].upper()}")
    print(f"{'═'*55}")

    print("\n  💊 Recommended Medicines:")
    for i, (name, conf) in enumerate(result["medicines"], 1):
        bar = "█" * int(conf * 20)
        print(f"  {i}. {name:<28} {conf*100:.1f}%")

    print("\n  ⚠️  Possible Side Effects:")
    for name, conf in result["side_effects"]:
        print(f"     • {name}")

    print("\n  🔄 Alternative Medicines:")
    for name, conf in result["alternatives"]:
        print(f"     • {name}")
    print()


def main():
    parser = argparse.ArgumentParser(description="MedAI Prediction CLI")
    parser.add_argument("--disease",   type=str,   default=None, help="Disease name to query")
    parser.add_argument("--threshold", type=float, default=0.3,  help="Confidence threshold (0-1)")
    parser.add_argument("--top_k",     type=int,   default=5,    help="Max results per category")
    args = parser.parse_args()

    # Load trained model
    print("\nLoading MedAI model ...")
    if not os.path.exists(MODEL_DIR):
        print(f"ERROR: No trained model found at {MODEL_DIR}")
        print("       Please run  python train.py  first.")
        sys.exit(1)

    model = MedAIModel()
    model.load(MODEL_DIR)
    print("Model loaded successfully!\n")

    # Single query mode
    if args.disease:
        result = model.predict(args.disease, threshold=args.threshold, top_k=args.top_k)
        print_result(result)
        return

    # Interactive mode
    print("=" * 55)
    print("  MedAI — Disease to Medicine Predictor")
    print("  Type a disease name and press Enter.")
    print("  Type 'quit' to exit.")
    print("=" * 55)

    while True:
        try:
            disease = input("\n  Enter disease name: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Goodbye!")
            break

        if disease.lower() in ("quit", "exit", "q"):
            print("  Goodbye!")
            break

        if not disease:
            continue

        result = model.predict(disease, threshold=args.threshold, top_k=args.top_k)
        print_result(result)


if __name__ == "__main__":
    main()
