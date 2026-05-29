"""
Data Parser — reads medical data from .txt and .pdf files.
Supports the format:
    DISEASE: <name>
    MEDICINES: <comma-separated>
    SIDE_EFFECTS: <comma-separated>
    ALTERNATIVES: <comma-separated>
"""

import os
import re
import json


def parse_text_file(filepath):
    """Parse a structured .txt medical data file."""
    records = []
    current = {}

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                if current.get("disease"):
                    records.append(current)
                    current = {}
                continue

            if line.startswith("DISEASE:"):
                current["disease"] = line.replace("DISEASE:", "").strip().lower()
            elif line.startswith("MEDICINES:"):
                raw = line.replace("MEDICINES:", "").strip()
                current["medicines"] = [m.strip().lower() for m in raw.split(",")]
            elif line.startswith("SIDE_EFFECTS:"):
                raw = line.replace("SIDE_EFFECTS:", "").strip()
                current["side_effects"] = [s.strip().lower() for s in raw.split(",")]
            elif line.startswith("ALTERNATIVES:"):
                raw = line.replace("ALTERNATIVES:", "").strip()
                current["alternatives"] = [a.strip().lower() for a in raw.split(",")]

    if current.get("disease"):
        records.append(current)

    return records


def parse_pdf_file(filepath):
    """Parse a .pdf medical data file using pdfplumber."""
    try:
        import pdfplumber
    except ImportError:
        raise ImportError("Install pdfplumber: pip install pdfplumber")

    text_lines = []
    with pdfplumber.open(filepath) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                text_lines.extend(text.split("\n"))

    # Write to temp text and reuse text parser
    tmp_path = filepath.replace(".pdf", "_temp.txt")
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write("\n".join(text_lines))

    records = parse_text_file(tmp_path)
    os.remove(tmp_path)
    return records


def load_all_data(data_dir):
    """Load all .txt and .pdf files from a directory."""
    all_records = []
    for fname in os.listdir(data_dir):
        fpath = os.path.join(data_dir, fname)
        if fname.endswith(".txt"):
            records = parse_text_file(fpath)
            all_records.extend(records)
            print(f"  Loaded {len(records)} records from {fname}")
        elif fname.endswith(".pdf"):
            try:
                records = parse_pdf_file(fpath)
                all_records.extend(records)
                print(f"  Loaded {len(records)} records from {fname}")
            except Exception as e:
                print(f"  Could not parse {fname}: {e}")

    print(f"\nTotal records loaded: {len(all_records)}")
    return all_records


def save_processed_data(records, output_path):
    """Save parsed records to JSON for inspection."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    print(f"Saved processed data -> {output_path}")


if __name__ == "__main__":
    records = load_all_data("../data")
    save_processed_data(records, "../outputs/parsed_data.json")
    print("\nSample record:")
    print(json.dumps(records[0], indent=2))
