import json
import os
import re
import argparse
from typing import List

def split_into_sentences(text: str) -> List[str]:
    """
    Split paragraph text into a list of sentences.
    """
    text = re.sub(r'\s+', ' ', text.strip())
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    return [s.strip() for s in sentences if s.strip()]

def process_paragraphs(paragraphs: List[dict]) -> List[dict]:
    """
    Add a 'sentences' list to each paragraph using its coref_resolved field.
    """
    for p in paragraphs:
        if "coref_resolved" in p:
            p["sentences"] = split_into_sentences(p["coref_resolved"])
    return paragraphs

def main(input_path: str, output_path: str):
    try:
        paragraphs = []
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                paragraphs.append(json.loads(line.strip()))
    except Exception as e:
        print(f"Error reading input file: {e}")
        return

    enriched = process_paragraphs(paragraphs)

    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            for paragraph in enriched:
                f.write(json.dumps(paragraph, ensure_ascii=False) + '\n')
        print(f"✅ Sentences extracted and saved to {output_path}")
    except Exception as e:
        print(f"Error saving output file: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split coref-resolved paragraphs into sentences.")
    parser.add_argument('--input_file', type=str, help='Path to coref-resolved JSONL file')
    parser.add_argument('--output_file', type=str, help='Path to output JSONL file with sentence lists')
    args = parser.parse_args()

    if os.path.exists(args.input_file):
        main(args.input_file, args.output_file)
    else:
        print(f"❌ Input not found: {args.input_file}")
