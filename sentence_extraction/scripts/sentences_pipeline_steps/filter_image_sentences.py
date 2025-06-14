import json
import os
import re
import argparse
from typing import List

def filter_sentences(sentences: List[str]) -> List[str]:
    """
    Filter out sentences that contain Markdown image links.
    Example matched: ![alt text](image.png)
    """
    pattern = re.compile(
        r'!\s*\[\s*.*?\s*\]\s*\(\s*.*?\.(jpg|jpeg|png|gif|svg)\s*\)',
        re.IGNORECASE
    )
    return [s for s in sentences if not pattern.search(s)]

def process_paragraphs(paragraphs: List[dict]) -> List[dict]:
    """
    Add a 'filtered_sentences' field to each paragraph.
    """
    for p in paragraphs:
        if "sentences" in p:
            p["filtered_sentences"] = filter_sentences(p["sentences"])
    return paragraphs

def main(input_path: str, output_path: str):
    try:
        paragraphs = []
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                paragraphs.append(json.loads(line.strip()))
    except Exception as e:
        print(f"❌ Failed to read input file: {e}")
        return

    processed = process_paragraphs(paragraphs)

    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            for paragraph in processed:
                f.write(json.dumps(paragraph, ensure_ascii=False) + '\n')
        print(f"✅ Filtered sentences saved to: {output_path}")
    except Exception as e:
        print(f"❌ Failed to save output file: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter Markdown image links from sentences.")
    parser.add_argument('--input_file', type=str, help='Path to step1_sentences.jsonl')
    parser.add_argument('--output_file', type=str, help='Path to save filtered sentences JSONL')
    args = parser.parse_args()

    if os.path.exists(args.input_file):
        main(args.input_file, args.output_file)
    else:
        print(f"❌ Input not found: {args.input_file}")
