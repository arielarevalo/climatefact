import os
import json
import argparse
import re


def numeric_sort_key(filename):
    """Extract numeric part from filename for proper sorting"""
    numbers = re.findall(r"\d+", filename)
    if numbers:
        return int(numbers[-1])
    return 0


def process_markdowns_simple(directory: str, output_path: str):
    all_paragraphs = []

    md_files = [f for f in os.listdir(directory) if f.endswith(".md")]
    md_files.sort(key=numeric_sort_key)

    for filename in md_files:
        file_path = os.path.join(directory, filename)
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
            for idx, para in enumerate(paragraphs):
                all_paragraphs.append(
                    {"filename": filename, "paragraph_index": idx, "paragraph": para}
                )

    with open(output_path, "w", encoding="utf-8") as out_f:
        for paragraph_obj in all_paragraphs:
            out_f.write(json.dumps(paragraph_obj, ensure_ascii=False) + "\n")

    print(f"Saved {len(all_paragraphs)} paragraphs to {output_path}")
    print(f"Files processed in order: {md_files[:5]}...")


def main():
    parser = argparse.ArgumentParser(
        description="Extract paragraphs from markdown files."
    )
    parser.add_argument("--input_dir", type=str, help="Directory with markdown files")
    parser.add_argument("--output_file", type=str, help="Output JSONL file path")
    args = parser.parse_args()
    process_markdowns_simple(args.input_dir, args.output_file)


if __name__ == "__main__":
    exit(main())


# pip install numpy==1.23.5
# pip install torch==1.12.1 
# pip install spacy==3.3.1 thinc==8.0.17
# pip install pydantic==1.8.2
# pip install allennlp==2.10.1
# pip install allennlp-models==2.10.1