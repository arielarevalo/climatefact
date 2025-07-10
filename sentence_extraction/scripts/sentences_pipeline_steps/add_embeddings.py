import json
import os
import re
import uuid
import requests
import argparse
from typing import List, Dict, Any

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Warning: python-dotenv not installed. Install it with: pip install python-dotenv")

def extract_page_number(filename: str) -> int:
    match = re.search(r'_(\d+)\.', filename)
    return int(match.group(1)) + 1 if match else 1

def generate_unique_id() -> str:
    return str(uuid.uuid4())

def get_embedding(text: str, api_key: str, endpoint: str) -> List[float]:
    headers = {
        "Content-Type": "application/json",
        "api-key": api_key
    }
    data = {"input": text}

    try:
        response = requests.post(endpoint, headers=headers, json=data)
        response.raise_for_status()
        return response.json()["data"][0]["embedding"]
    except requests.exceptions.RequestException as e:
        print(f"⚠️ Error getting embedding for: {text[:50]}...")
        print(f"   Reason: {e}")
        return []

def process_paragraph(paragraph: dict, api_key: str, endpoint: str) -> List[dict]:
    results = []
    filename = paragraph.get("filename", "")
    page_number = extract_page_number(filename)

    for sentence in paragraph.get("filtered_sentences", []):
        print(f"Embedding: {sentence[:50]}...")
        embedding = get_embedding(sentence, api_key, endpoint)
        if embedding:
            results.append({
                "id": generate_unique_id(),
                "embedding": embedding,
                "text": sentence,
                "source": {
                    "name": filename,
                    "page": page_number
                }
            })
        else:
            print(f"⚠️ Skipping sentence: {sentence[:50]}...")
    return results

def main(input_path: str, output_path: str, api_key: str, endpoint: str):
    try:
        paragraphs = []
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                paragraphs.append(json.loads(line.strip()))
    except Exception as e:
        print(f"❌ Error reading input: {e}")
        return

    sentence_count = 0
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            for i, paragraph in enumerate(paragraphs):
                if "filtered_sentences" not in paragraph or "filename" not in paragraph:
                    print(f"Skipping paragraph {i+1}: missing fields")
                    continue
                sentence_objs = process_paragraph(paragraph, api_key, endpoint)
                for sentence_obj in sentence_objs:
                    f.write(json.dumps(sentence_obj, ensure_ascii=False) + '\n')
                    sentence_count += 1
        
        print(f"\n✅ Step complete: {sentence_count} embeddings saved to {output_path}")
        
        # Show sample by reading first line
        if sentence_count > 0:
            with open(output_path, 'r', encoding='utf-8') as f:
                sample = json.loads(f.readline().strip())
                if len(sample["embedding"]) > 5:
                    sample["embedding"] = sample["embedding"][:3] + ["..."] + [f"({len(sample['embedding'])} total)"]
                print("\n🔎 Sample sentence object:\n" + json.dumps(sample, indent=2))
    except Exception as e:
        print(f"❌ Error writing output: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate sentence embeddings from filtered content.")
    parser.add_argument('--input_file', type=str, help='Path to step2_filtered_sentences.jsonl')
    parser.add_argument('--output_file', type=str, help='Path to output embeddings JSONL')
    args = parser.parse_args()

    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")

    if not api_key or not endpoint:
        raise ValueError("❌ AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT must be set in the environment or .env file")

    if os.path.exists(args.input_file):
        main(args.input_file, args.output_file, api_key, endpoint)
    else:
        print(f"❌ Input file not found: {args.input_file}")
