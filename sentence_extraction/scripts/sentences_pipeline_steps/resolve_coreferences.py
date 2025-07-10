import os
import json
import argparse
import re
from typing import Dict, List

from allennlp.predictors.predictor import Predictor
from allennlp.data.tokenizers import SpacyTokenizer
from nltk.tokenize.treebank import TreebankWordDetokenizer
from tqdm import tqdm


def fix_hyphen_spacing(text: str) -> str:
    """
    Joins hyphenated word pairs split by spaces.
    Handles Unicode-aware word boundaries.
    """
    return re.sub(r"\b(\w+)\s*-\s*(\w+)\b", r"\1-\2", text)


class CorefResolver:
    """Base class for coreference resolvers."""

    def resolve(self, text: str) -> str:
        raise NotImplementedError


class AllenNLPCorefResolver(CorefResolver):
    """Coreference resolver using AllenNLP SpanBERT model."""

    def __init__(
        self,
        model_url: str = "https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2021.03.10.tar.gz",
    ):
        print("Initializing AllenNLP Coref Resolver...")
        print("Warning: This will download a large model (~1.3GB) if not already cached. This may take several minutes...")
        self.predictor = Predictor.from_path(model_url)
        self.tokenizer = SpacyTokenizer()

    def resolve(self, text: str) -> str:
        result = self.predictor.predict(document=text)
        document_tokens: List[str] = result["document"]
        clusters: List[List[List[int]]] = result.get("clusters", [])

        token_replacements: Dict[int, str] = {}
        for cluster in clusters:
            rep_start, rep_end = cluster[0]
            rep_span_tokens = document_tokens[rep_start : rep_end + 1]
            rep_text = TreebankWordDetokenizer().detokenize(rep_span_tokens)
            for mention in cluster[1:]:
                m_start, m_end = mention
                token_replacements[m_start] = (m_end, rep_text)

        resolved_tokens: List[str] = []
        i = 0
        while i < len(document_tokens):
            if i in token_replacements:
                end, rep_text = token_replacements[i]
                rep_tokens = [t.text for t in self.tokenizer.tokenize(rep_text)]
                resolved_tokens.extend(rep_tokens)
                i = end + 1
            else:
                resolved_tokens.append(document_tokens[i])
                i += 1

        resolved_str = TreebankWordDetokenizer().detokenize(resolved_tokens)
        return fix_hyphen_spacing(resolved_str)


def resolve_paragraph(resolver: CorefResolver, para: Dict) -> Dict:
    """Resolve a single paragraph with error handling."""
    try:
        resolved_text = resolver.resolve(para["paragraph"])
    except Exception as e:
        print(f"Warning resolving paragraph {para['paragraph_index']}: {e}")
        resolved_text = para["paragraph"]
    return {
        "filename": para["filename"],
        "paragraph_index": para["paragraph_index"],
        "original_paragraph": para["paragraph"],
        "coref_resolved": resolved_text,
    }


def resolve_coreferences(input_file: str, output_file: str):
    """Resolve coreferences in paragraphs using sequential processing."""
    paragraphs = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            paragraphs.append(json.loads(line.strip()))

    resolver = AllenNLPCorefResolver()
    total = len(paragraphs)

    print(f"Resolving {total} paragraphs using sequential processing...")

    with open(output_file, "w", encoding="utf-8") as out_f:
        for para in tqdm(paragraphs, desc="Resolving paragraphs"):
            resolved_para = resolve_paragraph(resolver, para)
            out_f.write(json.dumps(resolved_para, ensure_ascii=False) + "\n")
    
    print(f"Saved {len(paragraphs)} resolved paragraphs to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Resolve coreferences using AllenNLP.")
    parser.add_argument("--input_file", type=str)
    parser.add_argument("--output_file", type=str)
    args = parser.parse_args()

    if os.path.exists(args.input_file):
        resolve_coreferences(args.input_file, args.output_file)
    else:
        print(f"Input not found: {args.input_file}")


if __name__ == "__main__":
    exit(main())
