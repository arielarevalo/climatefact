#!/usr/bin/env python3
"""
Hybrid Concept Extraction System

This script implements a hybrid approach combining regex-based and NER-based concept extraction
from passages in the knowledge building system. It processes passages.jsonl and extracts
concepts using multiple methods for comprehensive coverage.

Authors: Climate Concept Extraction Team
Date: 2024
"""

import argparse
import json
import logging
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

# Add parent directory to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import configuration
from config import (
    REGEX_MAP,
    PATTERN_SCHEMA,
    DOMAIN_SPECIFIC_PATTERNS,
    ENTITY_RULER_PATTERNS,
    SPACY_MODEL,
    TRANSFORMERS_NER_MODEL,
    NLTK_STOPWORDS_LANG,
    LOG_FORMAT,
    LOG_LEVEL,
    BATCH_SIZE,
    PROGRESS_INTERVAL,
    CONFIDENCE_THRESHOLDS,
    EXTRACTION_PRIORITY
)

# Optional NLP libraries with graceful fallback
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

try:
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.tag import pos_tag
    from nltk.chunk import ne_chunk
    from nltk.corpus import stopwords
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

try:
    import transformers
    TRANSFORMERS_AVAILABLE = True
    nltk.download('punkt_tab')
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# Set up logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format=LOG_FORMAT
)
logger = logging.getLogger(__name__)


# ===================================================================
# HYBRID CONCEPT EXTRACTION - NER COMPONENT
# ===================================================================

class ConceptExtractor:
    """Hybrid concept extractor combining regex, NER, and EntityRuler approaches"""
    
    def __init__(self):
        self.nlp = None
        self.ner_pipeline = None
        self.stopwords = set()
        self.entity_ruler = None
        self._setup_models()
    
    def _setup_models(self):
        """Initialize NLP models and resources"""
        try:
            # Load spaCy model
            if SPACY_AVAILABLE:
                self.nlp = spacy.load(SPACY_MODEL)
                logger.info(f"Successfully loaded spaCy {SPACY_MODEL} model")
                
                # Setup EntityRuler
                self._setup_entity_ruler()
            
        except OSError:
            logger.warning(f"spaCy {SPACY_MODEL} model not found. Please install it manually.")
            logger.warning(f"Run: python -m spacy download {SPACY_MODEL}")
            self.nlp = None
        
        # Setup NLTK resources
        if NLTK_AVAILABLE:
            try:
                import nltk as nltk_module
                from nltk.corpus import stopwords as nltk_stopwords
                nltk_module.download('punkt', quiet=True)
                nltk_module.download('averaged_perceptron_tagger', quiet=True)
                nltk_module.download('maxent_ne_chunker', quiet=True)
                nltk_module.download('words', quiet=True)
                nltk_module.download('stopwords', quiet=True)
                self.stopwords = set(nltk_stopwords.words(NLTK_STOPWORDS_LANG))
                logger.info("Successfully initialized NLTK resources")
            except Exception as e:
                logger.warning(f"Failed to initialize NLTK resources: {e}")
        
        # Setup transformers NER pipeline
        if TRANSFORMERS_AVAILABLE and TRANSFORMERS_NER_MODEL:
            try:
                import transformers as tf
                pipeline_func = getattr(tf, 'pipeline', None)
                if pipeline_func:
                    self.ner_pipeline = pipeline_func(
                        "ner",
                        model=TRANSFORMERS_NER_MODEL,
                        aggregation_strategy="simple"
                    )
                    logger.info("Successfully initialized transformers NER pipeline")
                else:
                    logger.warning("Transformers pipeline function not available")
                    self.ner_pipeline = None
            except Exception as e:
                logger.warning(f"Failed to initialize transformers NER pipeline: {e}")
                self.ner_pipeline = None
        else:
            if not TRANSFORMERS_NER_MODEL:
                logger.info("Transformers NER model disabled in config")
            self.ner_pipeline = None
    
    def _setup_entity_ruler(self):
        """Setup EntityRuler with climate science patterns"""
        if not self.nlp:
            return
        
        try:
            # Add EntityRuler to the pipeline
            if "entity_ruler" not in self.nlp.pipe_names:
                self.entity_ruler = self.nlp.add_pipe("entity_ruler", before="ner")
            else:
                self.entity_ruler = self.nlp.get_pipe("entity_ruler")
            
            # Create EntityRuler patterns
            entity_patterns = []
            
            for concept_key, variations in ENTITY_RULER_PATTERNS.items():
                concept_type = PATTERN_SCHEMA.get(concept_key, "UNKNOWN")
                for variation in variations:
                    entity_patterns.append({
                        "label": concept_type,
                        "pattern": variation,
                        "id": concept_key
                    })
            
            # Add patterns to EntityRuler
            if entity_patterns:
                self.entity_ruler.add_patterns(entity_patterns)
                logger.info(f"Added {len(entity_patterns)} patterns to EntityRuler")
                
        except Exception as e:
            logger.warning(f"Failed to setup EntityRuler: {e}")
            self.entity_ruler = None
    
    def extract_spacy_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract entities using spaCy NER"""
        if not self.nlp:
            return []
        
        doc = self.nlp(text)
        entities = []
        
        for ent in doc.ents:
            # Map spaCy entity types to our schema
            entity_type = self._map_spacy_entity_type(ent.label_)
            if entity_type:
                entities.append({
                    'text': ent.text,
                    'type': entity_type,
                    'start': ent.start_char,
                    'end': ent.end_char,
                    'confidence': CONFIDENCE_THRESHOLDS['spacy'],
                    'source': 'spacy'
                })
        
        return entities
    
    def extract_nltk_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract entities using NLTK NER"""
        if not NLTK_AVAILABLE:
            return []
        
        try:
            # Import NLTK functions locally
            from nltk.tokenize import word_tokenize
            from nltk.tag import pos_tag
            from nltk.chunk import ne_chunk
            
            # Tokenize and tag
            tokens = word_tokenize(text)
            pos_tags = pos_tag(tokens)
            chunks = ne_chunk(pos_tags)
            
            entities = []
            current_entity = []
            current_label = None
            
            for chunk in chunks:
                if hasattr(chunk, 'label'):
                    # This is a named entity
                    if current_label != chunk.label():
                        # Flush previous entity
                        if current_entity:
                            entity_text = ' '.join(current_entity)
                            entity_type = self._map_nltk_entity_type(current_label)
                            if entity_type:
                                # Find the entity position in text (approximate)
                                start_pos = text.lower().find(entity_text.lower())
                                end_pos = start_pos + len(entity_text) if start_pos >= 0 else 0
                                entities.append({
                                    'text': entity_text,
                                    'type': entity_type,
                                    'start': max(0, start_pos),
                                    'end': max(len(entity_text), end_pos),
                                    'confidence': CONFIDENCE_THRESHOLDS['nltk'],
                                    'source': 'nltk'
                                })
                        
                        # Start new entity
                        current_entity = [chunk[0][0]]
                        current_label = chunk.label()
                    else:
                        # Continue current entity
                        current_entity.append(chunk[0][0])
                else:
                    # Flush current entity if any
                    if current_entity:
                        entity_text = ' '.join(current_entity)
                        entity_type = self._map_nltk_entity_type(current_label)
                        if entity_type:
                            # Find the entity position in text (approximate)
                            start_pos = text.lower().find(entity_text.lower())
                            end_pos = start_pos + len(entity_text) if start_pos >= 0 else 0
                            entities.append({
                                'text': entity_text,
                                'type': entity_type,
                                'start': max(0, start_pos),
                                'end': max(len(entity_text), end_pos),
                                'confidence': CONFIDENCE_THRESHOLDS['nltk'],
                                'source': 'nltk'
                            })
                        current_entity = []
                        current_label = None
            
            # Handle final entity
            if current_entity:
                entity_text = ' '.join(current_entity)
                entity_type = self._map_nltk_entity_type(current_label)
                if entity_type:
                    # Find the entity position in text (approximate)
                    start_pos = text.lower().find(entity_text.lower())
                    end_pos = start_pos + len(entity_text) if start_pos >= 0 else 0
                    entities.append({
                        'text': entity_text,
                        'type': entity_type,
                        'start': max(0, start_pos),
                        'end': max(len(entity_text), end_pos),
                        'confidence': CONFIDENCE_THRESHOLDS['nltk'],
                        'source': 'nltk'
                    })
            
            return entities
            
        except Exception as e:
            logger.error(f"Error in NLTK entity extraction: {e}")
            return []
    
    def extract_transformers_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract entities using transformers NER pipeline"""
        if not self.ner_pipeline:
            return []
        
        try:
            # Use transformers NER
            results = self.ner_pipeline(text)
            entities = []
            
            for result in results:
                # Safely access dictionary fields
                entity_group = result.get('entity_group', '')
                word = result.get('word', '')
                start = result.get('start', 0)
                end = result.get('end', 0)
                score = result.get('score', 0.0)
                
                entity_type = self._map_transformers_entity_type(entity_group)
                if entity_type and score >= CONFIDENCE_THRESHOLDS['transformers']:
                    entities.append({
                        'text': word,
                        'type': entity_type,
                        'start': start,
                        'end': end,
                        'confidence': score,
                        'source': 'transformers'
                    })
            
            return entities
            
        except Exception as e:
            logger.error(f"Error in transformers entity extraction: {e}")
            return []
    
    def extract_entity_ruler_concepts(self, text: str) -> List[Dict[str, Any]]:
        """Extract concepts using spaCy EntityRuler"""
        if not self.nlp or not self.entity_ruler:
            return []
        
        doc = self.nlp(text)
        concepts = []
        
        for ent in doc.ents:
            # Check if this entity was matched by EntityRuler
            if ent.ent_id_:  # EntityRuler entities have ent_id_
                concepts.append({
                    'text': ent.text,
                    'type': ent.label_,
                    'start': ent.start_char,
                    'end': ent.end_char,
                    'confidence': CONFIDENCE_THRESHOLDS['entity_ruler'],
                    'source': 'entity_ruler',
                    'ruler_id': ent.ent_id_
                })
        
        return concepts
    
    def _map_spacy_entity_type(self, spacy_label: str) -> str | None:
        """Map spaCy entity labels to our concept schema"""
        mapping = {
            'ORG': 'AGENCIES',
            'GPE': 'LOCATION',
            'PERSON': 'PERSON',
            'DATE': 'TEMPORAL',
            'MONEY': 'ECONOMIC',
            'PERCENT': 'QUANTITATIVE',
            'QUANTITY': 'QUANTITATIVE',
            'CARDINAL': 'QUANTITATIVE',
            'PRODUCT': 'TECHNOLOGY',
            'EVENT': 'CLIMATE_VAR',
            'FAC': 'TECHNOLOGY',
            'LANGUAGE': 'OTHER',
            'LAW': 'POLICY',
            'LOC': 'LOCATION',
            'NORP': 'SOCIAL',
            'ORDINAL': 'QUANTITATIVE',
            'TIME': 'TEMPORAL',
            'WORK_OF_ART': 'REPORTS'
        }
        return mapping.get(spacy_label)
    
    def _map_nltk_entity_type(self, nltk_label: str | None) -> str | None:
        """Map NLTK entity labels to our concept schema"""
        if not nltk_label:
            return None
        mapping = {
            'ORGANIZATION': 'AGENCIES',
            'PERSON': 'PERSON',
            'LOCATION': 'LOCATION',
            'GPE': 'LOCATION',
            'GSP': 'LOCATION',
            'FACILITY': 'TECHNOLOGY'
        }
        return mapping.get(nltk_label)
    
    def _map_transformers_entity_type(self, transformers_label: str) -> str | None:
        """Map transformers entity labels to our concept schema"""
        mapping = {
            'ORG': 'AGENCIES',
            'PER': 'PERSON',
            'LOC': 'LOCATION',
            'MISC': 'OTHER'
        }
        return mapping.get(transformers_label)
    
    def extract_domain_specific_concepts(self, text: str) -> List[Dict[str, Any]]:
        """Extract climate science specific concepts using custom rules"""
        concepts = []
        
        for concept_type, patterns in DOMAIN_SPECIFIC_PATTERNS.items():
            for pattern in patterns:
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    concepts.append({
                        'text': match.group(),
                        'type': concept_type,
                        'start': match.start(),
                        'end': match.end(),
                        'confidence': CONFIDENCE_THRESHOLDS['domain_specific'],
                        'source': 'domain_specific'
                    })
        
        return concepts
    
    def extract_all_concepts(self, text: str) -> Dict[str, List[Dict[str, Any]]]:
        """Extract concepts using all available methods"""
        results = {
            'regex': [],
            'entity_ruler': [],
            'spacy': [],
            'nltk': [],
            'transformers': [],
            'domain_specific': []
        }
        
        # Extract using EntityRuler (prioritized over regex for exact matches)
        results['entity_ruler'] = self.extract_entity_ruler_concepts(text)
        
        # Extract using existing regex patterns (for complex patterns not suitable for EntityRuler)
        results['regex'] = self.extract_regex_concepts(text)
        
        # Extract using various NER approaches
        results['spacy'] = self.extract_spacy_entities(text)
        results['nltk'] = self.extract_nltk_entities(text)
        results['transformers'] = self.extract_transformers_entities(text)
        results['domain_specific'] = self.extract_domain_specific_concepts(text)
        
        return results
    
    def extract_regex_concepts(self, text: str) -> List[Dict[str, Any]]:
        """Extract concepts using the existing regex patterns"""
        concepts = []
        
        for key, pattern in REGEX_MAP.items():
            label = PATTERN_SCHEMA.get(key, "UNKNOWN")
            
            for match in pattern.finditer(text):
                concepts.append({
                    'text': match.group(),
                    'type': label,
                    'start': match.start(),
                    'end': match.end(),
                    'confidence': CONFIDENCE_THRESHOLDS['regex'],
                    'source': 'regex',
                    'regex_key': key
                })
        
        return concepts
    
    def merge_overlapping_concepts(self, concepts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge overlapping concepts, prioritizing higher confidence and longer spans"""
        if not concepts:
            return []
        
        # Filter out concepts without required fields and sort by start position, then by confidence (descending)
        valid_concepts = [c for c in concepts if 'start' in c and 'end' in c and 'confidence' in c]
        sorted_concepts = sorted(valid_concepts, key=lambda x: (x['start'], -x['confidence']))
        
        merged = []
        for concept in sorted_concepts:
            # Check for overlap with existing concepts
            overlapping = False
            for i, existing in enumerate(merged):
                if self._concepts_overlap(concept, existing):
                    # Choose the better concept
                    if self._should_replace_concept(concept, existing):
                        merged[i] = concept
                    overlapping = True
                    break
            
            if not overlapping:
                merged.append(concept)
        
        return merged
    
    def _concepts_overlap(self, concept1: Dict[str, Any], concept2: Dict[str, Any]) -> bool:
        """Check if two concepts overlap in text span"""
        return not (concept1['end'] <= concept2['start'] or concept2['end'] <= concept1['start'])
    
    def _should_replace_concept(self, new_concept: Dict[str, Any], existing_concept: Dict[str, Any]) -> bool:
        """Determine if new concept should replace existing one"""
        new_priority = EXTRACTION_PRIORITY.get(new_concept['source'], 0)
        existing_priority = EXTRACTION_PRIORITY.get(existing_concept['source'], 0)
        
        # Prioritize by source first
        if new_priority != existing_priority:
            return new_priority > existing_priority
        
        # If same source, prioritize longer spans
        new_length = new_concept['end'] - new_concept['start']
        existing_length = existing_concept['end'] - existing_concept['start']
        if new_length != existing_length:
            return new_length > existing_length
        
        # Finally, prioritize higher confidence
        return new_concept['confidence'] > existing_concept['confidence']


def build_hybrid_concept_index(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Build concept index using hybrid extraction approach"""
    extractor = ConceptExtractor()
    
    # Traditional regex-based index
    regex_index = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))
    
    # New hybrid index with detailed concept information
    hybrid_index = defaultdict(lambda: defaultdict(list))
    
    # Concept statistics
    stats = {
        'total_sentences': len(data),
        'extraction_methods': defaultdict(int),
        'concept_types': defaultdict(int),
        'concepts_per_sentence': []
    }
    
    logger.info(f"Processing {len(data)} sentences with hybrid concept extraction...")
    
    for i, entry in enumerate(data):
        if i % PROGRESS_INTERVAL == 0:
            logger.info(f"Processed {i}/{len(data)} sentences")
        
        text = entry["text"]
        sent_id = entry["id"]
        
        # Extract concepts using all methods
        all_concepts = extractor.extract_all_concepts(text)
        
        # Flatten and merge concepts
        flat_concepts = []
        for method, concepts in all_concepts.items():
            for concept in concepts:
                flat_concepts.append(concept)
                stats['extraction_methods'][method] += 1
        
        # Merge overlapping concepts
        merged_concepts = extractor.merge_overlapping_concepts(flat_concepts)
        stats['concepts_per_sentence'].append(len(merged_concepts))
        
        # Build traditional regex index for backward compatibility
        for key, pattern in REGEX_MAP.items():
            label = PATTERN_SCHEMA.get(key, "UNKNOWN")
            if pattern.search(text):
                regex_index[label][key][key].add(sent_id)
        
        # Build hybrid index
        for concept in merged_concepts:
            concept_type = concept['type']
            stats['concept_types'][concept_type] += 1
            
            hybrid_index[concept_type][concept['text']].append({
                'sentence_id': sent_id,
                'confidence': concept['confidence'],
                'source': concept['source'],
                'start': concept['start'],
                'end': concept['end']
            })
    
    # Format traditional index
    final_regex_index = {
        label: {
            regex_key: [
                {"concept": concept, "sentences": sorted(list(sids))}
                for concept, sids in concept_dict.items()
            ]
            for regex_key, concept_dict in regex_group.items()
        }
        for label, regex_group in regex_index.items()
    }
    
    # Format hybrid index
    final_hybrid_index = {
        concept_type: {
            concept_text: {
                'sentences': concept_instances,
                'total_occurrences': len(concept_instances),
                'unique_sentences': len(set(inst['sentence_id'] for inst in concept_instances)),
                'avg_confidence': sum(inst['confidence'] for inst in concept_instances) / len(concept_instances),
                'sources': list(set(inst['source'] for inst in concept_instances))
            }
            for concept_text, concept_instances in concepts.items()
        }
        for concept_type, concepts in hybrid_index.items()
    }
    
    # Calculate final statistics
    if stats['concepts_per_sentence']:
        stats['avg_concepts_per_sentence'] = sum(stats['concepts_per_sentence']) / len(stats['concepts_per_sentence'])
        stats['max_concepts_per_sentence'] = max(stats['concepts_per_sentence'])
    
    return {
        'regex_index': final_regex_index,
        'hybrid_index': final_hybrid_index,
        'statistics': dict(stats)
    }


def load_jsonl_sentences(jsonl_path: str) -> List[Dict[str, Any]]:
    """Load sentences from JSONL file"""
    sentences = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                sentences.append(json.loads(line))
    return sentences


def build_concept_index(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Legacy function for backward compatibility"""
    logger.warning("Using legacy build_concept_index function. Consider using build_hybrid_concept_index for better results.")
    
    index = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))

    for entry in data:
        text = entry["text"]
        sent_id = entry["id"]

        for key, pattern in REGEX_MAP.items():
            label = PATTERN_SCHEMA.get(key, "UNKNOWN")

            if key == "specific_year":
                for match in pattern.findall(text):
                    index[label][key][match].add(sent_id)
            else:
                if pattern.search(text):
                    index[label][key][key].add(sent_id)

    final_index = {
        label: {
            regex_key: [
                {"concept": concept, "sentences": sorted(list(sids))}
                for concept, sids in concept_dict.items()
            ]
            for regex_key, concept_dict in regex_group.items()
        }
        for label, regex_group in index.items()
    }

    return final_index


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build concept index from sentences using hybrid extraction")
    parser.add_argument("--input_file", required=True, help="Input JSONL file path")
    parser.add_argument("--output_file", required=True, help="Output JSON file path")
    parser.add_argument("--method", choices=['hybrid', 'regex'], default='hybrid',
                       help="Extraction method: 'hybrid' for regex+NER, 'regex' for regex-only")
    parser.add_argument("--verbose", action='store_true', help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info(f"Loading sentences from {args.input_file}")
    data = load_jsonl_sentences(args.input_file)
    logger.info(f"Loaded {len(data)} sentences")
    
    if args.method == 'hybrid':
        logger.info("Using hybrid concept extraction (regex + NER)")
        concept_index = build_hybrid_concept_index(data)
    else:
        logger.info("Using regex-only concept extraction")
        concept_index = build_concept_index(data)
    
    logger.info(f"Saving concept index to {args.output_file}")
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(concept_index, f, ensure_ascii=False, indent=2)

    logger.info(f"Concept index saved to: {args.output_file}")
    
    # Print summary statistics
    if args.method == 'hybrid' and 'statistics' in concept_index:
        stats = concept_index['statistics']
        logger.info("=== EXTRACTION STATISTICS ===")
        logger.info(f"Total sentences processed: {stats['total_sentences']}")
        logger.info(f"Average concepts per sentence: {stats.get('avg_concepts_per_sentence', 0):.2f}")
        logger.info(f"Max concepts in single sentence: {stats.get('max_concepts_per_sentence', 0)}")
        logger.info("Concepts by extraction method:")
        extraction_methods = stats.get('extraction_methods', {})
        for method, count in extraction_methods.items():
            logger.info(f"  {method}: {count}")
        logger.info("Concepts by type:")
        concept_types = stats.get('concept_types', {})
        for concept_type, count in concept_types.items():
            logger.info(f"  {concept_type}: {count}")
    
    logger.info("Concept extraction completed successfully!")
