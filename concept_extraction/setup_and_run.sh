#!/bin/bash

# =============================================================================
# Knowledge Building - Hybrid Concept Extraction Setup Script
# =============================================================================
# This script helps newcomers get started with the hybrid concept extraction
# system after they have:
# 1. Created and activated a virtual environment
# 2. Installed requirements with: pip install -r requirements.txt
#
# Usage:
#   ./setup_and_run.sh [--demo]
#
# Options:
#   --demo    Run a demonstration with sample data
#   --help    Show this help message
# =============================================================================

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR"

echo -e "${BLUE}=== Knowledge Building - Hybrid Concept Extraction Setup ===${NC}"
echo

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if we're in a virtual environment
check_venv() {
    if [[ -z "$VIRTUAL_ENV" ]]; then
        print_error "No virtual environment detected!"
        echo "Please activate your virtual environment first:"
        echo "  source .venv/bin/activate"
        echo "  # or"
        echo "  source venv/bin/activate"
        exit 1
    else
        print_status "Virtual environment detected: $VIRTUAL_ENV"
    fi
}

# Function to install spaCy models
install_spacy_models() {
    print_status "Installing required spaCy models..."
    
    # Check if en_core_web_sm is already installed
    if python -c "import spacy; spacy.load('en_core_web_sm')" 2>/dev/null; then
        print_status "spaCy en_core_web_sm model already installed ✓"
    else
        print_status "Installing spaCy en_core_web_sm model..."
        python -m spacy download en_core_web_sm
        print_status "spaCy model installed ✓"
    fi
}

# Function to download NLTK data
setup_nltk() {
    print_status "Setting up NLTK data..."
    python -c "
import nltk
import ssl

# Handle SSL certificate issues
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download required NLTK data
downloads = ['punkt', 'averaged_perceptron_tagger', 'maxent_ne_chunker', 'words', 'stopwords']
for item in downloads:
    try:
        nltk.download(item, quiet=True)
        print(f'Downloaded {item} ✓')
    except Exception as e:
        print(f'Failed to download {item}: {e}')
"
    print_status "NLTK setup complete ✓"
}

# Function to create sample data for demo
create_sample_data() {
    local sample_file="$PROJECT_DIR/sample_passages.jsonl"
    
    if [[ -f "$sample_file" ]]; then
        print_status "Sample data already exists: $sample_file"
        return
    fi
    
    print_status "Creating sample data for demonstration..."
    
    cat > "$sample_file" << 'EOF'
{"id": "sample_001", "text": "Global surface temperature has increased by approximately 1.1°C since the pre-industrial period, with CO₂ concentrations reaching 415 ppm in 2021."}
{"id": "sample_002", "text": "The IPCC AR6 report highlights that limiting warming to 1.5°C requires rapid, far-reaching transitions in energy, land, urban infrastructure, and industrial systems."}
{"id": "sample_003", "text": "Methane (CH4) emissions from agriculture and fossil fuel extraction contribute significantly to short-lived climate forcers, with a 100-year global warming potential 28 times that of CO₂."}
{"id": "sample_004", "text": "The SSP1-1.9 scenario represents a pathway consistent with limiting warming to 1.5°C, requiring net-zero CO₂ emissions by 2050 and substantial negative emissions thereafter."}
{"id": "sample_005", "text": "Sea-level rise of 0.2-0.3 meters is projected by 2050 under RCP4.5, with potential for 1-2 meters by 2100 under high-emission scenarios like SSP5-8.5."}
{"id": "sample_006", "text": "Carbon capture and storage (CCS) technologies, including direct air carbon capture (DACCS), are essential for achieving net-zero emissions in hard-to-abate sectors."}
{"id": "sample_007", "text": "The Paris Agreement aims to keep global warming well below 2°C and pursue efforts to limit it to 1.5°C through nationally determined contributions (NDCs)."}
{"id": "sample_008", "text": "Renewable energy sources like photovoltaic (PV) solar and wind power have experienced dramatic cost reductions, with LCOE falling by 70-90% since 2010."}
{"id": "sample_009", "text": "Climate impact drivers (CIDs) such as heatwaves, droughts, and extreme precipitation events are projected to intensify under continued warming."}
{"id": "sample_010", "text": "The UNFCCC process emphasizes the importance of adaptation measures and loss-and-damage financing for vulnerable developing countries and SIDS."}
EOF
    
    print_status "Sample data created: $sample_file"
}

# Function to run the concept extraction
run_concept_extraction() {
    local input_file="$1"
    local output_file="$2"
    local method="${3:-hybrid}"
    
    print_status "Running concept extraction..."
    print_status "Input: $input_file"
    print_status "Output: $output_file"
    print_status "Method: $method"
    echo
    
    cd "$PROJECT_DIR"
    python "$PROJECT_DIR/scripts/build_concept_index.py" \
        --input_file "$input_file" \
        --output_file "$output_file" \
        --method "$method" \
        --verbose
}

# Function to show results summary
show_results_summary() {
    local output_file="$1"
    
    if [[ ! -f "$output_file" ]]; then
        print_warning "Output file not found: $output_file"
        return
    fi
    
    print_status "Results summary:"
    echo
    
    # Extract key statistics using python
    python -c "
import json
import sys

try:
    with open('$output_file', 'r') as f:
        data = json.load(f)
    
    stats = data.get('statistics', {})
    hybrid_index = data.get('hybrid_index', {})
    
    print('📊 EXTRACTION STATISTICS:')
    print(f'   Total sentences: {stats.get(\"total_sentences\", 0)}')
    print(f'   Avg concepts per sentence: {stats.get(\"avg_concepts_per_sentence\", 0):.2f}')
    print(f'   Max concepts in one sentence: {stats.get(\"max_concepts_per_sentence\", 0)}')
    print()
    
    print('🔍 TOP CONCEPT TYPES:')
    concept_types = stats.get('concept_types', {})
    sorted_types = sorted(concept_types.items(), key=lambda x: x[1], reverse=True)
    for concept_type, count in sorted_types[:5]:
        print(f'   {concept_type}: {count} concepts')
    print()
    
    print('📋 EXTRACTION METHODS:')
    methods = stats.get('extraction_methods', {})
    for method, count in methods.items():
        print(f'   {method}: {count} extractions')
    print()
    
    print('🎯 SAMPLE EXTRACTED CONCEPTS:')
    sample_count = 0
    for concept_type, concepts in hybrid_index.items():
        if sample_count >= 10:
            break
        for concept_text, details in concepts.items():
            if sample_count >= 10:
                break
            occurrences = details.get('total_occurrences', 0)
            confidence = details.get('avg_confidence', 0)
            print(f'   {concept_type}: \"{concept_text}\" (occurs {occurrences}x, conf: {confidence:.2f})')
            sample_count += 1
    
except Exception as e:
    print(f'Error reading results: {e}', file=sys.stderr)
    sys.exit(1)
"
}

# Function to show help
show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo
    echo "Setup and run the hybrid concept extraction system."
    echo
    echo "OPTIONS:"
    echo "  --demo      Run demonstration with sample data"
    echo "  --help      Show this help message"
    echo
    echo "EXAMPLES:"
    echo "  $0 --demo                                    # Run demo with sample data"
    echo "  $0                                          # Setup only (install models)"
    echo
    echo "PREREQUISITES:"
    echo "1. Virtual environment must be activated"
    echo "2. Requirements must be installed: pip install -r requirements.txt"
    echo
    echo "FOR CUSTOM DATA:"
    echo "After running setup, use the concept extraction script directly:"
    echo "  python scripts/build_concept_index.py --input_file your_data.jsonl --output_file results.json"
}

# Main execution
main() {
    local run_demo=false
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --demo)
                run_demo=true
                shift
                ;;
            --help)
                show_help
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    # Check prerequisites
    check_venv
    
    # Install required models and data
    install_spacy_models
    setup_nltk
    
    print_status "Setup complete! ✓"
    echo
    
    if [[ "$run_demo" == true ]]; then
        # Create sample data and run demo
        create_sample_data
        
        local input_file="$PROJECT_DIR/sample_passages.jsonl"
        local output_file="$PROJECT_DIR/demo_concept_index.json"
        
        run_concept_extraction "$input_file" "$output_file" "hybrid"
        
        echo
        print_status "Demo completed! 🎉"
        echo
        show_results_summary "$output_file"
        
        echo
        print_status "Demo files created:"
        echo "  📝 Input data: $input_file"
        echo "  📊 Results: $output_file"
        echo
        print_status "Next steps:"
        echo "  • Examine the results in: $output_file"
        echo "  • Try with your own data: python scripts/build_concept_index.py --input_file your_data.jsonl --output_file results.json"
        echo "  • Use --method regex for regex-only extraction"
        echo "  • Add --verbose flag for detailed logging"
        
    else
        print_status "Setup completed successfully! 🎉"
        echo
        print_status "Ready to extract concepts from your data!"
        echo
        print_status "Usage examples:"
        echo "  # Run with sample data:"
        echo "  $0 --demo"
        echo
        echo "  # Run with your own data:"
        echo "  python scripts/build_concept_index.py --input_file passages.jsonl --output_file concept_index.json"
        echo
        echo "  # Use regex-only method:"
        echo "  python scripts/build_concept_index.py --input_file passages.jsonl --output_file concept_index.json --method regex"
    fi
}

# Run main function
main "$@"
