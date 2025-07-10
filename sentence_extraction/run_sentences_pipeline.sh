#!/bin/bash

# Knowledge Building Pipeline - Sentences Processing
# This script runs the complete pipeline to process markdown files into sentences with embeddings

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to load environment variables from .env file
load_env() {
    local env_file="$(dirname "${BASH_SOURCE[0]}")/.env"
    if [[ -f "$env_file" ]]; then
        print_status "Loading environment variables from: $env_file"
        # Export variables from .env file, ignoring comments and empty lines
        export $(grep -v '^#' "$env_file" | grep -v '^$' | xargs)
    fi
}

# Function to print colored output
print_status() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

# Function to show usage
usage() {
    echo "Usage: $0 --input_dir <markdown_dir> --output_file <final_output.jsonl> [--intermediate_dir <temp_dir>]"
    echo ""
    echo "Options:"
    echo "  --input_dir         Directory containing markdown files"
    echo "  --output_file       Final output JSONL file with embeddings"
    echo "  --intermediate_dir  Directory for intermediate files (optional, uses temp dir if not specified)"
    echo "  --help             Show this help message"
    echo ""
    echo "Environment variables required:"
    echo "  AZURE_OPENAI_API_KEY    Azure OpenAI API key"
    echo "  AZURE_OPENAI_ENDPOINT   Azure OpenAI endpoint URL"
    exit 1
}

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STEPS_DIR="$SCRIPT_DIR/scripts/sentences_pipeline_steps"
VENV_DIR="$SCRIPT_DIR/venv"

# Function to setup Python virtual environment
setup_venv() {
    # Check if virtual environment exists
    if [[ ! -f "$VENV_DIR/bin/python" ]]; then
        print_status "Creating Python virtual environment..."
        python3 -m venv "$VENV_DIR"
        if [[ $? -ne 0 ]]; then
            print_error "Failed to create virtual environment."
            exit 1
        fi
    fi

    # Activate virtual environment
    print_status "Activating virtual environment..."
    source "$VENV_DIR/bin/activate"
    if [[ -z "$VIRTUAL_ENV" ]]; then
        print_error "Failed to activate virtual environment."
        exit 1
    fi

    # Install dependencies
    if [[ -f "$SCRIPT_DIR/requirements.txt" ]]; then
        print_status "Installing dependencies from requirements.txt..."
        pip install -r "$SCRIPT_DIR/requirements.txt"
        if [[ $? -ne 0 ]]; then
            print_error "Failed to install Python dependencies."
            exit 1
        fi
    else
        print_error "requirements.txt not found at: $SCRIPT_DIR/requirements.txt"
        exit 1
    fi
}

# Function to cleanup Python virtual environment
cleanup_venv() {
    if [[ -d "$VENV_DIR" ]]; then
        print_status "Cleaning up virtual environment..."
        rm -rf "$VENV_DIR"
    fi
}

# Load environment variables first
load_env

# Setup Python virtual environment
setup_venv

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --input_dir)
            INPUT_DIR="$2"
            shift 2
            ;;
        --output_file)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        --intermediate_dir)
            INTERMEDIATE_DIR="$2"
            shift 2
            ;;
        --help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# Validate required arguments
if [[ -z "$INPUT_DIR" || -z "$OUTPUT_FILE" ]]; then
    print_error "Missing required arguments"
    usage
fi

# Check if input directory exists
if [[ ! -d "$INPUT_DIR" ]]; then
    print_error "Input directory does not exist: $INPUT_DIR"
    exit 1
fi

# Check for required environment variables
if [[ -z "$AZURE_OPENAI_API_KEY" || -z "$AZURE_OPENAI_ENDPOINT" ]]; then
    print_error "Required environment variables not set:"
    print_error "  AZURE_OPENAI_API_KEY"
    print_error "  AZURE_OPENAI_ENDPOINT"
    print_error ""
    print_error "You can set them in a .env file or export them:"
    print_error "  export AZURE_OPENAI_API_KEY='your-key'"
    print_error "  export AZURE_OPENAI_ENDPOINT='your-endpoint'"
    exit 1
fi

# Set up intermediate directory
if [[ -z "$INTERMEDIATE_DIR" ]]; then
    INTERMEDIATE_DIR=$(mktemp -d -t sentences_pipeline_XXXXXX)
    CLEANUP_INTERMEDIATE=true
    print_status "Created temporary intermediate directory: $INTERMEDIATE_DIR"
else
    mkdir -p "$INTERMEDIATE_DIR"
    CLEANUP_INTERMEDIATE=false
fi

# Check if steps directory exists
if [[ ! -d "$STEPS_DIR" ]]; then
    print_error "Steps directory not found: $STEPS_DIR"
    exit 1
fi

# Define intermediate file names
STEP1_OUTPUT="$INTERMEDIATE_DIR/step1_paragraphs.jsonl"
STEP2_OUTPUT="$INTERMEDIATE_DIR/step2_coref_resolved.jsonl"
STEP3_OUTPUT="$INTERMEDIATE_DIR/step3_sentences.jsonl"
STEP4_OUTPUT="$INTERMEDIATE_DIR/step4_filtered_sentences.jsonl"

# Cleanup function
cleanup() {
    if [[ "$CLEANUP_INTERMEDIATE" == "true" && -d "$INTERMEDIATE_DIR" ]]; then
        print_status "Cleaning up temporary directory: $INTERMEDIATE_DIR"
        rm -rf "$INTERMEDIATE_DIR"
    fi
    # Deactivate and cleanup virtual environment
    cleanup_venv
}

# Set trap to cleanup on exit
trap cleanup EXIT

# Pipeline execution
print_status "Starting Knowledge Building Pipeline"
print_status "Input directory: $INPUT_DIR"
print_status "Output file: $OUTPUT_FILE"
print_status "Intermediate directory: $INTERMEDIATE_DIR"
echo ""

# Step 1: Split paragraphs
print_status "Step 1/5: Splitting paragraphs from markdown files..."
if python "$STEPS_DIR/split_paragraphs.py" --input_dir "$INPUT_DIR" --output_file "$STEP1_OUTPUT"; then
    print_success "Step 1 completed: Paragraphs extracted"
else
    print_error "Step 1 failed: split_paragraphs.py"
    exit 1
fi

# Step 2: Resolve coreferences
print_status "Step 2/5: Resolving coreferences..."
if python "$STEPS_DIR/resolve_coreferences.py" --input_file "$STEP1_OUTPUT" --output_file "$STEP2_OUTPUT"; then
    print_success "Step 2 completed: Coreferences resolved"
else
    print_error "Step 2 failed: resolve_coreferences.py"
    exit 1
fi

# Step 3: Split sentences
print_status "Step 3/5: Splitting sentences..."
if python "$STEPS_DIR/split_sentences.py" --input_file "$STEP2_OUTPUT" --output_file "$STEP3_OUTPUT"; then
    print_success "Step 3 completed: Sentences split"
else
    print_error "Step 3 failed: split_sentences.py"
    exit 1
fi

# Step 4: Filter image sentences
print_status "Step 4/5: Filtering image sentences..."
if python "$STEPS_DIR/filter_image_sentences.py" --input_file "$STEP3_OUTPUT" --output_file "$STEP4_OUTPUT"; then
    print_success "Step 4 completed: Image sentences filtered"
else
    print_error "Step 4 failed: filter_image_sentences.py"
    exit 1
fi

# Step 5: Add embeddings
print_status "Step 5/5: Adding embeddings..."
if python "$STEPS_DIR/add_embeddings.py" --input_file "$STEP4_OUTPUT" --output_file "$OUTPUT_FILE"; then
    print_success "Step 5 completed: Embeddings added"
else
    print_error "Step 5 failed: add_embeddings.py"
    exit 1
fi

echo ""
print_success "Pipeline completed successfully!"
print_success "Final output saved to: $OUTPUT_FILE"

# Show final statistics
if [[ -f "$OUTPUT_FILE" ]]; then
    SENTENCE_COUNT=$(wc -l < "$OUTPUT_FILE")
    print_status "Total sentences with embeddings: $SENTENCE_COUNT"
fi

echo ""
print_status "Pipeline execution finished"
