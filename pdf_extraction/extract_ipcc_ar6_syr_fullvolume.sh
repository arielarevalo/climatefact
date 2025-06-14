#!/bin/bash

# Exit on any error
set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="$SCRIPT_DIR/extraction.log"
VENV_DIR="$SCRIPT_DIR/.venv"
INPUT_PDF="$SCRIPT_DIR/docs/IPCC_AR6_SYR_FullVolume.pdf"
OUTPUT_DIR="$SCRIPT_DIR/docs/IPCC_AR6_SYR_FullVolume_markdown"
START_PAGE=18
END_PAGE=130

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_FILE"
}

# Error handling function
error_exit() {
    log "${RED}ERROR: $1${NC}"
    exit 1
}

# Success message function
success() {
    log "${GREEN}SUCCESS: $1${NC}"
}

# Warning message function
warning() {
    log "${YELLOW}WARNING: $1${NC}"
}

# Info message function
info() {
    log "${BLUE}INFO: $1${NC}"
}

# Cleanup function
cleanup() {
    if [[ -n "$VIRTUAL_ENV" ]] && [[ "$VIRTUAL_ENV" == "$VENV_DIR" ]]; then
        info "Deactivating virtual environment..."
        deactivate 2>/dev/null || true
    fi
}

# Set trap for cleanup on exit
trap cleanup EXIT

# Check prerequisites
check_prerequisites() {
    info "Checking prerequisites..."
    
    # Check if Python is installed
    if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
        error_exit "Python is not installed or not in PATH"
    fi
    
    # Check if input PDF exists
    if [[ ! -f "$INPUT_PDF" ]]; then
        error_exit "Input PDF not found: $INPUT_PDF"
    fi
    
    # Check if requirements.txt exists
    if [[ ! -f "$SCRIPT_DIR/requirements.txt" ]]; then
        error_exit "requirements.txt not found in $SCRIPT_DIR"
    fi
    
    success "Prerequisites check passed"
}

# Create virtual environment
create_venv() {
    info "Setting up virtual environment..."
    
    if [[ -d "$VENV_DIR" ]]; then
        warning "Virtual environment already exists, removing old one..."
        rm -rf "$VENV_DIR"
    fi
    
    # Create virtual environment
    if command -v python3 &> /dev/null; then
        python3 -m venv "$VENV_DIR" || error_exit "Failed to create virtual environment"
    else
        python -m venv "$VENV_DIR" || error_exit "Failed to create virtual environment"
    fi
    
    success "Virtual environment created"
}

# Activate virtual environment
activate_venv() {
    info "Activating virtual environment..."
    
    # shellcheck source=/dev/null
    source "$VENV_DIR/bin/activate" || error_exit "Failed to activate virtual environment"
    
    success "Virtual environment activated"
}

# Install dependencies
install_dependencies() {
    info "Installing dependencies..."
    
    # Upgrade pip
    pip install --upgrade pip || error_exit "Failed to upgrade pip"
    
    # Install requirements
    pip install -r "$SCRIPT_DIR/requirements.txt" || error_exit "Failed to install requirements"
    
    success "Dependencies installed successfully"
}

# Run PDF extraction
run_extraction() {
    info "Starting PDF extraction..."
    info "Input PDF: $INPUT_PDF"
    info "Output directory: $OUTPUT_DIR"
    info "Page range: $START_PAGE to $END_PAGE"
    
    # Create output directory if it doesn't exist
    mkdir -p "$OUTPUT_DIR"
    
    # Run the extraction script
    python "$SCRIPT_DIR/pdf_to_markdown.py" \
        --input_pdf "$INPUT_PDF" \
        --output_dir "$OUTPUT_DIR" \
        --start_page "$START_PAGE" \
        --end_page "$END_PAGE" || error_exit "PDF extraction failed"
    
    success "PDF extraction completed successfully"
}

# Display results
display_results() {
    info "Extraction Results:"
    
    if [[ -d "$OUTPUT_DIR" ]]; then
        local file_count
        file_count=$(find "$OUTPUT_DIR" -name "*.md" | wc -l)
        success "Generated $file_count markdown files in $OUTPUT_DIR"
        
        # Display first few files as example
        info "Sample output files:"
        find "$OUTPUT_DIR" -name "*.md" | head -5 | while read -r file; do
            echo "  - $(basename "$file")"
        done
    else
        warning "Output directory not found"
    fi
}

# Main execution
main() {
    info "Starting IPCC AR6 SYR Full Volume PDF extraction..."
    info "Log file: $LOG_FILE"
    
    check_prerequisites
    create_venv
    activate_venv
    install_dependencies
    run_extraction
    display_results
    
    success "All operations completed successfully!"
    info "Check the log file for detailed information: $LOG_FILE"
}

# Run main function
main "$@"