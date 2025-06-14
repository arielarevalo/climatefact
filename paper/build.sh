#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

build_paper() {
    local lang="$1"
    local output_name="$2"
    local src_dir="${SCRIPT_DIR}/${lang}"

    echo "==> Building ${lang} paper..."
    cd "${src_dir}"

    export TEXINPUTS=".:${SCRIPT_DIR}:${TEXINPUTS:-}"

    pdflatex -interaction=nonstopmode main.tex > /dev/null 2>&1
    bibtex main > /dev/null 2>&1
    pdflatex -interaction=nonstopmode main.tex > /dev/null 2>&1
    pdflatex -interaction=nonstopmode main.tex > /dev/null 2>&1

    cp main.pdf "${SCRIPT_DIR}/${output_name}"

    # Clean auxiliary files
    rm -f main.aux main.bbl main.blg main.log main.out main.toc main.synctex.gz main.pdf

    echo "    ${output_name} ready."
}

build_paper "es" "climatefact_es.pdf"
build_paper "en" "climatefact_en.pdf"

echo "==> Done. PDFs are in ${SCRIPT_DIR}/"
