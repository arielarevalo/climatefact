import argparse
import subprocess
import shutil
import sys
from pathlib import Path


def convert_pdf_to_markdown_pages(input_pdf: str, output_dir: str, start_page: int = 1, end_page: int = -1) -> None:
    """
    Convert each page of a PDF to individual Markdown files.

    Args:
        input_pdf: Path to the input PDF file
        output_dir: Directory to save the Markdown files
        start_page: First page to process (1-indexed, inclusive)
        end_page: Last page to process (1-indexed, inclusive). If -1, process until end of document
    """
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Get PDF file name without extension for naming
    pdf_name = Path(input_pdf).stem
    
    # Check if mineru is available
    try:
        subprocess.run(["mineru", "--help"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: mineru not found. Make sure it's installed and in your PATH.")
        sys.exit(1)
    
    successful_pages = 0
    failed_pages = []
    
    page = start_page
    while True:
        # If end_page is specified and we've reached it, stop
        if end_page != -1 and page > end_page:
            break
            
        print(f"Processing page {page}...", end=" ", flush=True)
        
        try:
            # Run mineru for single page with output capture
            result = subprocess.run([
                "mineru", "-p", input_pdf, "-o", output_dir, 
                "-m", "txt", "-s", str(page), "-e", str(page)
            ], capture_output=True, text=True, check=True)
            
            # Locate the generated .md file
            pdf_dir = output_path / pdf_name / "txt"
            md_files = list(pdf_dir.glob("*.md"))
            
            if md_files:
                source_file = md_files[0]  # Should be only one file
                target_file = output_path / f"{pdf_name}_{page}.md"
                shutil.move(str(source_file), str(target_file))
                successful_pages += 1
                print("✓")
                page += 1
            else:
                print("✗ (no markdown generated)")
                failed_pages.append(page)
                break
                
        except subprocess.CalledProcessError as e:
            print(f"✗ (mineru failed: {e.stderr.strip() if e.stderr else 'unknown error'})")
            failed_pages.append(page)
            # If we're at the start page and it fails, this might be a real error
            if page == start_page:
                print(f"Error: Failed to process starting page {page}. Check if PDF is valid and page exists.")
                break
            # Otherwise, we might have reached the end of the document
            break
        except Exception as e:
            print(f"✗ (unexpected error: {e})")
            failed_pages.append(page)
            break
    
    # Clean up the working directory structure
    working_dir = output_path / pdf_name
    if working_dir.exists():
        shutil.rmtree(working_dir)

    # Report results
    print(f"\nExtraction complete:")
    print(f"  Successfully processed: {successful_pages} pages")
    if failed_pages:
        print(f"  Failed pages: {failed_pages}")
    print(f"  Output directory: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert PDF pages to individual Markdown files using mineru"
    )
    parser.add_argument("--input_pdf", required=True, help="Path to the input PDF file")
    parser.add_argument("--output_dir", required=True, help="Directory to save the Markdown files")
    parser.add_argument("--start_page", type=int, default=1, help="First page to process (default: 1)")
    parser.add_argument("--end_page", type=int, default=-1, help="Last page to process (-1 for all pages, default: -1)")

    args = parser.parse_args()

    # Validate input file exists
    if not Path(args.input_pdf).exists():
        print(f"Error: Input PDF file '{args.input_pdf}' does not exist.")
        return 1

    # Validate page range
    if args.start_page < 1:
        print("Error: start_page must be >= 1")
        return 1
    
    if args.end_page != -1 and args.end_page < args.start_page:
        print("Error: end_page must be >= start_page or -1")
        return 1

    try:
        convert_pdf_to_markdown_pages(args.input_pdf, args.output_dir, args.start_page, args.end_page)
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
