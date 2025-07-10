# IPCC AR6 SYR PDF to Markdown Extraction

Quick script to extract pages 18-130 from the IPCC AR6 SYR Full Volume PDF and convert them to individual markdown files.

## What you need

- Python 3.7+
- The IPCC PDF file (put it in `docs/IPCC_AR6_SYR_FullVolume.pdf`)

## How to run it

1. Put the PDF in the right spot:
   ```
   pdf_extraction/docs/IPCC_AR6_SYR_FullVolume.pdf
   ```

2. Run the script:
   ```bash
   cd pdf_extraction
   chmod +x extract_ipccar6_syr_fullvolume.sh
   ./extract_ipccar6_syr_fullvolume.sh
   ```

That's it. The script will:
- Set up a Python virtual environment
- Install the mineru library 
- Extract pages 18-130 from the PDF
- Save each page as a separate markdown file in `docs/IPCC_AR6_SYR_FullVolume_markdown/`

## If something breaks

Check `extraction.log` for what went wrong. Common issues:
- PDF file not in the right place
- Python not installed 
- No internet connection (needed to install mineru)

## Output

You'll get files like:
- `IPCC_AR6_SYR_FullVolume_18.md`
- `IPCC_AR6_SYR_FullVolume_19.md`
- ... and so on up to page 130

The script tells you how many files it created when it's done.
