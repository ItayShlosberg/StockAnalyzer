#!/usr/bin/env python3
"""
Markdown to PDF Converter

Converts markdown files to nicely styled PDF documents.

Usage:
    python md_to_pdf.py <input_md_file> <output_folder> <output_filename>

Example:
    python md_to_pdf.py report.md /path/to/output my_report.pdf
    /Users/itay.shlosberg/Desktop/JLL/workplace/JLLT-EDP-EnterpriseEntityResolution/.venv/bin/python /Users/itay.shlosberg/Desktop/JLL/workplace/JLLT-EDP-EnterpriseEntityResolution/python_lib_uc/research/utils/md_to_pdf.py "/Users/itay.shlosberg/Downloads/blocking_analysis_report.md" /Users/itay.shlosberg/Downloads test_output.pdf
"""

import argparse
import os
import sys

import markdown2
from weasyprint import HTML


CSS_STYLES = """
body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
    font-size: 10pt;
    line-height: 1.4;
    max-width: 100%;
    margin: 0;
    padding: 15px;
    color: #333;
}
h1 {
    color: #2c3e50;
    border-bottom: 2px solid #3498db;
    padding-bottom: 10px;
    font-size: 22pt;
}
h2 {
    color: #34495e;
    border-bottom: 1px solid #bdc3c7;
    padding-bottom: 5px;
    margin-top: 25px;
    font-size: 16pt;
}
h3 {
    color: #7f8c8d;
    margin-top: 18px;
    font-size: 13pt;
}

/* Image constraints */
img {
    max-width: 100% !important;
    height: auto !important;
    display: block;
    margin: 10px auto;
}

/* Figure constraints */
figure {
    max-width: 100% !important;
    margin: 10px 0;
    page-break-inside: avoid;
}

/* Table constraints */
table {
    border-collapse: collapse;
    width: 100%;
    max-width: 100%;
    margin: 12px 0;
    font-size: 8pt;
    table-layout: fixed;
    word-wrap: break-word;
}
th, td {
    border: 1px solid #bdc3c7;
    padding: 5px 8px;
    text-align: left;
    overflow-wrap: break-word;
    word-break: break-word;
}
th {
    background-color: #3498db;
    color: white;
    font-weight: bold;
}
tr:nth-child(even) {
    background-color: #f8f9fa;
}

/* Code block constraints */
code {
    background-color: #f4f4f4;
    padding: 1px 4px;
    border-radius: 3px;
    font-family: 'Monaco', 'Menlo', monospace;
    font-size: 8pt;
    word-break: break-all;
}
pre {
    background-color: #2d2d2d;
    color: #f8f8f2;
    padding: 12px;
    border-radius: 5px;
    overflow-x: hidden;
    font-size: 8pt;
    white-space: pre-wrap;
    word-wrap: break-word;
    max-width: 100%;
}
pre code {
    background-color: transparent;
    color: inherit;
    padding: 0;
    white-space: pre-wrap;
    word-wrap: break-word;
}

hr {
    border: none;
    border-top: 1px solid #bdc3c7;
    margin: 15px 0;
}
strong {
    color: #2c3e50;
}

/* Prevent overflow on any element */
* {
    max-width: 100%;
    box-sizing: border-box;
}

@page {
    margin: 0.6in;
    size: letter;
}
"""


HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <style>
    {css}
    </style>
</head>
<body>
{content}
</body>
</html>
"""


def convert_md_to_pdf(input_md_path: str, output_folder: str, output_filename: str) -> str:
    """
    Convert a markdown file to a styled PDF.

    Args:
        input_md_path: Path to the input markdown file
        output_folder: Directory where the PDF will be saved
        output_filename: Name of the output PDF file (should end with .pdf)

    Returns:
        Path to the generated PDF file
    """
    if not os.path.isfile(input_md_path):
        raise FileNotFoundError(f"Input file not found: {input_md_path}")

    if not output_filename.lower().endswith(".pdf"):
        output_filename += ".pdf"

    os.makedirs(output_folder, exist_ok=True)

    output_path = os.path.join(output_folder, output_filename)

    with open(input_md_path, "r", encoding="utf-8") as f:
        md_content = f.read()

    html_content = markdown2.markdown(
        md_content,
        extras=["tables", "fenced-code-blocks", "code-friendly"],
    )

    styled_html = HTML_TEMPLATE.format(css=CSS_STYLES, content=html_content)

    HTML(string=styled_html).write_pdf(output_path)

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Convert a markdown file to a styled PDF document.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python md_to_pdf.py report.md ./output report.pdf
    python md_to_pdf.py /path/to/input.md /path/to/output output_name.pdf
        """,
    )
    parser.add_argument("input_md_file", help="Path to the input markdown file")
    parser.add_argument("output_folder", help="Directory where the PDF will be saved")
    parser.add_argument("output_filename", help="Name of the output PDF file")

    args = parser.parse_args()

    try:
        output_path = convert_md_to_pdf(
            args.input_md_file,
            args.output_folder,
            args.output_filename,
        )
        print(f"PDF generated successfully: {output_path}")
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error generating PDF: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()