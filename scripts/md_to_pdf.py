import os
import subprocess
import sys
import tempfile
import markdown

def md_to_pdf(input_path, output_path=None):
    if not os.path.exists(input_path):
        print(f"Error: File '{input_path}' not found.")
        return

    if output_path is None:
        output_path = os.path.splitext(input_path)[0] + ".pdf"

    # Read the markdown file
    with open(input_path, 'r', encoding='utf-8') as f:
        md_text = f.read()

    # Convert to HTML
    html_content = markdown.markdown(md_text, extensions=['extra', 'codehilite', 'toc'])

    # GitHub-flavored-ish CSS
    css = """
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif, "Apple Color Emoji", "Segoe UI Emoji";
            font-size: 16px;
            line-height: 1.5;
            word-wrap: break-word;
            max-width: 980px;
            margin: 0 auto;
            padding: 45px;
        }
        h1, h2, h3, h4, h5, h6 {
            margin-top: 24px;
            margin-bottom: 16px;
            font-weight: 600;
            line-height: 1.25;
        }
        h1 { font-size: 2em; border-bottom: 1px solid #eaecef; padding-bottom: 0.3em; }
        h2 { font-size: 1.5em; border-bottom: 1px solid #eaecef; padding-bottom: 0.3em; }
        code {
            padding: 0.2em 0.4em;
            margin: 0;
            font-size: 85%;
            background-color: rgba(27,31,35,0.05);
            border-radius: 3px;
            font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
        }
        pre {
            padding: 16px;
            overflow: auto;
            font-size: 85%;
            line-height: 1.45;
            background-color: #f6f8fa;
            border-radius: 3px;
        }
        pre code {
            background-color: transparent;
            padding: 0;
        }
        blockquote {
            padding: 0 1em;
            color: #6a737d;
            border-left: 0.25em solid #dfe2e1;
            margin: 0 0 16px 0;
        }
        table {
            border-spacing: 0;
            border-collapse: collapse;
            margin-top: 0;
            margin-bottom: 16px;
            width: 100%;
        }
        table th, table td {
            padding: 6px 13px;
            border: 1px solid #dfe2e5;
        }
        table tr {
            background-color: #fff;
            border-top: 1px solid #c6cbd1;
        }
        table tr:nth-child(2n) {
            background-color: #f6f8fa;
        }
        img {
            max-width: 100%;
            box-sizing: content-box;
            background-color: #fff;
        }
        /* Alert styles */
        .admonition {
            padding: 15px;
            margin-bottom: 20px;
            border: 1px solid transparent;
            border-radius: 4px;
        }
    </style>
    """

    full_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        {css}
    </head>
    <body>
        {html_content}
    </body>
    </html>
    """

    # Save to temp HTML file
    with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode='w', encoding='utf-8') as tmp:
        tmp.write(full_html)
        tmp_path = tmp.name

    try:
        # Run Chrome headless to print to PDF
        chrome_cmd = [
            "google-chrome",
            "--headless",
            "--no-sandbox",
            "--disable-gpu",
            f"--print-to-pdf={output_path}",
            tmp_path
        ]
        
        result = subprocess.run(chrome_cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"Successfully created: {output_path}")
        else:
            print(f"Error during PDF generation: {result.stderr}")
            
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python md_to_pdf.py <input_md_file> [output_pdf_file]")
    else:
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else None
        md_to_pdf(input_file, output_file)
