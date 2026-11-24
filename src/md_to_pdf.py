"""
src/md_to_pdf.py
Converts writeup/final_writeup.md and all 6 300 DPI publication figures into an
executive, front-office publication PDF using headless Chromium (100% offline).
"""

import re
import base64
import shutil
import subprocess
from pathlib import Path
import markdown

ROOT_DIR = Path(__file__).resolve().parent.parent
WRITEUP_DIR = ROOT_DIR / "writeup"
FIGURES_DIR = ROOT_DIR / "figures"

HTML_OUT = WRITEUP_DIR / "_build_writeup.html"
PDF_OUT = WRITEUP_DIR / "nfl_big_data_bowl_2027_final_submission.pdf"

CANDIDATE_BROWSERS = [
    shutil.which("google-chrome"),
    shutil.which("chromium"),
    shutil.which("chromium-browser"),
    shutil.which("msedge"),
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
]


def encode_image_base64(img_path: Path) -> str:
    with open(img_path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/png;base64,{data}"


def latex_inline_to_html(expr: str) -> str:
    """Convert inline LaTeX math into crisp typographic HTML without network MathJax."""
    s = expr.strip()
    # Vectors & accents
    s = re.sub(r"\\vec\{([a-zA-Z])\}", r"<b>\1</b>", s)
    s = re.sub(r"\\hat\{\\tau\}", "<b>&tau;&#770;</b>", s)
    s = re.sub(r"\\dot\{([a-zA-Z])\}", r"<i>\1&#775;</i>", s)
    s = re.sub(r"\\ddot\{([a-zA-Z])\}", r"<i>\1&#776;</i>", s)
    # Text / operators
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\operatorname\{([^}]*)\}", r"\1", s)
    # Subscripts and superscripts: _{...}, _x, ^{...}, ^x (before replacing Greek entities)
    s = re.sub(r"_\{([^}]+)\}", r"<sub>\1</sub>", s)
    s = re.sub(r"_(\\[a-zA-Z]+|[a-zA-Z0-9])", r"<sub>\1</sub>", s)
    s = re.sub(r"\^\{([^}]+)\}", r"<sup>\1</sup>", s)
    s = re.sub(r"\^(\\[a-zA-Z]+|[a-zA-Z0-9])", r"<sup>\1</sup>", s)
    # Symbols
    replacements = {
        r"\kappa": "&kappa;",
        r"\tau": "&tau;",
        r"\beta": "&beta;",
        r"\Delta": "&Delta;",
        r"\approx": "&approx;",
        r"\le": "&le;",
        r"\ge": "&ge;",
        r"\times": "&times;",
        r"\cdot": "&middot;",
        r"\in": "&isin;",
        r"\{": "{",
        r"\}": "}",
        r"\qquad": "&nbsp;&nbsp;&nbsp;&nbsp;",
    }
    for k, v in replacements.items():
        s = s.replace(k, v)
    # Clean spacing around =, <, >
    s = s.replace("<", " &lt; ").replace(">", " &gt; ")
    # Restore HTML tags that got spaced
    for tag in ["b", "i", "sub", "sup"]:
        s = s.replace(f" &lt; {tag} &gt; ", f"<{tag}>").replace(f" &lt; /{tag} &gt; ", f"</{tag}>")
        s = s.replace(f"&lt; {tag} &gt;", f"<{tag}>").replace(f"&lt; /{tag} &gt;", f"</{tag}>")
    s = re.sub(r"\s+", " ", s).strip()
    return f'<span class="math-inline">{s}</span>'


def markdown_to_html(md_text: str) -> str:
    # 1. Replace images with base64 embedded figure containers
    def replace_img(match):
        alt = match.group(1)
        raw_src = match.group(2)
        fname = Path(raw_src).name
        target = FIGURES_DIR / fname
        if target.exists():
            b64 = encode_image_base64(target)
            return f'\n<div class="figure-container"><img src="{b64}" alt="{alt}"/><p class="figure-caption">{alt}</p></div>\n'
        return f"<p><strong>[Image: {alt}]</strong></p>"

    text = re.sub(r"!\[(.*?)\]\((.*?)\)", replace_img, md_text)

    # 2. Replace display equation block ($$...$$) with clean HTML fraction layout
    display_eq_html = (
        '<div class="eq-block">'
        '<i>&kappa;</i>(<i>t</i>) = '
        '<span class="frac"><span class="num">|<i>x&#775;y&#776;</i> &minus; <i>y&#775;x&#776;</i>|</span>'
        '<span class="den">(<i>x&#775;</i><sup>2</sup> + <i>y&#775;</i><sup>2</sup>)<sup>3/2</sup></span></span>'
        ', &nbsp;&nbsp;&nbsp;&nbsp; <i>a<sub>n</sub></i>(<i>t</i>) = <i>v</i>(<i>t</i>)<sup>2</sup> &middot; <i>&kappa;</i>(<i>t</i>)'
        '</div>'
    )
    text = re.sub(r"\$\$[^$]+\$\$", display_eq_html, text, flags=re.DOTALL)

    # 3. Protect inline math ($...$) before markdown parsing so underscores don't become <em>
    math_tokens = {}
    def protect_math(match):
        key = f"MATHTOKEN{len(math_tokens)}ENDTOKEN"
        math_tokens[key] = latex_inline_to_html(match.group(1))
        return key

    text = re.sub(r"\$([^$\n]+)\$", protect_math, text)

    # 4. Ensure blank line before list blocks so Python-Markdown parses <ul>/<ol> cleanly
    text = re.sub(r"([^\n])\n(\* |\d+\. )", r"\1\n\n\2", text)

    # 5. Unescape markdown asterisks used for significance stars (\*\*\* -> ***)
    text = text.replace(r"\*\*\*", "&#42;&#42;&#42;").replace(r"\*\*", "&#42;&#42;").replace(r"\*", "&#42;")

    # 6. Parse markdown using Python-Markdown
    html_body = markdown.markdown(text, extensions=["tables", "fenced_code", "sane_lists"])

    # 6. Restore protected math tokens
    for key, val in math_tokens.items():
        html_body = html_body.replace(key, val)

    # 7. Wrap tables in .table-container for styling & page-break control
    html_body = html_body.replace("<table>", '<div class="table-container"><table>')
    html_body = html_body.replace("</table>", "</table></div>")

    return html_body


def build_pdf():
    print("Reading writeup/final_writeup.md...")
    with open(WRITEUP_DIR / "final_writeup.md", "r", encoding="utf-8") as f:
        md_text = f.read()

    html_body = markdown_to_html(md_text)

    full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>NFL Big Data Bowl 2027 — Final Submission</title>
<style>
    @page {{
        size: letter;
        margin: 0.48in 0.52in 0.48in 0.52in;
        @bottom-right {{
            content: "Page " counter(page);
            font-size: 8.5pt;
            color: #718096;
        }}
    }}
    body {{
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
        color: #1A202C;
        line-height: 1.44;
        font-size: 9.3pt;
        background: #FFFFFF;
        margin: 0;
        padding: 0;
    }}
    h1 {{
        color: #0B162A;
        font-size: 18pt;
        font-weight: 800;
        border-bottom: 3px solid #0072B2;
        padding-bottom: 6px;
        margin-top: 0;
        margin-bottom: 10px;
        letter-spacing: -0.3px;
        line-height: 1.2;
    }}
    h2 {{
        color: #0B162A;
        font-size: 12.5pt;
        font-weight: 700;
        border-bottom: 1.5px solid #CBD5E0;
        padding-bottom: 2px;
        margin-top: 13px;
        margin-bottom: 6px;
        page-break-after: avoid;
    }}
    h3 {{
        color: #1E293B;
        font-size: 10.5pt;
        font-weight: 700;
        margin-top: 10px;
        margin-bottom: 4px;
        page-break-after: avoid;
    }}
    p {{
        margin-top: 0;
        margin-bottom: 5.5px;
        text-align: justify;
    }}
    a {{
        color: #0072B2;
        text-decoration: none;
    }}
    strong {{
        color: #0B162A;
    }}
    code {{
        background: #F1F5F9;
        color: #0F172A;
        font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, Courier, monospace;
        font-size: 8.6pt;
        padding: 1px 4px;
        border-radius: 3px;
        border: 1px solid #E2E8F0;
    }}
    hr {{
        border: none;
        border-top: 1px solid #E2E8F0;
        margin: 12px 0;
    }}
    .math-inline {{
        font-family: "Cambria Math", "Times New Roman", Georgia, serif;
        font-size: 10pt;
    }}
    .eq-block {{
        background: #F8FAFC;
        border-left: 3px solid #0072B2;
        padding: 8px 14px;
        margin: 10px auto;
        text-align: center;
        font-family: "Cambria Math", "Times New Roman", Georgia, serif;
        font-size: 11pt;
        page-break-inside: avoid;
    }}
    .frac {{
        display: inline-block;
        vertical-align: middle;
        text-align: center;
        margin: 0 2px;
    }}
    .frac .num {{
        display: block;
        border-bottom: 1px solid #1A202C;
        padding: 0 4px 1px 4px;
        font-size: 9.8pt;
    }}
    .frac .den {{
        display: block;
        padding: 1px 4px 0 4px;
        font-size: 9.8pt;
    }}
    .table-container {{
        width: 100%;
        margin: 10px 0;
        page-break-inside: avoid;
    }}
    table {{
        width: 100%;
        border-collapse: collapse;
        font-size: 7.9pt;
        margin-bottom: 6px;
    }}
    th {{
        background: #0B162A;
        color: #FFFFFF;
        font-weight: 700;
        text-align: left;
        padding: 5px 6px;
        border: 1px solid #0B162A;
    }}
    td {{
        padding: 4px 6px;
        border: 1px solid #CBD5E0;
        vertical-align: top;
    }}
    tr:nth-child(even) td {{
        background: #F8FAFC;
    }}
    .figure-container {{
        width: 100%;
        text-align: center;
        margin: 5px 0;
        page-break-inside: avoid;
    }}
    .figure-container img {{
        max-width: 91%;
        height: auto;
        border: 1px solid #CBD5E0;
        border-radius: 4px;
    }}
    .figure-caption {{
        font-size: 8.1pt;
        color: #334155;
        font-weight: 600;
        margin-top: 3px;
        margin-bottom: 4px;
        text-align: center;
    }}
    ul, ol {{
        margin-top: 2px;
        margin-bottom: 8px;
        padding-left: 20px;
    }}
    li {{
        margin-bottom: 3.5px;
    }}
</style>
</head>
<body>
{html_body}
</body>
</html>"""

    WRITEUP_DIR.mkdir(parents=True, exist_ok=True)
    html_target = HTML_OUT
    pdf_target = PDF_OUT
    with open(html_target, "w", encoding="utf-8") as f:
        f.write(full_html)

    print(f"Standalone HTML generated at {html_target}")

    browsers = [Path(b) for b in CANDIDATE_BROWSERS if b and Path(b).exists()]
    if not browsers:
        raise FileNotFoundError("No Chromium-based browser (Chrome/Edge) found for PDF compilation.")

    success = False
    for browser_bin in browsers:
        print(f"Attempting PDF compilation with {browser_bin.name}...")
        cmd = [
            str(browser_bin),
            "--headless=new",
            "--disable-gpu",
            "--no-sandbox",
            "--no-pdf-header-footer",
            f"--print-to-pdf={pdf_target}",
            str(html_target),
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            if pdf_target.exists() and pdf_target.stat().st_size > 1000:
                success = True
                print(f"PDF successfully compiled using {browser_bin.name}!")
                break
        except Exception as e:
            print(f"Compilation with {browser_bin.name} failed: {e}")

    if html_target.exists():
        html_target.unlink()

    if not success:
        raise RuntimeError("Failed to compile PDF with available browsers.")

    pdf_size_mb = pdf_target.stat().st_size / (1024 * 1024)
    print(f"PDF compiled at {pdf_target} ({pdf_size_mb:.2f} MB)")


if __name__ == "__main__":
    build_pdf()
