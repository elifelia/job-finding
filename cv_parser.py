"""
CV Parser — extracts text from PDF and uses Claude API to parse into structured JSON.
Handles both English and Indonesian CVs; normalizes all output to English.
"""

import json
import re
import os
from pathlib import Path

import pymupdf as fitz  # PyMuPDF (use pymupdf to avoid deprecation warning)
import anthropic

# Configure Claude API with environment variable
api_key = os.getenv("ANTHROPIC_API_KEY")
if api_key:
    print(f"[CV Parser] Claude API configured with ANTHROPIC_API_KEY")
else:
    print(f"[CV Parser] Warning: ANTHROPIC_API_KEY not set")


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract raw text from a PDF file using PyMuPDF."""
    doc = fitz.open(pdf_path)
    pages = []
    for page in doc:
        text = page.get_text("text")
        if text.strip():
            pages.append(text)
    doc.close()
    return "\n\n---PAGE BREAK---\n\n".join(pages)


PARSE_PROMPT = """You are a CV/resume parser. Extract structured information from the CV text below.

IMPORTANT RULES:
1. Output ONLY valid JSON — no explanation, no markdown, no code fences.
2. If the CV is in Indonesian or any other language, translate ALL values to English.
3. Normalize job titles to standard English equivalents (e.g. "Manajer Pemasaran" → "Marketing Manager").
4. If a field is missing or unclear, use null.
5. Infer seniority level from years of experience and job titles.

Extract into this exact JSON structure:
{
  "name": "Full Name",
  "email": "email@example.com",
  "phone": "+1234567890 or null",
  "location": "City, Country or null",
  "linkedin": "url or null",
  "summary": "2-3 sentence professional summary in English",
  "seniority_level": "junior | mid | senior | lead | manager | director | executive",
  "years_of_experience": 5,
  "current_title": "Most recent job title in English",
  "industries": ["Tech", "Finance", "etc — infer from work history"],
  "skills": {
    "technical": ["Python", "SQL", "etc"],
    "soft": ["Leadership", "Communication", "etc"],
    "tools": ["Figma", "Jira", "etc"],
    "languages": [{"language": "English", "proficiency": "native | fluent | professional | conversational | basic"}]
  },
  "work_experience": [
    {
      "title": "Job Title in English",
      "company": "Company Name",
      "location": "City, Country or Remote",
      "start_date": "YYYY-MM or YYYY",
      "end_date": "YYYY-MM or YYYY or present",
      "duration_months": 24,
      "description": "Key responsibilities and achievements in English",
      "skills_used": ["Python", "Team Management", "etc"]
    }
  ],
  "education": [
    {
      "degree": "Bachelor of Science in Computer Science",
      "institution": "University Name",
      "location": "City, Country",
      "graduation_year": 2018,
      "gpa": "3.8/4.0 or null"
    }
  ],
  "certifications": ["AWS Certified Solutions Architect", "etc"],
  "preferred_locations": ["Remote", "Singapore", "etc — infer if mentioned"],
  "open_to_remote": true,
  "raw_language": "en | id | other — the original language of the CV"
}

CV TEXT:
{cv_text}
"""


def parse_cv_with_claude(cv_text: str, api_key: str | None = None) -> dict:
    """
    Send CV text to Claude API and get structured JSON back.
    api_key: optional; if None, uses ANTHROPIC_API_KEY env var.
    """
    if not api_key:
        api_key = os.getenv("ANTHROPIC_API_KEY")

    if not api_key:
        raise ValueError("No ANTHROPIC_API_KEY found. Set ANTHROPIC_API_KEY environment variable.")

    client = anthropic.Anthropic(api_key=api_key)
    prompt = PARSE_PROMPT.replace("{cv_text}", cv_text)

    message = client.messages.create(
        model="claude-haiku-5-5",
        max_tokens=16384,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    # Handle both TextBlock and ThinkingBlock responses
    raw = ""
    for block in message.content:
        if hasattr(block, 'text'):
            raw = block.text.strip()
            break

    if not raw:
        raise ValueError("No text response from Claude")

    # Strip markdown code fences if model accidentally adds them
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"Claude returned invalid JSON: {e}\n\nRaw output:\n{raw}")


def parse_cv(pdf_path: str, api_key: str | None = None) -> dict:
    """
    Full pipeline: PDF path → structured CV dict.

    Returns dict with all fields plus:
      _source_file: original filename
      _char_count: length of extracted text
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"CV file not found: {pdf_path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a PDF file, got: {path.suffix}")

    print(f"[CV Parser] Extracting text from {path.name}...")
    cv_text = extract_text_from_pdf(str(path))

    if len(cv_text.strip()) < 100:
        raise ValueError("Extracted text too short — the PDF may be image-based or empty.")

    print(f"[CV Parser] Extracted {len(cv_text)} characters. Sending to Claude API...")
    parsed = parse_cv_with_claude(cv_text, api_key=api_key)

    parsed["_source_file"] = path.name
    parsed["_char_count"] = len(cv_text)

    print(f"[CV Parser] Done. Detected language: {parsed.get('raw_language', 'unknown')}")
    if parsed.get("raw_language") != "en":
        print(f"[CV Parser] Non-English CV detected — all fields normalized to English.")

    return parsed


def save_parsed_cv(parsed: dict, output_dir: str = "data/outputs") -> str:
    """Save parsed CV JSON to disk. Returns the output file path."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    source = parsed.get("_source_file", "cv").replace(".pdf", "")
    safe_name = re.sub(r"[^\w\-]", "_", source)
    out_path = out_dir / f"{safe_name}_parsed.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(parsed, f, indent=2, ensure_ascii=False)

    print(f"[CV Parser] Saved to {out_path}")
    return str(out_path)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python cv_parser.py path/to/cv.pdf")
        sys.exit(1)

    pdf_path = sys.argv[1]
    result = parse_cv(pdf_path)
    output_path = save_parsed_cv(result, output_dir="../../data/outputs")

    print("\n=== PARSED CV SUMMARY ===")
    print(f"Name:        {result.get('name')}")
    print(f"Title:       {result.get('current_title')}")
    print(f"Seniority:   {result.get('seniority_level')}")
    print(f"Experience:  {result.get('years_of_experience')} years")
    print(f"Industries:  {', '.join(result.get('industries', []))}")
    print(f"Tech skills: {', '.join(result.get('skills', {}).get('technical', [])[:8])}")
    print(f"Language:    {result.get('raw_language')}")
    print(f"\nFull output: {output_path}")
