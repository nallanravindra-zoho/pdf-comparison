import anthropic
import requests
import uuid
import json
import re
import base64
from datetime import datetime
import google.generativeai as genai
from fastapi import FastAPI, BackgroundTasks, Request
from fastapi.responses import JSONResponse
from weasyprint import HTML
import time
import concurrent.futures
import threading
import os
from dotenv import load_dotenv

app = FastAPI()

# ─────────────────────────────────────────────
# WEASYPRINT WARMUP
# Pre-loads fonts at startup so first PDF is fast
# ─────────────────────────────────────────────
def _warmup_weasyprint():
    try:
        HTML(string="<html><body><p>warmup</p></body></html>").write_pdf()
        print("WeasyPrint warmed up")
    except Exception as e:
        print(f"WeasyPrint warmup failed: {e}")

threading.Thread(target=_warmup_weasyprint, daemon=True).start()

# Loads .env file locally — ignored in Cloud Run (uses env vars directly)
load_dotenv()
# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────


CLIENT_ID      = os.environ["CLIENT_ID"]
CLIENT_SECRET  = os.environ["CLIENT_SECRET"]
REFRESH_TOKEN  = os.environ["REFRESH_TOKEN"]
ZOHO_BASE_URL  = os.environ.get("ZOHO_BASE_URL", "https://www.zohoapis.com")
CLAUDE_API_KEY = os.environ["CLAUDE_API_KEY"]
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
PPO_PDF_FIELD        = os.environ.get("PPO_PDF_FIELD", "PPO_PDF_FIELD")
VQ_PDF_FIELD         = os.environ.get("VQ_PDF_FIELD", "VQ_PDF")
ACCESS_TOKEN_URL     = os.environ.get("ACCESS_TOKEN_URL", "https://accounts.zoho.com/oauth/v2/token")

# ─────────────────────────────────────────────
# IN-MEMORY STORES
# ─────────────────────────────────────────────
jobs      = {}
last_poll = {}

# ─────────────────────────────────────────────
# CACHES — avoid repeated API calls
# ─────────────────────────────────────────────
_gemini_model_cache = None

# OPT 3: Zoho token cache — reuse for 55 min instead of fetching every job
_token_cache = {"token": None, "expires_at": 0}

# ─────────────────────────────────────────────
# PROMPT LOADER — reads from Zoho CRM AIPrompts module, 5-min cache
#
# Module API name : AIPrompts
# Field API names : GEMINI_PROMPT  (multiline text)
#                   CLAUDE_PROMPT  (multiline text)
#
# The loader fetches the first record from AIPrompts, extracts both field
# values in a single CRM call, and caches them for _PROMPT_CACHE_TTL seconds.
# Falls back to stale cache if CRM is unreachable.
# ─────────────────────────────────────────────
_PROMPT_CACHE_TTL = 300          # seconds — lower to 60 during active prompt tuning
_crm_prompt_cache: dict = {      # keys: "gemini" | "claude"
    "gemini": (None, 0),
    "claude": (None, 0),
}

def _sanitise_prompt(text: str) -> str:
    """Strip invisible/problematic unicode characters that CRM editors silently
    inject (BOM U+FEFF, word-joiner U+2060, zero-width space U+200B, etc.).
    These cause JSONDecodeError when they leak into Claude's output."""
    INVISIBLE = (
        "\ufeff",   # BOM
        "\u2060",   # word joiner
        "\u200b",   # zero-width space
        "\u200c",   # zero-width non-joiner
        "\u200d",   # zero-width joiner
        "\u00a0",   # non-breaking space → replace with regular space
        "\u2028",   # line separator
        "\u2029",   # paragraph separator
    )
    for ch in INVISIBLE:
        text = text.replace(ch, " " if ch == "\u00a0" else "")
    return text.strip()


def _fetch_prompts_from_crm() -> tuple[str, str]:
    """Fetch GEMINI_PROMPT and CLAUDE_PROMPT from the Zoho CRM AIPrompts module.
    Returns (gemini_prompt, claude_prompt).
    Raises RuntimeError if the module is empty or fields are blank."""
    token = get_access_token()
    url   = f"{ZOHO_BASE_URL}/crm/v3/AIPrompts"
    headers = {"Authorization": f"Zoho-oauthtoken {token}"}
    params  = {"fields": "GEMINI_PROMPT,CLAUDE_PROMPT", "per_page": 1}

    r = requests.get(url, headers=headers, params=params, timeout=20)
    print(f"[prompts] AIPrompts CRM fetch status: {r.status_code}")
    r.raise_for_status()

    data = r.json().get("data", [])
    if not data:
        raise RuntimeError(
            "AIPrompts module is empty — please create a record with "
            "GEMINI_PROMPT and CLAUDE_PROMPT field values."
        )

    record = data[0]
    gemini_prompt = _sanitise_prompt(record.get("GEMINI_PROMPT") or "")
    claude_prompt = _sanitise_prompt(record.get("CLAUDE_PROMPT") or "")

    if not gemini_prompt:
        raise RuntimeError("GEMINI_PROMPT field in AIPrompts CRM record is blank.")
    if not claude_prompt:
        raise RuntimeError("CLAUDE_PROMPT field in AIPrompts CRM record is blank.")

    print(f"[prompts] Loaded from CRM AIPrompts — "
          f"Gemini: {len(gemini_prompt)} chars, Claude: {len(claude_prompt)} chars")
    return gemini_prompt, claude_prompt


def load_gemini_prompt() -> str:
    """Return the Gemini extraction prompt, using the 5-min cache."""
    now  = time.time()
    text, fetched_at = _crm_prompt_cache["gemini"]
    if text and (now - fetched_at) < _PROMPT_CACHE_TTL:
        return text
    return _refresh_prompt_cache()[0]


def load_claude_prompt() -> str:
    """Return the Claude matching prompt, using the 5-min cache."""
    now  = time.time()
    text, fetched_at = _crm_prompt_cache["claude"]
    if text and (now - fetched_at) < _PROMPT_CACHE_TTL:
        return text
    return _refresh_prompt_cache()[1]


def _refresh_prompt_cache() -> tuple[str, str]:
    """Fetch both prompts from CRM, update the cache, and return (gemini, claude).
    Falls back to stale cache values if CRM is unreachable."""
    now = time.time()
    try:
        gemini_prompt, claude_prompt = _fetch_prompts_from_crm()
        _crm_prompt_cache["gemini"] = (gemini_prompt, now)
        _crm_prompt_cache["claude"] = (claude_prompt, now)
        return gemini_prompt, claude_prompt
    except Exception as e:
        print(f"[prompts] ⚠️  CRM fetch failed: {e}")
        gemini_text, _ = _crm_prompt_cache["gemini"]
        claude_text, _ = _crm_prompt_cache["claude"]
        if gemini_text and claude_text:
            print("[prompts] Using stale cached prompts")
            return gemini_text, claude_text
        raise RuntimeError(
            "Could not load prompts from CRM AIPrompts module and no cache available."
        ) from e


# ─────────────────────────────────────────────
# GLOBAL ERROR HANDLER
# ─────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"error": str(exc), "type": type(exc).__name__}
    )


# ─────────────────────────────────────────────
# 1. ZOHO AUTH — with token caching
# ─────────────────────────────────────────────
def get_access_token():
    global _token_cache

    # OPT 3: Return cached token if still valid (60s buffer before expiry)
    if _token_cache["token"] and time.time() < _token_cache["expires_at"] - 60:
        print("✅ Using cached access token")
        return _token_cache["token"]

    params = {
        "refresh_token": REFRESH_TOKEN,
        "client_id":     CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "grant_type":    "refresh_token"
    }
    print("GOT ACCES TOKEN URL is :",ACCESS_TOKEN_URL)
    r = requests.post(ACCESS_TOKEN_URL, params=params, timeout=30)
    token = r.json().get("access_token")
    if not token:
        raise Exception("Failed to get access token: " + str(r.json()))

    # Cache for 55 minutes (tokens last 1 hour)
    _token_cache["token"]      = token
    _token_cache["expires_at"] = time.time() + 3300
    print("✅ Fresh access token obtained and cached")
    return token


# ─────────────────────────────────────────────
# 2. FETCH ZOHO QUOTE
# ─────────────────────────────────────────────
def fetch_zoho_quote(quote_id, access_token):
    url = f"{ZOHO_BASE_URL}/crm/v3/Quotes/{quote_id}"
    headers = {"Authorization": f"Zoho-oauthtoken {access_token}"}
    response = requests.get(url, headers=headers, timeout=30)
    print(f"Zoho response status: {response.status_code}")
    response.raise_for_status()
    quote = response.json()["data"][0]
    print(f"✅ Quote fetched: {quote.get('Subject', quote_id)}")
    return quote


# ─────────────────────────────────────────────
# 3. DOWNLOAD FILE FROM ZOHO
# ─────────────────────────────────────────────
def download_zoho_file(file_id, token):
    headers = {"Authorization": f"Zoho-oauthtoken {token}"}
    url = f"{ZOHO_BASE_URL}/crm/v3/files?id={file_id}"
    r = requests.get(url, headers=headers, timeout=60)
    r.raise_for_status()
    print(f"✅ Downloaded file {file_id} ({len(r.content)} bytes)")
    return r.content


# ─────────────────────────────────────────────
# 4. CHECK FOR EXISTING REPORT IN ATTACHMENTS
# ─────────────────────────────────────────────
def check_existing_report(quote_id, token,report_name=None):
    url     = f"{ZOHO_BASE_URL}/crm/v3/Quotes/{quote_id}/Attachments"
    headers = {"Authorization": f"Zoho-oauthtoken {token}"}
    params  = {"fields": "id,File_Name,Created_Time,Size"}
    r       = requests.get(url, headers=headers, params=params, timeout=30)

    print(f"📎 Check attachments status: {r.status_code}")
    print(f"📎 Check attachments response: {r.text[:300]}")

    if r.status_code in (204, 404):
        print("📎 No attachments found")
        return None

    r.raise_for_status()
    data = r.json().get("data", [])
    print(f"📎 Found {len(data)} attachments: {[a.get('File_Name') for a in data]}")

    for attachment in data:
        fname = attachment.get("File_Name", "")
        if report_name and fname == report_name:
            print(f"Found existing report: {fname}")
            return attachment
        elif not report_name and fname.startswith("DOC_Compare_"):
            print(f"Found existing report: {fname}")
            return attachment

    return None


# ─────────────────────────────────────────────
# 5. ATTACH PDF TO ZOHO QUOTE
# ─────────────────────────────────────────────
def attach_pdf_to_quote(quote_id, pdf_bytes, token,report_name="SKU_Audit_Report.pdf"):
    existing = check_existing_report(quote_id, token, report_name)
    if existing:
        del_url = f"{ZOHO_BASE_URL}/crm/v3/Quotes/{quote_id}/Attachments/{existing['id']}"
        headers = {"Authorization": f"Zoho-oauthtoken {token}"}
        r = requests.delete(del_url, headers=headers, timeout=30)
        print(f"Deleted old report: {r.status_code}")

    url     = f"{ZOHO_BASE_URL}/crm/v3/Quotes/{quote_id}/Attachments"
    headers = {"Authorization": f"Zoho-oauthtoken {token}"}
    files   = {"file": (report_name, pdf_bytes, "application/pdf")}  # uses dynamic name

    print(f"Uploading: {report_name}")
    r = requests.post(url, headers=headers, files=files, timeout=60)
    print(f"Upload status: {r.status_code} {r.text[:200]}")

    if r.status_code == 400:
        return attach_via_filestore(quote_id, pdf_bytes, token, report_name)

    r.raise_for_status()
    attachment_id = r.json().get("data", [{}])[0].get("details", {}).get("id")
    print(f"Attached as {report_name}. ID: {attachment_id}")
    return attachment_id


# ─────────────────────────────────────────────
# FALLBACK — upload via file store then link
# ─────────────────────────────────────────────
def attach_via_filestore(quote_id, pdf_bytes, token,report_name="SKU_Audit_Report.pdf"):
    headers    = {"Authorization": f"Zoho-oauthtoken {token}"}
    upload_url = f"{ZOHO_BASE_URL}/crm/v3/files"
    files      = {"file": (report_name, pdf_bytes, "application/pdf")}

    r = requests.post(upload_url, headers=headers, files=files, timeout=60)
    r.raise_for_status()

    file_id = r.json().get("data", [{}])[0].get("details", {}).get("id")
    if not file_id:
        raise Exception("No file_id from filestore: " + r.text[:200])

    attach_url = f"{ZOHO_BASE_URL}/crm/v3/Quotes/{quote_id}/Attachments"
    r = requests.post(
        attach_url,
        headers={**headers, "Content-Type": "application/json"},
        json={"attachments": [{"id": file_id}]},
        timeout=30
    )
    r.raise_for_status()
    attachment_id = r.json().get("data", [{}])[0].get("details", {}).get("id")
    print(f"Attached via filestore as {report_name}. ID: {attachment_id}")
    return attachment_id


# ─────────────────────────────────────────────
# 6. FORMAT ZOHO QUOTE LINE ITEMS
# ─────────────────────────────────────────────
def format_zoho_quote(quote: dict) -> str:
    lines = []

    # ── Header / reference fields ─────────────────────────────
    quote_currency        = quote.get("Currency", {})
    partner_po_currency   = quote.get("Partner_PO_Currency", "N/A")
    vendor_quote_currency = quote.get("Vendor_Quote_Currency", "N/A")
    exchange_rate         = quote.get("Exchange_Rate", "N/A")
    reseller              = quote.get("Reseller", "N/A")
    partner_po_ref        = quote.get("Partner_PO_Ref", "N/A")
    vendor                = quote.get("Vendor", "N/A")
    vendor_quote_ref      = quote.get("Vendor_Quote_Ref", "N/A")

    lines.append("## QUOTE HEADER")
    lines.append(f"  Quote_Currency        : {quote_currency}")
    lines.append(f"  Partner_PO_Currency   : {partner_po_currency}")
    lines.append(f"  Vendor_Quote_Currency : {vendor_quote_currency}")
    lines.append(f"  Exchange_Rate         : {exchange_rate}")
    lines.append(f"  Reseller              : {reseller}")
    lines.append(f"  Partner_PO_Ref        : {partner_po_ref}")
    lines.append(f"  Vendor                : {vendor}")
    lines.append(f"  Vendor_Quote_Ref      : {vendor_quote_ref}")
    lines.append("")

    # ── Line items ────────────────────────────────────────────
    items = quote.get("Quoted_Items", [])
    print(f"✅ Zoho quote has {len(items)} line items")
    lines.append("## LINE ITEMS")
    for i, item in enumerate(items, 1):
        product_name_field = item.get("Product_Name") or {}
        sku          = product_name_field.get("Product_Code") or product_name_field.get("name") or "N/A"
        desc         = item.get("Description", "N/A")
        qty          = item.get("Quantity", "N/A")
        buy_price_zq = item.get("Buy_Price", "N/A")
        list_price_zq= item.get("List_Price", "N/A")

        lines.append(f"  {i}. SKU          : {sku}")
        lines.append(f"     Description  : {desc}")
        lines.append(f"     Quantity     : {qty}")
        lines.append(f"     buy_price_zq : {buy_price_zq}")
        lines.append(f"     list_price_zq: {list_price_zq}")
        lines.append("")
    return "\n".join(lines)


# ─────────────────────────────────────────────
# 7. GEMINI MODEL SELECTION (cached)
# ─────────────────────────────────────────────
def get_gemini_model() -> str:
    global _gemini_model_cache
    if _gemini_model_cache:
        print(f"✅ Using cached Gemini model: {_gemini_model_cache}")
        return _gemini_model_cache

    genai.configure(api_key=GEMINI_API_KEY)
    try:
        models = [
            m.name for m in genai.list_models(
                request_options={"timeout": 10}
            )
            if "generateContent" in m.supported_generation_methods
        ]
        for m in models:
            if "gemini-1.5-flash" in m:
                _gemini_model_cache = m
                print(f"✅ Selected + cached: {m}")
                return _gemini_model_cache
        _gemini_model_cache = models[0]
        print(f"✅ Fallback + cached: {_gemini_model_cache}")
        return _gemini_model_cache
    except Exception as e:
        print(f"⚠️  list_models failed ({e}), using gemini-1.5-flash")
        _gemini_model_cache = "models/gemini-1.5-flash"
        return _gemini_model_cache


# ─────────────────────────────────────────────
# 8. GEMINI PDF EXTRACTION
#    OPT 1: Returns compact JSON string
#    (fewer tokens to Claude vs verbose text)
# ─────────────────────────────────────────────
def extract_pdf_gemini(pdf_bytes: bytes, label: str, model_name: str, job_id: str = None, prompt: str = "") -> str:
    """Extract header fields AND line items from a PDF using Gemini.

    The updated Gemini prompt returns:
      { "header": { reseller_name, partner_po_ref, vendor_name, vendor_quote_ref },
        "line_items": [ { line_num, sku, description, quantity, list_unit_price }, ... ] }

    Legacy flat-array responses are still handled gracefully.
    Returns a formatted text block with HEADER FIELDS and LINE ITEMS sections
    for Claude to consume.
    """
    if job_id and is_cancelled(job_id):
        print(f"🔍 Cancelled before {label} extraction")
        raise Exception("Job cancelled by user")

    model = genai.GenerativeModel(
        model_name=model_name,
        generation_config={"temperature": 0, "response_mime_type": "application/json"}
    )

    print(f"🔍 Extracting {label} via Gemini ({model_name})...")
    response = model.generate_content(
        [{"mime_type": "application/pdf", "data": pdf_bytes}, prompt],
        request_options={"timeout": 120}
    )

    raw = response.text
    print(f"✅ Gemini {label}: {len(raw)} chars")

    try:
        clean = raw.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(clean)

        # ── Detect new format vs legacy flat array ──────────────
        if isinstance(parsed, dict) and "line_items" in parsed:
            # New format: { "header": {...}, "line_items": [...] }
            header = parsed.get("header") or {}
            items  = parsed.get("line_items") or []
        elif isinstance(parsed, list):
            # Legacy flat array — no header fields available
            header = {}
            items  = parsed
        else:
            raise ValueError(f"Unexpected Gemini response structure: {type(parsed)}")

        print(f"✅ Extracted {len(items)} items from {label}")
        print(f"   Header fields: {header}")

        lines = [f"## {label}"]

        # Header section — always emit all four keys so Claude has them
        lines.append("### HEADER FIELDS")
        lines.append(f"  reseller_name    : {header.get('reseller_name') or 'null'}")
        lines.append(f"  partner_po_ref   : {header.get('partner_po_ref') or 'null'}")
        lines.append(f"  vendor_name      : {header.get('vendor_name') or 'null'}")
        lines.append(f"  vendor_quote_ref : {header.get('vendor_quote_ref') or 'null'}")
        lines.append("")

        # Line items section
        lines.append("### LINE ITEMS")
        for item in items:
            lines.append(
                f"  {item.get('line_num','')}. SKU: {item.get('sku','N/A')} | "
                f"Desc: {item.get('description','N/A')} | "
                f"Qty: {item.get('quantity','N/A')} | "
                f"list_unit_price: {item.get('list_unit_price','N/A')}"
            )

        return "\n".join(lines)

    except (json.JSONDecodeError, ValueError) as e:
        print(f"⚠️  Gemini JSON parse error for {label}: {e}")
        return raw


# ─────────────────────────────────────────────
# 10. CLAUDE COMPARISON
#     OPT 2: Auto model selection based on size
#     OPT 1: Receives compact JSON (fewer tokens)
# ─────────────────────────────────────────────

def run_comparison(zoho_text: str, ppo_text: str, vq_text: str, job_id: str = None) -> dict:
    if job_id and is_cancelled(job_id):
        raise Exception("Job cancelled by user")

    # Always use Sonnet with temperature=0 for consistency
    model_name    = "claude-sonnet-4-6"
    max_tokens    = 32000
    matching_prompt = load_claude_prompt()
    print(f"[claude] Prompt length: {len(matching_prompt)} chars")
    print(f"[claude] Prompt start: {repr(matching_prompt[:80])}")
    print(f"[claude] Prompt end:   {repr(matching_prompt[-80:])}")
    print(f"Streaming from Claude ({model_name})...")

    client    = anthropic.Anthropic(api_key=CLAUDE_API_KEY)
    full_text = ""

    with client.messages.stream(
        model=model_name,
        max_tokens=max_tokens,
        temperature=0,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": "## ZOHO QUOTE (ZQ):\n\n" + zoho_text + "\n\n---"},
                {"type": "text", "text": "## VENDOR QUOTE (VQ) JSON:\n\n" + vq_text + "\n\n---"},
                {"type": "text", "text": "## Partner PO (PO):\n\n" + ppo_text + "\n\n---"},
                {"type": "text", "text": matching_prompt}
            ]
        }]
    ) as stream:
        for text_chunk in stream.text_stream:
            full_text += text_chunk
            if job_id and is_cancelled(job_id):
                print(f"[{job_id}] Cancelled during Claude streaming")
                raise Exception("Job cancelled during Claude streaming")

        # get_final_message() MUST stay inside the with block.
        # The SDK tears down the stream on context exit — calling it
        # outside returns an empty/incomplete object in newer SDK versions.
        message = stream.get_final_message()
        stop_reason = message.stop_reason
        print(f"Claude response: {len(full_text)} chars | Stop: {stop_reason}")

        if stop_reason == "max_tokens":
            raise Exception("Claude truncated — increase max_tokens")

        print(f"CLAUDE RESPONSE: {full_text}")

        # Strip markdown fences then extract the outermost JSON object.
        # Using regex instead of bare json.loads so any stray leading character
        # (BOM, word-joiner, prose preamble) cannot cause a char-0 parse failure.
        clean = full_text.replace("```json", "").replace("```", "").strip()
        json_match = re.search(r'\{.*\}', clean, re.DOTALL)
        if not json_match:
            print(f"[claude] FULL RAW RESPONSE:\n{full_text}")
            raise Exception(
                f"No JSON object found in Claude response. "
                f"stop_reason={stop_reason}, "
                f"full_text_len={len(full_text)}, "
                f"raw_repr={repr(full_text[:300])}"
            )
        return json.loads(json_match.group())


# ─────────────────────────────────────────────
# 11. GENERATE PDF REPORT
# Uses WeasyPrint — landscape A4, colour pills
# ─────────────────────────────────────────────
def generate_pdf_report(result: dict, quote_subject: str, job_id: str = None, initiated_by: str = "") -> bytes:
    if job_id and is_cancelled(job_id):
        raise Exception("Job cancelled by user")

    print("Generating PDF report...")
    t0 = time.time()

    def status_badge(status):
        if not status or status == "-":
            return '<span class="pill pill-na">N/A</span>'
        s = status.lower()
        # Substring matching (checked in this order) so this works for qty statuses
        # ("Match"/"Needs Review"/"Mismatch"), price statuses, and header statuses
        # ("Not Found"). "mismatch" must be checked before "match".
        if "mismatch"  in s: return '<span class="pill pill-miss">Mismatch</span>'
        if "not found" in s: return '<span class="pill pill-review">Not Found</span>'
        if "review"    in s: return '<span class="pill pill-review">Review</span>'
        if "match"     in s: return '<span class="pill pill-match">Match</span>'
        return f'<span class="pill pill-na">{status}</span>'

    fc = (result.get("final_call") or "").upper()
    if "CLEAR" in fc:
        banner_bg, banner_border = "#d1fae5", "#10b981"
    elif "HOLD" in fc:
        banner_bg, banner_border = "#fee2e2", "#ef4444"
    else:
        banner_bg, banner_border = "#fef3c7", "#f59e0b"

    fc_details = "".join([f"<li>{d}</li>" for d in (result.get("final_call_detail") or [])])

    # ── Currency overview block ──────────────────────────────
    currencies     = result.get("currencies_detected") or {}
    qc             = currencies.get("quote_currency") or "—"
    ppc            = currencies.get("partner_po_currency") or "—"
    vqc            = currencies.get("vendor_quote_currency") or "—"
    currency_notes = currencies.get("notes") or ""

    currency_block = f"""<div class="card currency-card">
        <div class="card-title">Currency Overview</div>
        <div class="currency-row">
          <div class="currency-item"><span class="currency-tag-label">Zoho Quote</span><span class="currency-tag-value">{qc}</span></div>
          <div class="currency-item"><span class="currency-tag-label">Partner PO</span><span class="currency-tag-value">{ppc}</span></div>
          <div class="currency-item"><span class="currency-tag-label">Vendor Quote</span><span class="currency-tag-value">{vqc}</span></div>
        </div>
        {f'<p class="currency-notes">{currency_notes}</p>' if currency_notes else ""}
      </div>"""

    # ── Document header validation block ────────────────────────
    dhv = result.get("document_header_validation") or {}

    def _dhv_row(label, field_data):
        if not field_data:
            return ""
        zq_val  = field_data.get("zq_value") or "—"
        pdf_val = field_data.get("pdf_value") or "—"
        status  = field_data.get("status") or "—"
        note    = field_data.get("note") or ""
        return (
            f"<tr>"
            f"<td style='font-weight:600;font-size:9px'>{label}</td>"
            f"<td style='font-size:9px;font-family:monospace'>{zq_val}</td>"
            f"<td style='font-size:9px;font-family:monospace'>{pdf_val}</td>"
            f"<td style='text-align:center'>{status_badge(status)}</td>"
            f"<td style='font-size:9px;color:#6b7280'>{note}</td>"
            f"</tr>"
        )

    dhv_rows = (
        _dhv_row("Reseller",        dhv.get("reseller"))
        + _dhv_row("Partner PO Ref",  dhv.get("partner_po_ref"))
        + _dhv_row("Vendor",          dhv.get("vendor"))
        + _dhv_row("Vendor Quote Ref",dhv.get("vendor_quote_ref"))
    )

    header_validation_block = ""
    if dhv_rows:
        header_validation_block = f"""<div class="card">
        <div class="card-title">Document Header Validation</div>
        <table>
          <thead>
            <tr>
              <th style="width:110px">Field</th>
              <th style="width:170px">Zoho Quote Value</th>
              <th style="width:170px">PDF Extracted Value</th>
              <th style="width:82px;text-align:center">Status</th>
              <th>Note</th>
            </tr>
          </thead>
          <tbody>{dhv_rows}</tbody>
        </table>
      </div>"""

    if job_id and is_cancelled(job_id):
        raise Exception("Job cancelled by user")

    table_rows = ""
    for i, r in enumerate(result.get("matching_table", [])):
        row_bg = "#ffffff" if i % 2 == 0 else "#f9fafb"
        table_rows += f"""<tr style="background:{row_bg}">
            <td style="text-align:center;font-weight:600">{r.get("num") or ""}</td>
            <td style="font-family:monospace;font-size:9px;color:#374151;word-break:break-all">{r.get("sku") or ""}</td>
            <td style="text-align:center">{r.get("ppo_qty") or "-"}</td>
            <td style="text-align:center;font-weight:600">{r.get("zq_qty") or "-"}</td>
            <td style="text-align:center">{status_badge(r.get("zq_status"))}</td>
            <td style="text-align:center;font-weight:600">{r.get("vq_qty") or "-"}</td>
            <td style="text-align:center">{status_badge(r.get("ppo_status"))}</td>
            <td style="text-align:center">{status_badge(r.get("vq_status"))}</td>
            <td style="text-align:center">{status_badge(r.get("list_price_comparison_status"))}</td>
            <td style="text-align:center">{status_badge(r.get("buy_price_comparison_status"))}</td>
            <td style="font-size:9px;color:#6b7280;line-height:1.4">{r.get("notes") or ""}</td>
        </tr>"""

    if job_id and is_cancelled(job_id):
        raise Exception("Job cancelled by user")

    must_resolve = "".join([f'<li class="item-red">{i}</li>'   for i in (result.get("must_resolve") or [])]) \
                   or '<li style="color:#6b7280;font-style:italic">None — all items cleared</li>'
    needs_review = "".join([f'<li class="item-amber">{i}</li>' for i in (result.get("needs_review") or [])]) \
                   or '<li style="color:#6b7280;font-style:italic">None — no items flagged for review</li>'
    unmatched    = "".join([f'<span class="tag">{i}</span>'    for i in (result.get("unmatched_items") or [])])
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    by_line      = f" &nbsp;|&nbsp; Initiated by: {initiated_by}" if initiated_by else ""

    html_content = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8"/>
<style>
  @page {{ size: A4 landscape; margin: 12mm; }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: Arial, Helvetica, sans-serif; font-size: 11px; color: #1a1a2e; background: #f4f6f9; }}
  .header {{ margin-bottom: 12px; }}
  .header h1 {{ font-size: 18px; font-weight: bold; color: #1a1a2e; margin-bottom: 2px; }}
  .header .subtitle {{ font-size: 9px; color: #6b7280; }}
  .banner {{ border-radius: 6px; padding: 9px 12px; margin-bottom: 12px; border-left: 5px solid {banner_border}; background: {banner_bg}; }}
  .banner-title {{ font-weight: bold; font-size: 12px; color: #1a1a2e; margin-bottom: 3px; }}
  .banner ul {{ list-style: none; padding: 0; margin: 0; }}
  .banner ul li {{ font-size: 9px; color: #374151; padding: 1px 0; line-height: 1.5; }}
  .banner ul li:before {{ content: "- "; }}
  .card {{ background: #fff; border-radius: 6px; padding: 10px 12px; margin-bottom: 12px; border: 1px solid #e5e7eb; }}
  .card-title {{ font-size: 9px; font-weight: bold; text-transform: uppercase; letter-spacing: 0.06em; color: #374151; border-bottom: 2px solid #f3f4f6; padding-bottom: 5px; margin-bottom: 8px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 9px; }}
  thead th {{ background: #1a1a2e; color: #fff; padding: 6px 7px; text-align: left; font-weight: 600; font-size: 8px; text-transform: uppercase; letter-spacing: 0.04em; }}
  thead th:nth-child(1) {{ width: 28px; }}
  thead th:nth-child(2) {{ width: 120px; }}
  thead th:nth-child(3),
  thead th:nth-child(4),
  thead th:nth-child(6) {{ width: 42px; text-align: center; }}
  thead th:nth-child(5),
  thead th:nth-child(7),
  thead th:nth-child(8),
  thead th:nth-child(9),
  thead th:nth-child(10) {{ width: 68px; text-align: center; }}
  thead th:nth-child(11) {{ width: auto; }}
  tbody td {{ padding: 5px 7px; border-bottom: 1px solid #f3f4f6; vertical-align: top; line-height: 1.4; }}
  .pill {{ display: inline-block; padding: 1px 6px; border-radius: 8px; font-size: 8px; font-weight: 600; }}
  .pill-match  {{ background: #d1fae5; color: #065f46; }}
  .pill-review {{ background: #fef3c7; color: #92400e; }}
  .pill-miss   {{ background: #fee2e2; color: #991b1b; }}
  .pill-na     {{ background: #f3f4f6; color: #9ca3af; }}
  .summary-label {{ font-weight: bold; font-size: 9px; margin: 7px 0 3px 0; }}
  .label-red   {{ color: #ef4444; }}
  .label-amber {{ color: #f59e0b; }}
  .label-green {{ color: #10b981; }}
  ul.summary-list {{ list-style: none; padding: 0; margin: 0 0 5px 0; }}
  ul.summary-list li {{ font-size: 9px; color: #374151; line-height: 1.5; padding: 2px 0 2px 8px; margin-bottom: 2px; }}
  li.item-red   {{ border-left: 3px solid #ef4444; }}
  li.item-amber {{ border-left: 3px solid #f59e0b; }}
  .tag {{ display: inline-block; background: #fee2e2; color: #991b1b; border-radius: 3px; padding: 1px 5px; font-size: 8px; margin: 2px; font-family: monospace; }}
  .overall-text {{ font-size: 9px; color: #374151; line-height: 1.6; }}
  .currency-card {{ padding-bottom: 10px; }}
  .currency-row {{ display: flex; gap: 12px; margin-bottom: 6px; }}
  .currency-item {{ background: #f4f6f9; border-radius: 5px; padding: 5px 10px; display: flex; flex-direction: column; gap: 1px; }}
  .currency-tag-label {{ font-size: 7px; text-transform: uppercase; letter-spacing: 0.05em; color: #9ca3af; font-weight: 700; }}
  .currency-tag-value {{ font-size: 11px; font-weight: 700; color: #1a1a2e; font-family: monospace; }}
  .currency-notes {{ font-size: 9px; color: #374151; line-height: 1.5; border-top: 1px solid #f3f4f6; padding-top: 6px; margin-top: 4px; }}
</style>
</head>
<body>
  <div class="header">
    <h1>Procurement Analysis Report</h1>
    <div class="subtitle">Quote: {quote_subject} &nbsp;|&nbsp; Generated: {generated_at}{by_line}</div>
  </div>
  {currency_block}
  {header_validation_block}
  <div class="banner">
    <div class="banner-title">{result.get("final_call","")}</div>
  </div>
  <div class="card">
    <div class="card-title">Section 1 - Three-Way Item Matching</div>
    <table>
      <thead>
        <tr>
          <th>#</th><th>ZQ SKU</th>
          <th style="text-align:center">PPO Qty</th>
          <th style="text-align:center">ZQ Qty</th>
          <th style="text-align:center">ZQ&#8596;PPO</th>
          <th style="text-align:center">VQ Qty</th>
          <th style="text-align:center">ZQ&#8596;VQ</th>
          <th style="text-align:center">PPO&#8596;VQ</th>
          <th style="text-align:center">ZQ-PPO Price</th>
          <th style="text-align:center">ZQ-VQ Price</th>
          <th>Notes</th>
        </tr>
      </thead>
      <tbody>{table_rows}</tbody>
    </table>
  </div>
  <div class="card">
    <div class="card-title">Section 2 - Summary</div>
    <div class="summary-label label-red">Must Resolve Before Processing</div>
    <ul class="summary-list">{must_resolve}</ul>
    <div class="summary-label label-amber">Needs Human Review</div>
    <ul class="summary-list">{needs_review}</ul>
    {"<div class='summary-label label-red'>Unmatched Items</div><div>" + unmatched + "</div>" if unmatched else ""}
    <div class="summary-label label-green">Overall</div>
    <p class="overall-text">{result.get("overall_summary","")}</p>
  </div>
</body>
</html>"""

    if job_id and is_cancelled(job_id):
        raise Exception("Job cancelled by user")

    pdf_bytes = HTML(string=html_content).write_pdf()
    print(f"PDF generated: {len(pdf_bytes)} bytes in {time.time()-t0:.1f}s")
    return pdf_bytes


# ─────────────────────────────────────────────
# JOB HELPERS
# ─────────────────────────────────────────────
def is_cancelled(job_id: str) -> bool:
    return jobs.get(job_id, {}).get("status") == "cancelled"


def auto_cancel_watcher(job_id: str, timeout_seconds: int = 6):
    while True:
        time.sleep(3)
        job = jobs.get(job_id, {})
        if job.get("status") in ("done", "error", "cancelled"):
            print(f"[{job_id}] 👁️  Watcher stopped — {job.get('status')}")
            return
        elapsed = time.time() - last_poll.get(job_id, 0)
        if elapsed > timeout_seconds:
            print(f"[{job_id}] 👁️  No poll for {elapsed:.0f}s — auto-cancelling")
            jobs[job_id] = {"status": "cancelled"}
            return
        print(f"[{job_id}] 👁️  Watcher: last poll {elapsed:.0f}s ago")


# ─────────────────────────────────────────────
# 12. BACKGROUND JOB
#     OPT 4: Parallel PDF downloads
# ─────────────────────────────────────────────
def process_quote_job(job_id: str, quote_id: str, initiated_by: str = ""):
    try:
        jobs[job_id] = {"status": "processing", "phase": "Initialising..."}
        # Reset the watcher clock to now — the 6s abandonment window starts
        # from when the job actually begins, not from when /analyze-quote
        # responded (which can be 5-10s earlier through the Zoho SDK proxy).
        last_poll[job_id] = time.time()
        t0 = time.time()

        if is_cancelled(job_id): return
        token = get_access_token()
        print(f"[{job_id}] ⏱ Auth: {time.time()-t0:.1f}s")
        jobs[job_id]["phase"] = "Fetching quote from Zoho..."

        if is_cancelled(job_id): return
        quote = fetch_zoho_quote(quote_id, token)
        print(f"[{job_id}] ⏱ Fetch quote: {time.time()-t0:.1f}s")
        print(f"Quote fields: {list(quote.keys())}")

        if is_cancelled(job_id): return
        zoho_text = format_zoho_quote(quote)
        print(f"Quote items to be sent to Gemini: {zoho_text}")
        # Build dynamic filename from Quotation_Reference field
        quote_ref    = quote.get("Quotation_Reference", "")
        # Sanitise — remove characters not allowed in filenames
        safe_ref     = re.sub(r'[^\w\-_.]', '_', str(quote_ref).strip())
        report_name  = f"DOC_Compare_{safe_ref}.pdf"
        print(f"[{job_id}] Report filename: {report_name}")

        # ── Validate attachments exist before proceeding ──────────
        ppo_field = quote.get(PPO_PDF_FIELD)
        vq_field = quote.get(VQ_PDF_FIELD)

        missing = []
        if not ppo_field or not isinstance(ppo_field, list) or len(ppo_field) == 0:
            missing.append(f"Partner PO PDF (field: {PPO_PDF_FIELD})")
        if not vq_field or not isinstance(vq_field, list) or len(vq_field) == 0:
            missing.append(f"Vendor Quote PDF (field: {VQ_PDF_FIELD})")

        if missing:
            raise Exception(
                "Required PDF attachments are missing from this quote record. "
                "Please attach the following files before running comparison:\n"
                + "\n".join(f"  - {m}" for m in missing)
            )

        fid_ppo = ppo_field[0].get('file_Id')
        fid_vq = vq_field[0].get('file_Id')

        if not fid_ppo:
            raise Exception(
                f"Partner PO PDF  is attached but has no file ID. "
                f"Please re-attach the {PPO_PDF_FIELD} file and try again."
            )
        if not fid_vq:
            raise Exception(
                f"Vendor Quote PDF is attached but has no file ID. "
                f"Please re-attach the {VQ_PDF_FIELD} file and try again."
            )

        print(f"[{job_id}] Attachments validated — PPO: {fid_ppo}, PO: {fid_vq}")
        jobs[job_id]["phase"] = "Downloading PDF attachments..."

        # OPT 4: Download both PDFs in parallel

        print(f"[{job_id}] 📥 Downloading PPO + VQ in parallel...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            future_ppo_dl = executor.submit(download_zoho_file, fid_ppo, token)
            future_vq_dl = executor.submit(download_zoho_file, fid_vq, token)

            while not (future_ppo_dl.done() and future_vq_dl.done()):
                time.sleep(0.5)
                if is_cancelled(job_id):
                    future_ppo_dl.cancel()
                    future_vq_dl.cancel()
                    return

            ppo_bytes = future_ppo_dl.result()
            vq_bytes = future_vq_dl.result()

        print(f"[{job_id}] ⏱ Downloads done: {time.time()-t0:.1f}s")

        invalid_files = []
        if not is_valid_pdf(ppo_bytes):
            invalid_files.append(f"Partner PO PDF (field: {PPO_PDF_FIELD})")
        if not is_valid_pdf(vq_bytes):
            invalid_files.append(f"Vendor Quote PDF (field: {VQ_PDF_FIELD})")

        if invalid_files:
            raise Exception(
                "The following attachments are not valid PDF files. "
                "Please remove them and attach a PDF instead:\n"
                + "\n".join(f"  - {f}" for f in invalid_files)
            )
        jobs[job_id]["phase"] = "Extracting line items with Gemini AI..."

        if is_cancelled(job_id): return
        gemini_model = get_gemini_model()

        # Load extraction prompt once — shared by both parallel PDF extractions
        gemini_prompt = load_gemini_prompt()

        # Gemini extractions in parallel
        print(f"[{job_id}] 🔍 Gemini extraction (parallel)...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            future_ppo = executor.submit(extract_pdf_gemini, ppo_bytes, "Partner PO PDF", gemini_model, job_id, gemini_prompt)
            future_vq  = executor.submit(extract_pdf_gemini, vq_bytes,  "VQ PDF",         gemini_model, job_id, gemini_prompt)

            while not (future_ppo.done() and future_vq.done()):
                time.sleep(1)
                if is_cancelled(job_id):
                    print(f"[{job_id}] ❌ Cancelled during Gemini extraction")
                    future_ppo.cancel()
                    future_vq.cancel()
                    return

            ppo_text = future_ppo.result()
            vq_text = future_vq.result()

        print("VQ TEXT from Gemini:"+vq_text)
        print("Partner PO TEXT from Gemini:"+ppo_text)
        print(f"[{job_id}] ⏱ Gemini done: {time.time()-t0:.1f}s")
        jobs[job_id]["phase"] = "Comparing documents with Claude AI..."

        if is_cancelled(job_id): return
        result = run_comparison(zoho_text, ppo_text, vq_text, job_id)
        print(f"The final result from claude is : {result}")

        print(f"[{job_id}] ⏱ Claude done: {time.time()-t0:.1f}s")
        jobs[job_id]["phase"] = "Generating PDF report..."

        if is_cancelled(job_id): return
        pdf_bytes     = generate_pdf_report(result, quote.get("Subject", quote_id), job_id, initiated_by)
        print(f"[{job_id}] ⏱ PDF Report generation done: {time.time()-t0:.1f}s")
        jobs[job_id]["phase"] = "Attaching report to Zoho quote..."

        if is_cancelled(job_id): return
        attachment_id = attach_pdf_to_quote(quote_id, pdf_bytes, token,report_name)
        print(f"[{job_id}] ⏱ Attachment of PDF done: {time.time()-t0:.1f}s")

        print(f"[{job_id}] ⏱ Total: {time.time()-t0:.1f}s")

        jobs[job_id] = {
            "status":        "done",
            "result":        result,
            "attachment_id": attachment_id,
            "generated_at":  datetime.now().isoformat(),
            "quote_ref":     quote_ref
        }
        print(f"[{job_id}] ✅ Complete")

    except Exception as e:
        import traceback
        print(f"❌ Job failed — {job_id}\n{traceback.format_exc()}")
        if jobs.get(job_id, {}).get("status") != "cancelled":
            jobs[job_id] = {"status": "error", "error": str(e)}

def is_valid_pdf(file_bytes: bytes) -> bool:
    """Check the actual file signature, not just filename/extension."""
    return file_bytes[:5] == b'%PDF-'
# ─────────────────────────────────────────────
# 13. FASTAPI ENDPOINTS
# ─────────────────────────────────────────────

@app.post("/analyze-quote")
def analyze_quote(payload: dict, background_tasks: BackgroundTasks):
    quote_id     = payload.get("quote_id")
    initiated_by = payload.get("initiated_by", "")
    if not quote_id:
        return JSONResponse(status_code=400, content={"error": "quote_id missing"})

    job_id            = str(uuid.uuid4())
    jobs[job_id]      = {"status": "processing", "phase": "Fetching quote from Zoho..."}
    last_poll[job_id] = time.time()

    background_tasks.add_task(process_quote_job, job_id, quote_id, initiated_by)

    threading.Thread(
        target=auto_cancel_watcher,
        args=(job_id, 6),   # 6s — if widget closes, cancel within one watcher cycle
        daemon=True
    ).start()

    print(f"[{job_id}] 🚀 Job started for quote {quote_id}")
    return {"job_id": job_id}


@app.get("/job-status/{job_id}")
def get_job_status(job_id: str):
    last_poll[job_id] = time.time()
    job = jobs.get(job_id)
    if not job:
        return {"status": "not_found"}
    return job


@app.get("/check-report/{quote_id}")
def check_report(quote_id: str):
    try:
        token      = get_access_token()
        attachment = check_existing_report(quote_id, token)
        if attachment:
            return {
                "exists":        True,
                "attachment_id": attachment.get("id"),
                "file_name":     attachment.get("File_Name"),
                "created_time":  attachment.get("Created_Time")
            }
        return {"exists": False}
    except Exception as e:
        return {"exists": False, "error": str(e)}


@app.post("/cancel-job/{job_id}")
def cancel_job(job_id: str):
    job    = jobs.get(job_id)
    status = job.get("status") if job else None
    print(f"[{job_id}] ❌ Cancel request — current status: {status}")
    if job and status == "processing":
        jobs[job_id] = {"status": "cancelled"}
        print(f"[{job_id}] ❌ Job cancelled by user")
    return {"cancelled": True}


@app.get("/download-report/{quote_id}")
def download_report(quote_id: str):
    try:
        token      = get_access_token()
        attachment = check_existing_report(quote_id, token)
        if not attachment:
            return JSONResponse(status_code=404, content={"error": "No report found"})
        pdf_bytes = download_zoho_file(attachment["id"], token)
        return {
            "pdf_base64": base64.b64encode(pdf_bytes).decode("utf-8"),
            "file_name":  attachment.get("File_Name")
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
