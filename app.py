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
# MARGIN GATE CONFIG — hardcoded, no env vars
# Vendor name is already read directly off the Quote (quote.get("Vendor")).
# Opportunity name is read the same way, off OPPORTUNITY_FIELD_ON_QUOTE below.
# The Quote only stores descriptive *names* for both, not record ids, so the
# gate resolves the real record by searching each related module by name.
# VERIFY these against your org with GET /inspect-quote/{quote_id} and edit
# these constants directly (no redeploy env-var juggling needed — just a
# code change + redeploy).
# ─────────────────────────────────────────────
OPPORTUNITY_FIELD_ON_QUOTE = "Deal_Name"   # field on Quotes holding the Opportunity's lookup

OPPORTUNITIES_MODULE       = "Deals"       # Zoho API module name — ALWAYS "Deals" even if
                                            # the UI tab is relabeled "Opportunities" (tab
                                            # renaming is display-only, doesn't change api_name)
OPPORTUNITIES_NAME_FIELD   = "Deal_Name"   # fallback only, not used while id is present
# NOTE: this org has FOUR different fields all labeled some variant of "Gross Margin"
# on the Deals/Opportunities module (Gross_Margin, Gross_Deal_Margin, GrossMarginCalc,
# Gross_Margin_USD) — confirmed via Setup > Developer Space > API Names. Only
# GrossMarginCalc (a formula field, label "Gross Margin(%)" — no space before the
# parenthesis) is actually populated on real records. Gross_Margin (decimal,
# label "Gross Margin (%)" — WITH a space) looked identical in the UI but was
# always null. Verify against a real record if this ever needs revisiting.
GROSS_MARGIN_FIELD         = "GrossMarginCalc"    # field on Deals holding Gross Margin %

VENDORS_MODULE       = "Vendors"        # custom module API name
VENDORS_NAME_FIELD   = "Name"           # field used to search Vendors by name
VENDOR_MARGIN_FIELD  = "Min_Acceptable_Margin"  # field on Vendors holding Vendor Margin %

# ─────────────────────────────────────────────
# SUBTOTAL VALIDATION CONFIG — hardcoded, no env vars
# Compares PDF-extracted subtotals (converted to USD) against two Opportunity
# fields. Fetched in the SAME Opportunity API call the Margin Gate already
# makes (see resolve_margin_by_name's extra_fields param) — no extra round trip.
# VERIFY these field names against your org with GET /inspect-quote/{quote_id}.
# ─────────────────────────────────────────────
AMOUNT_IN_USD_FIELD = "Amount_in_USD"   # field on Deals — compared against Partner PO Subtotal (converted to USD)
NET_TO_VENDOR_FIELD = "Net_to_Vendor"   # field on Deals — compared against Vendor Quote Subtotal (converted to USD)
                                         # ASSUMPTION: Net_to_Vendor is itself stored in USD already
                                         # (matching the Amount_in_USD naming convention on this Opportunity
                                         # module). If that's wrong, tell me and this needs a currency field too.
SUBTOTAL_TOLERANCE_USD = 1.00           # looser than the ±0.50 line-item tolerance — subtotals sum many
                                         # lines, so small per-line rounding differences compound

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
    Raises RuntimeError if the module is empty, unreachable, or fields are blank —
    always with Zoho's actual response detail included, never a bare status code."""
    token = get_access_token()
    url   = f"{ZOHO_BASE_URL}/crm/v3/AIPrompts"
    headers = {"Authorization": f"Zoho-oauthtoken {token}"}
    params  = {"fields": "GEMINI_PROMPT,CLAUDE_PROMPT", "per_page": 1}

    r = requests.get(url, headers=headers, params=params, timeout=20)
    print(f"[prompts] AIPrompts CRM fetch status: {r.status_code}")

    if r.status_code == 204:
        raise RuntimeError(
            "AIPrompts module is empty — please create a record with "
            "GEMINI_PROMPT and CLAUDE_PROMPT field values."
        )

    if not r.ok:
        # Include Zoho's actual response body — a bare "400 Client Error" tells
        # you nothing about WHY (wrong module name, wrong field name, no
        # permission, etc.). This is almost always the real cause when prompts
        # can't load, so the detail matters here more than anywhere else.
        raise RuntimeError(
            f"AIPrompts CRM fetch failed: {r.status_code} {r.text[:500]}"
        )

    try:
        data = r.json().get("data", [])
    except ValueError as e:
        raise RuntimeError(
            f"AIPrompts CRM fetch returned an unparseable response "
            f"(status {r.status_code}): {r.text[:300]}"
        ) from e

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
    Falls back to stale cache values if CRM is unreachable — but if there's no
    cache to fall back on, the ORIGINAL specific error from _fetch_prompts_from_crm
    (e.g. "GEMINI_PROMPT field is blank", or the actual Zoho response body) is
    preserved in the message, not discarded in favour of a generic one. This was
    previously the main reason prompt-loading failures showed an unhelpful error:
    the real cause was being swallowed right here."""
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
            f"Could not load AI prompts from CRM AIPrompts module, and no cached "
            f"prompts are available to fall back on. Underlying error: {e}"
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
# 3b. MARGIN GATE — resolve Vendor / Opportunity records by name
#     and compare Gross Margin % vs Vendor Margin %
# ─────────────────────────────────────────────
def search_zoho_record_by_name(module: str, name: str, name_field: str, token: str) -> dict:
    """The Quote only stores descriptive names (e.g. quote['Vendor'] = 'Acme Corp'),
    not record ids, so we search the related module for a matching name.
    Requires `name_field` to be marked searchable in Zoho module setup."""
    if not name:
        return None
    url     = f"{ZOHO_BASE_URL}/crm/v3/{module}/search"
    headers = {"Authorization": f"Zoho-oauthtoken {token}"}
    params  = {"criteria": f"({name_field}:equals:{name})"}
    r = requests.get(url, headers=headers, params=params, timeout=30)
    if r.status_code == 204:
        return None
    if not r.ok:
        raise Exception(
            f"Zoho API error searching {module} for {name_field}='{name}': "
            f"{r.status_code} {r.text[:500]}"
        )
    data = r.json().get("data", [])
    return data[0] if data else None


def fetch_zoho_record(module: str, record_id: str, token: str, fields: str = None) -> dict:
    """Fetch a single record by id from any Zoho CRM module."""
    url     = f"{ZOHO_BASE_URL}/crm/v3/{module}/{record_id}"
    headers = {"Authorization": f"Zoho-oauthtoken {token}"}
    params  = {"fields": fields} if fields else {}
    r = requests.get(url, headers=headers, params=params, timeout=30)
    if not r.ok:
        raise Exception(
            f"Zoho API error fetching {module}/{record_id} (fields={fields}): "
            f"{r.status_code} {r.text[:500]}"
        )
    data = r.json().get("data", [])
    if not data:
        raise Exception(f"No record found in {module} with id {record_id}")
    return data[0]


def parse_percentage(value):
    """Coerce a margin value (number, '15', '15%', etc.) to a float, or None."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip().replace("%", "")
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def parse_amount(value):
    """Coerce a money value (number, '1,234.50', '$1234.50', 'USD 1234.50', etc.)
    to a float, or None. Strips thousands separators and common currency symbols/codes."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip()
    if not s:
        return None
    s = re.sub(r"[,$€£]", "", s)
    s = re.sub(r"\b(USD|AED|SAR|QAR)\b", "", s, flags=re.IGNORECASE).strip()
    try:
        return float(s)
    except ValueError:
        return None


# Fixed exchange rates to USD — same anchor rates used in the Claude matching
# prompt's Step 3c, kept in sync manually since Claude's copy lives in Zoho CRM.
_USD_RATES = {"USD": 1.0, "AED": 1 / 3.6725, "SAR": 1 / 3.7500, "QAR": 1 / 3.6500}


def convert_to_usd(amount, currency):
    """Convert amount (in `currency`) to USD using the fixed rate table.
    Returns None if amount is missing or currency is unsupported/blank —
    never guesses a rate for an unrecognised currency code."""
    if amount is None or not currency:
        return None
    rate = _USD_RATES.get(str(currency).strip().upper())
    if rate is None:
        return None
    return round(amount * rate, 2)


def resolve_margin_by_name(raw_value, module: str, name_field: str,
                            margin_field: str, token: str, extra_fields: str = None):
    """
    raw_value is whatever came straight off the Quote for Vendor / Opportunity —
    it can be:
      - a Zoho lookup dict {"id": "...", "name": "..."} → fetch directly by id.
        This is the expected path for both Vendor and Opportunity on this org's
        Quotes module. If this fetch fails, the error is NOT swallowed/retried
        via search — it propagates up so the real cause (wrong module name,
        wrong field name, etc.) is visible instead of being masked.
      - a plain string (display name only, no id — not currently expected for
        Vendor/Opportunity on this org, kept as a safety net for other future
        uses of this function) → search-by-name
      - None / "N/A" / "" → nothing to resolve

    extra_fields: optional comma-separated list of additional field API names to
    fetch on the SAME record (avoids a second round trip for callers that need
    more than just the margin field — e.g. the Opportunity record is also used
    for Subtotal Validation). Returns (matched_display_name, margin_value_or_None,
    full_record_dict_or_None) — the record dict contains margin_field plus any
    extra_fields requested, so callers can pull additional values off it directly.
    """
    if not raw_value or raw_value == "N/A":
        return None, None, None

    fields_to_request = margin_field if not extra_fields else f"{margin_field},{extra_fields}"

    if isinstance(raw_value, dict):
        record_id    = raw_value.get("id")
        display_name = raw_value.get("name")
        if record_id:
            # Direct id fetch only — no fallback to search. We already have the
            # display name from this lookup dict, so we only need to request the
            # margin field (avoids depending on name_field being a valid/requestable
            # field on this module, which it may not be after a module rename).
            record = fetch_zoho_record(module, record_id, token, fields=fields_to_request)
            return display_name, parse_percentage(record.get(margin_field)), record
        raw_value = display_name  # dict with no id at all — only then try name search

    if not raw_value:
        return None, None, None

    found = search_zoho_record_by_name(module, raw_value, name_field, token)
    if not found:
        print(f"⚠️  No {module} record found matching name '{raw_value}' (searched field '{name_field}')")
        return raw_value, None, None

    record = fetch_zoho_record(module, found["id"], token, fields=f"{fields_to_request},{name_field}")
    return record.get(name_field) or raw_value, parse_percentage(record.get(margin_field)), record


def check_margin_gate(quote: dict, token: str) -> dict:
    """
    Opportunity Gross Margin % vs Vendor Margin % check — runs first, before
    any PDF work. Returns 'blocked': True when BOTH margins resolve and
    gross_margin < vendor_margin — despite the field's name, this NO LONGER
    stops the pipeline (kept as 'blocked' rather than renamed to avoid
    touching every call site; treat it as "flagged for review" wherever it's
    read downstream, e.g. process_quote_job maps it straight into
    result["margin_gate"]["needs_review"]). Any failure — missing data,
    unresolvable name, unexpected API error — results in the gate being
    SKIPPED, logged clearly, never flagged. A margin-gate config issue must
    never silently stall a legitimate comparison run.

    Also resolves amount_in_usd / net_to_vendor off the SAME Opportunity record
    fetch (see AMOUNT_IN_USD_FIELD / NET_TO_VENDOR_FIELD) — these aren't used by
    the gate itself, they're carried in the outcome for the later Subtotal
    Validation check so that check doesn't need its own Opportunity API call.
    """
    outcome = {
        "blocked":          False,
        "opportunity_name": None,
        "vendor_name":      None,
        "gross_margin":     None,
        "vendor_margin":    None,
        "skipped_reason":   None,
        "amount_in_usd":    None,
        "net_to_vendor":    None,
    }

    try:
        vendor_raw      = quote.get("Vendor")
        opportunity_raw = quote.get(OPPORTUNITY_FIELD_ON_QUOTE)

        vendor_display, vendor_margin, _ = resolve_margin_by_name(
            vendor_raw, VENDORS_MODULE, VENDORS_NAME_FIELD, VENDOR_MARGIN_FIELD, token
        )
        opp_display, gross_margin, opp_record = resolve_margin_by_name(
            opportunity_raw, OPPORTUNITIES_MODULE, OPPORTUNITIES_NAME_FIELD, GROSS_MARGIN_FIELD, token,
            extra_fields=f"{AMOUNT_IN_USD_FIELD},{NET_TO_VENDOR_FIELD}"
        )
    except Exception as e:
        print(f"⚠️  Margin gate failed unexpectedly — skipping gate, proceeding to comparison. Reason: {e}")
        outcome["skipped_reason"] = f"Margin gate error: {e}"
        return outcome

    outcome["opportunity_name"] = opp_display
    outcome["vendor_name"]      = vendor_display
    outcome["gross_margin"]     = gross_margin
    outcome["vendor_margin"]    = vendor_margin
    if opp_record:
        outcome["amount_in_usd"] = parse_amount(opp_record.get(AMOUNT_IN_USD_FIELD))
        outcome["net_to_vendor"] = parse_amount(opp_record.get(NET_TO_VENDOR_FIELD))

    if gross_margin is None or vendor_margin is None:
        missing = []
        if gross_margin  is None: missing.append(f"Gross Margin (Opportunity: {opp_display or 'not set on quote'})")
        if vendor_margin is None: missing.append(f"Vendor Margin (Vendor: {vendor_display or 'not set on quote'})")
        outcome["skipped_reason"] = "Missing: " + ", ".join(missing)
        print(f"⚠️  Margin gate skipped — {outcome['skipped_reason']}")
        return outcome

    print(f"📊 Margin gate — Gross Margin: {gross_margin}% | Vendor Margin: {vendor_margin}%")
    if gross_margin < vendor_margin:
        outcome["blocked"] = True

    return outcome


def check_subtotal_validation(quote: dict, margin: dict, ppo_header: dict, vq_header: dict) -> dict:
    """
    Subtotal Validation — a reporting-only card, never blocks the pipeline and
    never affects final_call. Computed deterministically in Python (same
    reasoning as the Margin Gate: this is arithmetic + a fixed exchange rate
    table, not a judgement call, so it shouldn't be delegated to Claude).

    Two independent checks:
      A. Partner PO Subtotal (converted to USD) vs Opportunity's Amount_in_USD
      B. Vendor Quote Subtotal (converted to USD) vs Opportunity's Net_to_Vendor
         (ASSUMES Net_to_Vendor is already stored in USD — see config note)

    Each check is "Match" | "Mismatch" | "Needs Review" (Needs Review when
    either side is missing/unresolvable — deliberately not forced into a binary
    match/mismatch when the data to make that call isn't actually available).
    overall_status is "Match" only if both checks are "Match".
    """
    partner_po_currency   = quote.get("Partner_PO_Currency") or ""
    vendor_quote_currency = quote.get("Vendor_Quote_Currency") or ""

    ppo_subtotal_raw = parse_amount((ppo_header or {}).get("partner_po_subtotal"))
    vq_subtotal_raw  = parse_amount((vq_header or {}).get("vendor_quote_subtotal"))

    ppo_subtotal_usd = convert_to_usd(ppo_subtotal_raw, partner_po_currency)
    vq_subtotal_usd  = convert_to_usd(vq_subtotal_raw, vendor_quote_currency)

    amount_in_usd = margin.get("amount_in_usd")
    net_to_vendor = margin.get("net_to_vendor")

    def _compare(pdf_val_usd, opp_val):
        if pdf_val_usd is None or opp_val is None:
            return "Needs Review"
        return "Match" if abs(pdf_val_usd - opp_val) <= SUBTOTAL_TOLERANCE_USD else "Mismatch"

    ppo_status = _compare(ppo_subtotal_usd, amount_in_usd)
    vq_status  = _compare(vq_subtotal_usd, net_to_vendor)

    if ppo_status == "Mismatch" or vq_status == "Mismatch":
        overall = "Mismatch"
    elif ppo_status == "Needs Review" or vq_status == "Needs Review":
        overall = "Needs Review"
    else:
        overall = "Match"

    print(f"💰 Subtotal validation — PPO: {ppo_subtotal_usd} USD vs Amount_in_USD {amount_in_usd} → {ppo_status} | "
          f"VQ: {vq_subtotal_usd} USD vs Net_to_Vendor {net_to_vendor} → {vq_status}")

    return {
        "overall_status": overall,
        "partner_po": {
            "label":             "Partner PO Subtotal vs Amount (USD)",
            "pdf_subtotal":      ppo_subtotal_raw,
            "pdf_currency":      partner_po_currency or None,
            "pdf_subtotal_usd":  ppo_subtotal_usd,
            "opportunity_value": amount_in_usd,
            "status":            ppo_status,
        },
        "vendor_quote": {
            "label":             "Vendor Quote Subtotal vs Net to Vendor (USD)",
            "pdf_subtotal":      vq_subtotal_raw,
            "pdf_currency":      vendor_quote_currency or None,
            "pdf_subtotal_usd":  vq_subtotal_usd,
            "opportunity_value": net_to_vendor,
            "status":            vq_status,
        },
    }


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
def extract_pdf_gemini(pdf_bytes: bytes, label: str, model_name: str, job_id: str = None, prompt: str = ""):
    """Extract header fields AND line items from a PDF using Gemini.

    The updated Gemini prompt returns:
      { "header": { reseller_name, partner_po_ref, vendor_name, vendor_quote_ref,
                     partner_po_subtotal, vendor_quote_subtotal },
        "line_items": [ { line_num, sku, description, quantity, list_unit_price }, ... ] }

    Legacy flat-array responses are still handled gracefully.
    Returns a TUPLE: (formatted_text_block, header_dict).
      - formatted_text_block: for Claude to consume (HEADER FIELDS + LINE ITEMS sections)
      - header_dict: the raw parsed header dict, used directly in Python for Subtotal
        Validation (check_subtotal_validation) without re-parsing Claude's prose. Empty
        dict {} if parsing failed — callers must handle a header dict with no keys.
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

        # Header section — always emit all six keys so Claude has them
        lines.append("### HEADER FIELDS")
        lines.append(f"  reseller_name         : {header.get('reseller_name') or 'null'}")
        lines.append(f"  partner_po_ref        : {header.get('partner_po_ref') or 'null'}")
        lines.append(f"  vendor_name           : {header.get('vendor_name') or 'null'}")
        lines.append(f"  vendor_quote_ref      : {header.get('vendor_quote_ref') or 'null'}")
        lines.append(f"  partner_po_subtotal   : {header.get('partner_po_subtotal') if header.get('partner_po_subtotal') is not None else 'null'}")
        lines.append(f"  vendor_quote_subtotal : {header.get('vendor_quote_subtotal') if header.get('vendor_quote_subtotal') is not None else 'null'}")
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

        return "\n".join(lines), header

    except (json.JSONDecodeError, ValueError) as e:
        print(f"⚠️  Gemini JSON parse error for {label}: {e}")
        return raw, {}


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

    # ── Margin gate status banner ─────────────────────────────
    # Shown whenever the gate actually ran ("checked" is True). Two variants:
    # pass (green) or needs_review (amber) — the gate is non-blocking, so this
    # PDF report always contains the full comparison regardless of which one
    # shows here; this is purely informational, same as the widget's banner.
    mg = result.get("margin_gate") or {}
    margin_status_block = ""
    if mg.get("checked"):
        gm = mg.get("gross_margin")
        vm = mg.get("vendor_margin")
        if mg.get("needs_review"):
            margin_status_block = f"""<div class="margin-needs-review-banner">
            <div class="mnr-title">&#9888; Margin Check — Needs Review</div>
            <p class="mnr-detail">Gross Margin ({gm if gm is not None else '—'}%) is less than
            Minimum Vendor Margin ({vm if vm is not None else '—'}%) — flagged for review. Document
            comparison completed as normal.</p>
            <div class="margin-stats-pdf">
              <div class="ms-item"><span class="ms-label">Opportunity</span><span class="ms-value">{mg.get('opportunity_name') or '—'}</span></div>
              <div class="ms-item"><span class="ms-label">Vendor</span><span class="ms-value">{mg.get('vendor_name') or '—'}</span></div>
              <div class="ms-item"><span class="ms-label">Gross Margin</span><span class="ms-value" style="color:#b91c1c">{gm if gm is not None else '—'}%</span></div>
              <div class="ms-item"><span class="ms-label">Minimum Vendor Margin</span><span class="ms-value" style="color:#065f46">{vm if vm is not None else '—'}%</span></div>
            </div>
          </div>"""
        else:
            margin_status_block = f"""<div class="margin-pass-banner">
            <span class="mp-title">&#10003; Margin check passed</span>
            <span class="mp-stat"><span class="mp-label">Gross Margin</span>{gm if gm is not None else '—'}%</span>
            <span class="mp-stat"><span class="mp-label">Minimum Vendor Margin</span>{vm if vm is not None else '—'}%</span>
          </div>"""

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

    # Build per-SKU expanded blocks for the PDF
    def sku_block(r, i):
        row_bg = "#ffffff" if i % 2 == 0 else "#f9fafb"

        # Overall worst status
        all_statuses = [
            r.get("zq_status",""), r.get("vq_status",""), r.get("ppo_status",""),
            r.get("list_price_comparison_status",""), r.get("buy_price_comparison_status","")
        ]
        worst = "match"
        for s in all_statuses:
            sl = (s or "").lower()
            if "mismatch" in sl: worst = "mismatch"; break
            if "review"   in sl and worst != "mismatch": worst = "review"
        if worst == "mismatch": pill_bg, pill_cl, pill_lbl = "#fee2e2","#991b1b","Mismatch"
        elif worst == "review": pill_bg, pill_cl, pill_lbl = "#fef3c7","#92400e","Review"
        else:                   pill_bg, pill_cl, pill_lbl = "#d1fae5","#065f46","Match"
        overall_pill = f'<span class="pill" style="background:{pill_bg};color:{pill_cl}">{pill_lbl}</span>'

        zq_qty  = r.get("zq_qty")  or "-"
        vq_qty  = r.get("vq_qty")  or "-"
        ppo_qty = r.get("ppo_qty") or "-"

        # Qty row — single Match if all qty statuses are clean Match
        all_qty_match = all(
            ("match" in (r.get(f) or "").lower() and "mismatch" not in (r.get(f) or "").lower())
            for f in ["zq_status","vq_status","ppo_status"]
        )
        if all_qty_match:
            qty_row = f"""<tr>
              <td class="rl">Qty</td>
              <td class="cv">{zq_qty}</td>
              <td class="cv">{ppo_qty}</td>
              <td class="cv">{vq_qty}</td>
              <td class="sc">{status_badge("Match")}</td>
            </tr>"""
        else:
            qty_row = f"""<tr>
              <td class="rl">Qty</td>
              <td class="cv">{zq_qty}</td>
              <td class="cv">{ppo_qty}</td>
              <td class="cv">{vq_qty}</td>
              <td class="sc">
                <div style="font-size:7px;color:#6b7280">ZQ↔PPO {status_badge(r.get("zq_status"))}</div>
                <div style="font-size:7px;color:#6b7280">ZQ↔VQ&nbsp;&nbsp;{status_badge(r.get("vq_status"))}</div>
                <div style="font-size:7px;color:#6b7280">PPO↔VQ {status_badge(r.get("ppo_status"))}</div>
              </td>
            </tr>"""

        note = (r.get("notes") or "").strip()
        note_row = f"""<tr>
          <td class="rl">Notes</td>
          <td colspan="4" style="font-size:8px;color:#6b7280;line-height:1.4">{note}</td>
        </tr>""" if note else ""

        return f"""<div class="sku-block" style="background:{row_bg}">
          <div class="sku-hdr">
            <span class="sku-num">{r.get("num") or i+1}</span>
            <span class="sku-code">{r.get("sku") or "—"}</span>
            <span style="margin-left:auto">{overall_pill}</span>
          </div>
          <table class="dt">
            <thead>
              <tr>
                <th style="width:65px"></th>
                <th>ZQ</th><th>PPO</th><th>VQ</th>
                <th style="width:110px;text-align:center">Status</th>
              </tr>
            </thead>
            <tbody>
              {qty_row}
              <tr style="background:#f8f8f8">
                <td class="rl">Price</td>
                <td class="cv" style="font-size:7px;color:#9ca3af;font-weight:600">ZQ</td>
                <td class="cv" style="font-size:7px;color:#9ca3af;font-weight:600">PPO</td>
                <td class="cv" style="font-size:7px;color:#9ca3af;font-weight:600">VQ</td>
                <td></td>
              </tr>
              <tr style="background:#f8f8f8">
                <td class="rl sub">List&nbsp;Price</td>
                <td class="cv">{r.get("list_price_zq") or "-"}</td>
                <td class="cv">{r.get("partner_ppo_price_original") or "-"}</td>
                <td class="cv" style="color:#d1d5db">—</td>
                <td class="sc">{status_badge(r.get("list_price_comparison_status"))}</td>
              </tr>
              <tr style="background:#f8f8f8">
                <td class="rl sub">Buy&nbsp;Price</td>
                <td class="cv">{r.get("buy_price_zq") or "-"}</td>
                <td class="cv" style="color:#d1d5db">—</td>
                <td class="cv">{r.get("vendor_quote_price") or "-"}</td>
                <td class="sc">{status_badge(r.get("buy_price_comparison_status"))}</td>
              </tr>
              {note_row}
            </tbody>
          </table>
        </div>"""

    sku_blocks = "".join(sku_block(r, i) for i, r in enumerate(result.get("matching_table", [])))

    if job_id and is_cancelled(job_id):
        raise Exception("Job cancelled by user")

    # ── Subtotal validation block (vs Opportunity Amount_in_USD / Net_to_Vendor) ──
    sv = result.get("subtotal_validation") or {}
    subtotal_validation_block = ""
    if sv:
        def _sv_row(d):
            if not d:
                return ""
            pdf_usd = d.get("pdf_subtotal_usd")
            opp_val = d.get("opportunity_value")
            pdf_raw = d.get("pdf_subtotal")
            pdf_usd_str = f"${pdf_usd:,.2f}" if pdf_usd is not None else "—"
            opp_val_str = f"${opp_val:,.2f}" if opp_val is not None else "—"
            pdf_raw_str = f"{pdf_raw:,.2f} {d.get('pdf_currency') or ''}".strip() if pdf_raw is not None else "—"
            return (
                f"<tr>"
                f"<td style='font-weight:600;font-size:9px'>{d.get('label','')}</td>"
                f"<td style='font-size:9px;font-family:monospace'>{pdf_raw_str}</td>"
                f"<td style='font-size:9px;font-family:monospace'>{pdf_usd_str}</td>"
                f"<td style='font-size:9px;font-family:monospace'>{opp_val_str}</td>"
                f"<td style='text-align:center'>{status_badge(d.get('status'))}</td>"
                f"</tr>"
            )
        sv_rows = _sv_row(sv.get("partner_po")) + _sv_row(sv.get("vendor_quote"))
        subtotal_validation_block = f"""<div class="card">
        <div class="card-title">Subtotal Validation (vs Opportunity)</div>
        <table>
          <thead>
            <tr>
              <th style="width:220px">Check</th>
              <th style="width:140px">PDF Subtotal</th>
              <th style="width:100px">Converted (USD)</th>
              <th style="width:120px">Opportunity Value</th>
              <th style="width:82px;text-align:center">Status</th>
            </tr>
          </thead>
          <tbody>{sv_rows}</tbody>
        </table>
        <p style="font-size:9px;color:#374151;margin-top:6px">Overall: {status_badge(sv.get('overall_status'))}</p>
      </div>"""

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
  /* ── SKU accordion blocks (PDF) ── */
  .sku-block  {{ border: 1px solid #e5e7eb; border-radius: 5px; margin-bottom: 6px; overflow: hidden; }}
  .sku-hdr    {{ display: flex; align-items: center; gap: 8px; padding: 5px 8px;
                 background: #1a1a2e; color: #fff; }}
  .sku-num    {{ font-size: 8px; color: #9ca3af; min-width: 14px; }}
  .sku-code   {{ font-family: monospace; font-size: 9px; font-weight: 700; color: #fff; }}
  .dt         {{ width: 100%; border-collapse: collapse; font-size: 8px; }}
  .dt thead th {{ background: #374151; color: #fff; padding: 4px 6px; text-align: left;
                  font-size: 7px; font-weight: 600; text-transform: uppercase; }}
  .dt tbody td {{ padding: 4px 6px; border-bottom: 1px solid #f3f4f6; vertical-align: middle; }}
  .dt .rl     {{ font-weight: 700; color: #374151; font-size: 8px; white-space: nowrap; }}
  .dt .rl.sub {{ font-weight: 400; color: #6b7280; padding-left: 14px; font-size: 7px; }}
  .dt .cv     {{ font-family: monospace; font-size: 8px; color: #1a1a2e; }}
  .dt .sc     {{ text-align: left; }}
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
  .margin-pass-banner {{
    background: #d1fae5; border: 1px solid #6ee7b7; border-radius: 6px;
    padding: 7px 12px; margin-bottom: 12px; font-size: 9px; color: #065f46;
    display: flex; align-items: center; gap: 16px;
  }}
  .margin-pass-banner .mp-title {{ font-weight: bold; }}
  .margin-pass-banner .mp-stat {{ font-family: monospace; font-weight: 600; }}
  .margin-pass-banner .mp-label {{ font-family: Arial, Helvetica, sans-serif; font-weight: normal; color: #047857; margin-right: 3px; }}
  .margin-needs-review-banner {{
    background: #fef3c7; border: 1px solid #fcd34d; border-radius: 6px;
    padding: 8px 12px; margin-bottom: 12px; color: #92400e;
  }}
  .margin-needs-review-banner .mnr-title {{ font-weight: bold; font-size: 10px; margin-bottom: 3px; }}
  .margin-needs-review-banner .mnr-detail {{ font-size: 9px; color: #78350f; line-height: 1.5; margin-bottom: 6px; }}
  .margin-stats-pdf {{ display: flex; gap: 10px; flex-wrap: wrap; }}
  .margin-stats-pdf .ms-item {{
    background: rgba(255,255,255,0.6); border-radius: 5px; padding: 4px 9px;
    display: flex; flex-direction: column; gap: 1px; min-width: 90px;
  }}
  .margin-stats-pdf .ms-label {{ font-size: 7px; text-transform: uppercase; letter-spacing: 0.05em; color: #78350f; font-weight: 700; }}
  .margin-stats-pdf .ms-value {{ font-size: 10px; font-weight: 700; color: #1a1a2e; font-family: monospace; }}
</style>
</head>
<body>
  <div class="header">
    <h1>Procurement Analysis Report</h1>
    <div class="subtitle">Quote: {quote_subject} &nbsp;|&nbsp; Generated: {generated_at}{by_line}</div>
  </div>
  {margin_status_block}
  {currency_block}
  {header_validation_block}
  <div class="banner">
    <div class="banner-title">{result.get("final_call","")}</div>
  </div>
  <div class="card">
    <div class="card-title">Section 1 - Three-Way Item Matching</div>
    {sku_blocks}
  </div>
  {subtotal_validation_block}
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

        # ── VALIDATE AI PROMPTS ARE LOADABLE — before any other work. This is
        #    deliberately the very first real step: previously prompts were only
        #    loaded deep into the job (Gemini prompt just before extraction,
        #    Claude prompt inside run_comparison), so a broken AIPrompts CRM
        #    record failed only after the margin gate AND both PDF downloads had
        #    already run — slow to fail, and the underlying error message was
        #    also being swallowed by _refresh_prompt_cache's fallback (fixed
        #    separately). Failing here means it fails fast, cheap, and with the
        #    real cause. This also warms the 5-min cache so the later calls to
        #    load_gemini_prompt()/load_claude_prompt() are free (no extra CRM hit).
        #    No dedicated phase shown for this — runs silently under "Initialising...".
        load_gemini_prompt()
        load_claude_prompt()
        print(f"[{job_id}] ⏱ Prompts verified: {time.time()-t0:.1f}s")

        jobs[job_id]["phase"] = "Fetching quote from Zoho..."

        if is_cancelled(job_id): return
        quote = fetch_zoho_quote(quote_id, token)
        print(f"[{job_id}] ⏱ Fetch quote: {time.time()-t0:.1f}s")
        print(f"Quote fields: {list(quote.keys())}")

        # ── MARGIN GATE — first check after reading the quote, before any
        #    PDF download/extraction/comparison. Gross Margin % (from the
        #    Opportunity) vs Vendor Margin % (from the Vendor). NON-BLOCKING:
        #    if Gross Margin is lower, this is surfaced as a "Needs Review"
        #    banner (widget + PDF report) — document comparison still runs in
        #    full underneath it. This used to short-circuit the whole job; that
        #    behavior was intentionally removed — see result["margin_gate"]
        #    below for how the outcome is now carried through instead.
        if is_cancelled(job_id): return
        jobs[job_id]["phase"] = "Checking vendor & opportunity margins..."
        margin = check_margin_gate(quote, token)

        if margin["blocked"]:
            print(f"[{job_id}] ⚠️  Margin gate NEEDS REVIEW — Gross {margin['gross_margin']}% < Vendor {margin['vendor_margin']}% — continuing comparison anyway")
        elif margin["skipped_reason"]:
            print(f"[{job_id}] ℹ️  Margin gate skipped ({margin['skipped_reason']}) — continuing to document comparison")
        else:
            print(f"[{job_id}] ✅ Margin gate passed — Gross {margin['gross_margin']}% >= Vendor {margin['vendor_margin']}%")

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

            ppo_text, ppo_header = future_ppo.result()
            vq_text, vq_header   = future_vq.result()

        print("VQ TEXT from Gemini:"+vq_text)
        print("Partner PO TEXT from Gemini:"+ppo_text)
        print(f"[{job_id}] ⏱ Gemini done: {time.time()-t0:.1f}s")
        jobs[job_id]["phase"] = "Comparing documents with Claude AI..."

        if is_cancelled(job_id): return
        result = run_comparison(zoho_text, ppo_text, vq_text, job_id)
        print(f"The final result from claude is : {result}")

        # Surface the margin gate outcome to the widget/PDF report. "checked" is
        # False if the gate was skipped (missing data / config issue) so the
        # widget shows nothing. "needs_review" is True when Gross Margin was
        # lower than Vendor Margin — this NO LONGER stops the pipeline (see the
        # Margin Gate block above); it's shown as a non-blocking banner instead,
        # with the full comparison result rendered underneath it as usual.
        result["margin_gate"] = {
            "checked":          margin["skipped_reason"] is None,
            "needs_review":     margin["blocked"],
            "opportunity_name": margin["opportunity_name"],
            "vendor_name":      margin["vendor_name"],
            "gross_margin":     margin["gross_margin"],
            "vendor_margin":    margin["vendor_margin"],
        }

        # Subtotal Validation — Partner PO / Vendor Quote subtotals (converted to
        # USD) vs the Opportunity's Amount_in_USD / Net_to_Vendor. Deterministic,
        # computed here in Python, not by Claude. Reporting-only — never blocks
        # the pipeline and doesn't factor into final_call.
        result["subtotal_validation"] = check_subtotal_validation(quote, margin, ppo_header, vq_header)

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


@app.get("/inspect-quote/{quote_id}")
def inspect_quote(quote_id: str):
    """
    DEBUG — hit this once with a real quote_id to confirm the margin-gate
    config constants (top of file) match your org before trusting the gate
    in production. Shows the raw Vendor / Opportunity field values on the
    quote, the full margin-gate resolution, and every field name on the quote
    so you can spot the correct Opportunity field if OPPORTUNITY_FIELD_ON_QUOTE
    is wrong. Remove or protect this endpoint once confirmed.
    """
    try:
        token = get_access_token()
        quote = fetch_zoho_quote(quote_id, token)

        result = {
            "config": {
                "OPPORTUNITY_FIELD_ON_QUOTE": OPPORTUNITY_FIELD_ON_QUOTE,
                "OPPORTUNITIES_MODULE":       OPPORTUNITIES_MODULE,
                "OPPORTUNITIES_NAME_FIELD":   OPPORTUNITIES_NAME_FIELD,
                "GROSS_MARGIN_FIELD":         GROSS_MARGIN_FIELD,
                "VENDORS_MODULE":             VENDORS_MODULE,
                "VENDORS_NAME_FIELD":         VENDORS_NAME_FIELD,
                "VENDOR_MARGIN_FIELD":        VENDOR_MARGIN_FIELD,
            },
            "raw_vendor_value":       quote.get("Vendor"),
            "raw_opportunity_value":  quote.get(OPPORTUNITY_FIELD_ON_QUOTE),
            "all_quote_field_names":  sorted(quote.keys()),
        }

        try:
            result["margin_gate_result"] = check_margin_gate(quote, token)
        except Exception as e:
            result["margin_gate_error"] = str(e)

        return result
    except Exception as e:
        return {"error": str(e)}


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