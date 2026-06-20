import anthropic
import requests
import uuid
import json
import io
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
PPO_PDF_FIELD   = os.environ.get("PPO_PDF_FIELD", "PPO_PDF_FIELD")
VQ_PDF_FIELD   = os.environ.get("VQ_PDF_FIELD", "VQ_PDF")
ACCESS_TOKEN_URL  = os.environ.get("ACCESS_TOKEN_URL", "https://accounts.zoho.com/oauth/v2/token")
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
    items = quote.get("Quoted_Items", [])
    print(f"✅ Zoho quote has {items} ")
    print(f"✅ Zoho quote has {len(items)} line items")
    for i, item in enumerate(items, 1):
        sku  = item.get("Product_Name", {}).get("Product_Code", "N/A")
        desc = item.get("Description", "N/A")
        qty  = item.get("Quantity", "N/A")
        quote_currency  = quote.get("Currency", {})
        Partner_PO_Currency = quote.get("Partner_PO_Currency", "N/A")
        Vendor_Quote_Currency  = quote.get("Vendor_Quote_Currency", "N/A")
        Exchange_Rate= quote.get("Exchange_Rate", "N/A")
        buy_price_zq=item.get("Buy_Price", "N/A")
        list_price_zq=item.get("List_Price", "N/A")

        lines.append(f"  {i}. SKU         : {sku}")
        lines.append(f"     Description : {desc}")
        lines.append(f"     Quantity    : {qty}")
        lines.append(f"     Quote_Currency         : {quote_currency}")
        lines.append(f"     Partner_PO_Currency : {Partner_PO_Currency}")
        lines.append(f"     Vendor_Quote_Currency    : {Vendor_Quote_Currency}")
        lines.append(f"     Exchange_Rate    : {Exchange_Rate}")
        lines.append(f"     buy_price_zq   : {buy_price_zq}")
        lines.append(f"     list_price_zq    : {list_price_zq}")
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
def extract_pdf_gemini(pdf_bytes: bytes, label: str, model_name: str, job_id: str = None) -> str:
    if job_id and is_cancelled(job_id):
        print(f"🔍 Cancelled before {label} extraction")
        raise Exception("Job cancelled by user")

    model = genai.GenerativeModel(
        model_name=model_name,
        generation_config={"temperature": 0, "response_mime_type": "application/json"}
    )

    prompt = """
Extract ALL line items from this procurement document, including duplicates.

IMPORTANT RULES:
1.⁠ ⁠Extract EVERY line item row — even if it looks identical to another row
2.⁠ ⁠Do NOT deduplicate or merge rows — if the same SKU appears 3 times, return 3 separate entries
3.⁠ ⁠Do NOT assume repeated rows are errors — they represent separate line items
4.⁠ ⁠Each physical row in the document = one entry in your output
5.⁠ ⁠Only skip rows that are clearly headers, totals, subtotals, taxes, shipping, or page numbers

For each line item return exactly these fields:
•⁠  ⁠line_num   : line number as it appears in the document (if none, use sequential 1, 2, 3...)
•⁠  ⁠sku        : product code or SKU exactly as written (use null if missing or not present)
•⁠  ⁠description: full product description exactly as written, no truncation
•⁠  ⁠quantity   : numeric value only, no units (use null if not present or not readable)
•⁠  list_unit_price  : numeric value only, no units (use null if not present or not readable)

Do NOT include: totals, taxes, dates, addresses, payment terms, or currency.

Return ONLY a valid JSON array. No explanation, no markdown, no backticks.

Example output format:
[
  {"line_num": 1, "sku": "FG-100F", "description": "FortiGate 100F Hardware", "quantity": 2, list_unit_price":4500},
  {"line_num": 2, "sku": null, "description": "FortiGate 100F Hardware", "quantity": 1, list_unit_price":null}
]
"""

    print(f"🔍 Extracting {label} via Gemini ({model_name})...")
    response = model.generate_content(
        [{"mime_type": "application/pdf", "data": pdf_bytes}, prompt],
        request_options={"timeout": 120}
    )

    raw = response.text
    print(f"✅ Gemini {label}: {len(raw)} chars")

    try:
        clean = raw.replace("```json", "").replace("```", "").strip()
        items = json.loads(clean)
        print(f"✅ Extracted {len(items)} items from {label}")

        # Formatted text — faster for Claude to process than minified JSON
        lines = [f"## {label}"]
        for item in items:
            lines.append(f"  {item.get('line_num','')}. SKU: {item.get('sku','N/A')} | Desc: {item.get('description','N/A')} | Qty: {item.get('quantity','N/A')} | list_unit_price: {item.get('list_unit_price','N/A')}")
        return "\n".join(lines)

    except json.JSONDecodeError as e:
        print(f"⚠️  Gemini JSON parse error: {e}")
        return raw


# ─────────────────────────────────────────────
# 9. MATCHING PROMPT
# ─────────────────────────────────────────────
MATCHING_PROMPT = """
You are a procurement document analyst for Cyberknight Technologies, a cybersecurity distributor.

Three sources of data are provided:
•⁠  ⁠Zoho Quote (ZQ) — the internal quote record from Cyberknight's CRM (text format)
•⁠  ⁠Vendor Quote (VQ) — what Cyberknight is BUYING from the vendor (JSON extracted line items)
•⁠  ⁠Partner PO — what the partner is BUYING from Cyberknight (JSON extracted line items)

Your job is to compare ALL THREE for SKU code + product description + quantity alignment, AND to validate pricing.
Reference numbers, addresses, and payment terms WILL differ — do not flag those.

---

## STEP 1 — CONSOLIDATE EACH DOCUMENT INDEPENDENTLY

Before any comparison, consolidate line items within EACH document separately.
Do NOT consolidate across documents.

### Consolidation Rules:

For each document independently:

1.⁠ ⁠Group rows by SKU where SKU is not null.

2.⁠ ⁠For rows where SKU is null:
   - Compare description to rows that have a SKU using this rule:
     - A null-SKU row is the SAME product as a SKU row ONLY IF all three of the following are true:
       a. The root product name is identical (e.g. both say "FortiGate 100F")
       b. The vendor/brand name is identical (e.g. both say "Fortinet")
       c. The tier, size, or license level indicator is identical (e.g. both say "UTP" or both say "Enterprise")
     - If ALL THREE match → same product → add quantity to the parent SKU row
     - If ANY ONE of the three does not match → keep as a separate line item
     - If null-SKU row has no description or description is too vague to evaluate → keep separate, set status to "Needs Review", note "Description insufficient for consolidation"

3.⁠ ⁠For rows where both SKU and description are null → keep as separate line item, flag as "Needs Review", note "Line item unreadable"

4.⁠ ⁠After consolidation, each unique SKU appears ONCE with total summed quantity.

5.⁠ ⁠If quantity is null or blank for any row → treat quantity as "?" and do not add to totals.

6.⁠ ⁠For pricing, record the unit price per line item after consolidation. Do NOT sum prices.

### Example:
Raw JSON input:

[
  {"line_num":1,"sku":"2256104","description":"SolarWinds Observability A250","quantity":1,"unit":"license","unit_price":500},
  {"line_num":2,"sku":null,"description":"SolarWinds Observability A250","quantity":1,"unit":null,"unit_price":500},
  {"line_num":3,"sku":null,"description":"SolarWinds Observability A250","quantity":1,"unit":null,"unit_price":500}
]

After consolidation:
SKU=2256104, Description="SolarWinds Observability A250", Total Qty=3, Unit Price=500 (unchanged)

---

## STEP 2 — COMPARE CONSOLIDATED TOTALS ACROSS ALL THREE DOCUMENTS

### Description Matching Rules:

Apply these rules in order. Use ONLY these exact status strings — no other values permitted:
"Match" | "Needs Review" | "Mismatch"

Rule 1 — Formatting differences only → "Match"
The following differences do NOT change the status from Match:
•⁠  ⁠Capitalization differences (FORTIGATE vs FortiGate)
•⁠  ⁠Punctuation differences (FortiGate-100F vs FortiGate 100F)
•⁠  ⁠Spacing differences
•⁠  ⁠Common abbreviations (Ent = Enterprise, Std = Standard, HW = Hardware, SW = Software)
•⁠  ⁠Term/period format differences using ONLY these equivalences:
    Y1 = 1YR = 12M = 12MO = Annual = 1Year = One Year = 1-Year
    Y2 = 2YR = 24M = 24MO = 2Year = Two Year = 2-Year
    Y3 = 3YR = 36M = 36MO = 3Year = Three Year = 3-Year
  If a term suffix appears that is NOT in the above list, set status to "Needs Review"

Rule 2 — Ambiguous differences → "Needs Review"
Use this status when:
•⁠  ⁠Descriptions share the same product name but differ in tier, size, or license level
•⁠  ⁠One document bundles items that another lists separately
•⁠  ⁠Term or period cannot be confirmed equivalent using the list above
•⁠  ⁠Unit of measure differs between documents for the same SKU (unless one unit is null)
•⁠  ⁠Quantity is "?" in one or more documents

Rule 3 — Clear discrepancy → "Mismatch"
Use this status when:
•⁠  ⁠A SKU or product exists in one document with NO equivalent in another
•⁠  ⁠Quantities differ after consolidation and the difference is not explainable by bundling
•⁠  ⁠Descriptions refer to clearly different products

When writing notes for a Mismatch caused by an absent document, determine the absent document mechanically:
•⁠  ⁠If zq_qty is null → absent from ZQ
•⁠  ⁠If vq_qty is null → absent from VQ
•⁠  ⁠If ppo_qty is null or "-" → absent from Partner PO
Always write: "Item present in [X] and [Y] but absent from [Z]" where Z is the document with null quantity. Never reverse this.
Rule 4 — SKU discrepancy between documents → "Needs Review"
If the same product is matched by description but carries a different SKU
in one or more documents (e.g. CS.INSIGHTB.SOLN vs CS.INSIGHT.SOLN):
  - Set all status fields to "Needs Review"
  - Note: "SKU discrepancy: [doc A] uses [SKU1], [doc B] uses [SKU2] — 
    confirm these are the same product"
  - Still record and compare quantities and prices as normal
  - Do NOT flag as Mismatch unless quantities or prices also differ
--

## STEP 3 — PRICE COMPARISON

Two comparisons must be performed per line item.

### Price field mapping:

| Comparison | ZQ field | ZQ currency | External document | External currency |
|---|---|---|---|---|
| A — Sell side | ⁠ list_price ⁠ | ⁠ quote_currency ⁠ | Partner PO ⁠ unit_price ⁠ | ⁠ partner_po_currency ⁠ |
| B — Buy side | ⁠ buy_price ⁠ | ⁠ quote_currency ⁠ | VQ ⁠ unit_price ⁠ | ⁠ vendor_quote_currency ⁠ |

---

### Step 3a — Identify currencies

Read the three currency fields from ZQ:
•⁠  ⁠⁠ quote_currency ⁠ — applies to both ⁠ list_price ⁠ and ⁠ buy_price ⁠ in ZQ
•⁠  ⁠⁠ partner_po_currency ⁠ — currency of Partner PO unit prices
•⁠  ⁠⁠ vendor_quote_currency ⁠ — currency of VQ unit prices

---

### Step 3b — Apply conversion direction

*Comparison A — Sell side (List Price vs. Partner PO)*
The ZQ list price is the reference. Convert the Partner PO price INTO ⁠ quote_currency ⁠, then compare.

•⁠  ⁠If ⁠ partner_po_currency ⁠ = ⁠ quote_currency ⁠ → no conversion needed; compare directly
•⁠  ⁠If currencies differ → convert Partner PO price to ⁠ quote_currency ⁠ using the rates below, then compare to ZQ ⁠ list_price ⁠

*Comparison B — Buy side (Buy Price vs. Vendor Quote)*
The VQ is the reference. Convert the ZQ buy price INTO ⁠ vendor_quote_currency ⁠, then compare.

•⁠  ⁠If ⁠ vendor_quote_currency ⁠ = ⁠ quote_currency ⁠ → no conversion needed; compare directly
•⁠  ⁠If currencies differ → convert ZQ ⁠ buy_price ⁠ to ⁠ vendor_quote_currency ⁠ using the rates below, then compare to VQ ⁠ unit_price ⁠

*Example:* ZQ buy price = AED 90, VQ currency = USD
→ Convert: 90 ÷ 3.6725 = USD 24.51
→ Compare USD 24.51 to VQ unit price
→ If VQ unit price = USD 24.51 → "Price Match"

---

### Step 3c — Fixed exchange rates

| From | To | Operation |
|------|----|-----------|
| USD | AED | × 3.6725 |
| USD | SAR | × 3.7500 |
| USD | QAR | × 3.6500 |
| AED | USD | ÷ 3.6725 |
| SAR | USD | ÷ 3.7500 |
| QAR | USD | ÷ 3.6500 |

For cross-rates not listed above (e.g. AED to SAR):
Convert source currency → USD first, then USD → target currency.

Always record:
•⁠  The original price in its source currency
•⁠  c⁠The converted price in the target currency
•⁠  ⁠The exchange rate and operation applied

---

### Step 3d — Compare and assign price status

Round all converted values to 2 decimal places before comparing.
A tolerance of ±0.50 in the comparison currency is permitted to absorb rounding differences.

Use ONLY these exact strings for price status:
"Price Match" | "Price Mismatch" | "Price Needs Review"

•⁠  ⁠If prices are within ±0.50 of each other → "Price Match"
•⁠  ⁠If prices differ by more than ±0.50 → "Price Mismatch"
•⁠  ⁠If unit price is null or missing in any document → "Price Needs Review", note "Price missing in [document name]"
•⁠  ⁠If currency is not one of USD, AED, SAR, QAR → "Price Needs Review", note "Unsupported currency: [currency code]"

---

## STEP 4 — DETERMINE FINAL CALL

Apply these rules in strict order — do not use judgement, apply mechanically:

Rule 1: If ANY item has ANY status field = "Mismatch" OR any price status = "Price Mismatch"
         → final_call = "ON HOLD — MISMATCH"

Rule 2: If no Mismatch exists but ANY item has ANY status field = "Needs Review" OR any price status = "Price Needs Review"
         → final_call = "QUERY TO SP"

Rule 3: If ALL status fields across ALL items = "Match" AND all price statuses = "Price Match"
         → final_call = "CLEAR TO PROCESS"

No exceptions. No other final_call values are permitted.

---

## OUTPUT FORMAT

Return ONLY a valid JSON object. No explanation, no markdown, no backticks.
Use ONLY these exact strings for all status fields: "Match" | "Needs Review" | "Mismatch"
Use ONLY these exact strings for all price status fields: "Price Match" | "Price Mismatch" | "Price Needs Review"

{
  "currencies_detected": {
    "quote_currency": "currency code from ZQ",
    "partner_po_currency": "currency code from Partner PO",
    "vendor_quote_currency": "currency code from VQ"
    "notes":" A 2 to 3 line description of what exchange rates were used for conversion and comparison  with an example in case there is a conversion required otherwise no description needed"
  },
  "matching_table": [
    {
      "num": 1,
      "sku": "consolidated SKU — use ZQ SKU if available, else VQ, else Partner PO",
      "description": "consolidated description",

      "zq_qty": "quantity from ZQ after consolidation, or null if not present",
      "zq_status": "Match|Needs Review|Mismatch",

      "vq_qty": "quantity from VQ after consolidation, or null if not present",
      "vq_status": "Match|Needs Review|Mismatch",

      "ppo_qty": "quantity from Partner PO after consolidation, or null if not present",
      "ppo_status": "Match|Needs Review|Mismatch",

      "list_price_zq": "list price from ZQ in quote_currency, or null",
      "partner_ppo_price_original": "unit price from Partner PO in partner_po_currency, or null",
      "partner_ppo_price_converted": "partner_po_price converted to quote_currency, or null if no conversion needed",
      "list_price_comparison_status": "Price Match|Price Mismatch|Price Needs Review",

      "buy_price_zq": "buy price from ZQ in quote_currency, or null",
      "buy_price_zq_converted": "buy_price_zq converted to vendor_quote_currency, or null if no conversion needed",
      "vendor_quote_price": "unit price from VQ in vendor_quote_currency, or null",
      "buy_price_comparison_status": "Price Match|Price Mismatch|Price Needs Review",

      "exchange_rate_used": "e.g. '1 USD = 3.6725 AED' or 'AED ÷ 3.6725 = USD' or null if no conversion needed",
      "notes": "specific reason for any non-Match or non-Price Match status, or null if all match"
    }
  ],
  "unmatched_items": [
      "List each item that is missing from one or more documents. Format: [ABSENT FROM: {document name}] SKU/Description — present in {other documents}. Example: [ABSENT FROM: Partner PO] GIB-DRP-L-WEB-BRANCH-1 — present in ZQ and VQ"
      "Small description of why price mismatch"
    ],
  "needs_review": [
    "List each item flagged Needs Review or Price Needs Review. Format: SKU/Description — specific reason"
  ],
  "must_resolve": [
    "List each item blocking processing. Format: SKU/Description — specific action required to resolve"
  ],
  "overall_summary": "X of Y items Match. Z items Mismatch. W items Needs Review. A items Price Mismatch. B items Price Needs Review.",
  "final_call": "CLEAR TO PROCESS|QUERY TO SP|ON HOLD — MISMATCH",
  "final_call_detail": [
    "One line per blocking or review item explaining exactly what needs to be resolved before processing"
  ]
}
 ⁠

---

## FIELD DEFINITIONS

currencies_detected          — the three currency codes read from ZQ; drives all conversion logic
matching_table               — one row per unique consolidated SKU/product across all three documents
unmatched_items              — items present in only one document, no equivalent found in others
needs_review                 — items with ambiguity in description, quantity, or price that cannot be auto-resolved
must_resolve                 — any item that prevents final_call from being CLEAR TO PROCESS
overall_summary              — always use the exact format shown above
final_call                   — determined mechanically by Step 4 rules only
final_call_detail            — empty array [] if final_call is CLEAR TO PROCESS
buy_price_zq_converted       — ZQ buy price expressed in vendor_quote_currency; this is what gets compared to VQ unit price
partner_po_price_converted   — Partner PO price expressed in quote_currency; this is what gets compared to ZQ list price

"""


# ─────────────────────────────────────────────
# 10. CLAUDE COMPARISON
#     OPT 2: Auto model selection based on size
#     OPT 1: Receives compact JSON (fewer tokens)
# ─────────────────────────────────────────────
def run_comparison(zoho_text: str, ppo_text: str, vq_text: str, job_id: str = None) -> dict:
    if job_id and is_cancelled(job_id):
        raise Exception("Job cancelled by user")

    # Always use Sonnet with temperature=0 for consistency
    model_name = "claude-sonnet-4-6"
    max_tokens = 32000
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
                {"type": "text", "text": "## ⁠Partner PO (PO):\n\n" + ppo_text + "\n\n---"},
                {"type": "text", "text": MATCHING_PROMPT}
            ]
        }]
    ) as stream:
        for text_chunk in stream.text_stream:
            full_text += text_chunk
            if job_id and is_cancelled(job_id):
                print(f"[{job_id}] Cancelled during Claude streaming")
                raise Exception("Job cancelled during Claude streaming")

    message = stream.get_final_message()
    print(f"Claude response: {len(full_text)} chars | Stop: {message.stop_reason}")

    if message.stop_reason == "max_tokens":
        raise Exception("Claude truncated — increase max_tokens")
    print("CLAUDE RESPONSE:{full_text}")
    clean = full_text.replace("```json", "").replace("```", "").strip()
    return json.loads(clean)


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
        # Substring matching (checked in this order) so this works for both
        # qty statuses ("Match"/"Needs Review"/"Mismatch") and price statuses
        # ("Price Match"/"Price Needs Review"/"Price Mismatch") — "mismatch"
        # must be checked before "match" since it contains "match" as a substring.
        if "mismatch" in s: return '<span class="pill pill-miss">Mismatch</span>'
        if "review" in s:   return '<span class="pill pill-review">Review</span>'
        if "match" in s:    return '<span class="pill pill-match">Match</span>'
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

    if job_id and is_cancelled(job_id):
        raise Exception("Job cancelled by user")

    table_rows = ""
    for i, r in enumerate(result.get("matching_table", [])):
        row_bg = "#ffffff" if i % 2 == 0 else "#f9fafb"
        table_rows += f"""<tr style="background:{row_bg}">
            <td style="text-align:center;font-weight:600">{r.get("num","")}</td>
            <td style="font-family:monospace;font-size:9px;color:#374151;word-break:break-all">{r.get("sku","") or "None"}</td>
            <td style="text-align:center">{r.get("ppo_qty","") or "-"}</td>
            <td style="text-align:center;font-weight:600">{r.get("zq_qty","") or "-"}</td>
            <td style="text-align:center">{status_badge(r.get("zq_status"))}</td>
            <td style="text-align:center;font-weight:600">{r.get("vq_qty","") or "-"}</td>
            <td style="text-align:center">{status_badge(r.get("ppo_status"))}</td>
            <td style="text-align:center">{status_badge(r.get("vq_status"))}</td>
            <td style="text-align:center">{status_badge(r.get("list_price_comparison_status"))}</td>
            <td style="text-align:center">{status_badge(r.get("buy_price_comparison_status"))}</td>
            <td style="font-size:9px;color:#6b7280;line-height:1.4">{r.get("notes","")}</td>
        </tr>"""

    if job_id and is_cancelled(job_id):
        raise Exception("Job cancelled by user")

    must_resolve = "".join([f'<li class="item-red">{i}</li>' for i in (result.get("must_resolve") or [])]) or "<li>None</li>"
    needs_review = "".join([f'<li class="item-amber">{i}</li>' for i in (result.get("needs_review") or [])]) or "<li>None</li>"
    unmatched    = "".join([f'<span class="tag">{i}</span>' for i in (result.get("unmatched_items") or [])])
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
  <div class="banner">
    <div class="banner-title">{result.get("final_call","")}</div>
   <!-- <ul>{fc_details}</ul> -->
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

        # Gemini extractions in parallel
        print(f"[{job_id}] 🔍 Gemini extraction (parallel)...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            future_ppo = executor.submit(extract_pdf_gemini, ppo_bytes, "Partner PO PDF", gemini_model, job_id)
            future_vq = executor.submit(extract_pdf_gemini, vq_bytes, "VQ PDF", gemini_model, job_id)

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
    jobs[job_id]      = {"status": "processing"}
    last_poll[job_id] = time.time()

    background_tasks.add_task(process_quote_job, job_id, quote_id, initiated_by)

    threading.Thread(
        target=auto_cancel_watcher,
        args=(job_id, 6),
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
