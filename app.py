from flask import Flask, request, jsonify
import os
import json
import logging
import requests as req
import subprocess, tempfile
from dotenv import load_dotenv
from groq import Groq

# Import from functions
from function.database import save_call_analysis, get_call_analysis

load_dotenv()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ─────────────────────────────────────────────
# FEATURE FLAG
#   USE_WHISPER=true  → Whisper (Cloudflare tunnel) + Groq Hindi correction
#   USE_WHISPER=false → Sarvam AI (default)
# ─────────────────────────────────────────────

USE_WHISPER = "true"
logger.info(f"Transcription backend: {'Whisper' if USE_WHISPER else 'Sarvam AI'}")

# ─────────────────────────────────────────────
# EXOTEL CONFIG
# ─────────────────────────────────────────────

EXOTEL_ACCOUNT_SID = os.getenv("EXOTEL_ACCOUNT_SID")
EXOTEL_API_KEY = os.getenv("EXOTEL_API_KEY")
EXOTEL_API_TOKEN = os.getenv("EXOTEL_API_TOKEN")
EXOTEL_BASE_URL = os.getenv("EXOTEL_BASE_URL", "https://api.exotel.com/v1/Accounts")

# Initialize Groq client
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))


# ─────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────


def _cleanup(path: str):
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception:
        pass

# ─────────────────────────────────────────────
# TRANSCRIPTION  (inline — flag-based)
# ─────────────────────────────────────────────


def _chunk_audio(file_path: str, chunk_duration: int = 29):

    output_dir = tempfile.mkdtemp()
    chunk_pattern = os.path.join(output_dir, "chunk_%03d.mp3")
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            file_path,
            "-f",
            "segment",
            "-segment_time",
            str(chunk_duration),
            "-acodec",
            "copy",
            chunk_pattern,
        ],
        capture_output=True,
        check=True,
    )
    chunks = sorted(
        [
            os.path.join(output_dir, f)
            for f in os.listdir(output_dir)
            if f.startswith("chunk_")
        ]
    )
    return chunks, output_dir


def _cleanup_dir(path: str):
    try:
        if os.path.exists(path):
            os.rmdir(path)
    except Exception:
        pass


def _transcribe_sarvam(file_path: str, language_code: str = "unknown"):
    from sarvamai import SarvamAI

    sarvam_client = SarvamAI(api_subscription_key=os.getenv("SARVAM_API_KEY"))
    chunks, tmp_dir = _chunk_audio(file_path)
    full_transcript = []
    detected_language = None
    try:
        for chunk_path in chunks:
            with open(chunk_path, "rb") as audio_file:
                response = sarvam_client.speech_to_text.transcribe(
                    file=("audio.mp3", audio_file, "audio/mpeg"),
                    model="saaras:v3",
                    mode="transcribe",
                    language_code=language_code,
                )
            full_transcript.append(response.transcript)
            if not detected_language and hasattr(response, "language_code"):
                detected_language = response.language_code
    finally:
        for f in chunks:
            _cleanup(f)
        _cleanup_dir(tmp_dir)
    return " ".join(full_transcript).strip(), detected_language


def _transcribe_whisper(file_path: str, language_code: str = "unknown"):
    whisper_url = os.getenv("WHISPER_URL")
    if not whisper_url:
        raise RuntimeError(
            "WHISPER_URL is not set in .env. "
            "Add: WHISPER_URL=https://your-tunnel.trycloudflare.com/transcribe"
        )
    logger.info(f"Sending audio to Whisper: {whisper_url}")

    if not os.path.exists(file_path):
        raise RuntimeError(f"Audio file not found: {file_path}")
    file_size = os.path.getsize(file_path)
    if file_size == 0:
        raise RuntimeError("Audio file is empty (0 bytes)")

    try:
        with open(file_path, "rb") as f:
            response = req.post(
                whisper_url,
                files={"file": ("audio.mp3", f, "audio/mpeg")},
                timeout=300,
            )
    except req.exceptions.ConnectionError as e:
        raise RuntimeError(
            f"Cannot reach Whisper server at {whisper_url}. "
            f"Make sure your Colab notebook is running. Error: {e}"
        )
    except req.exceptions.Timeout:
        raise RuntimeError(f"Whisper timed out after 300s ({file_size} bytes).")

    logger.info(f"Whisper response status: {response.status_code}")
    if response.status_code != 200:
        raise RuntimeError(
            f"Whisper returned HTTP {response.status_code}: {response.text[:300]}"
        )

    try:
        data = response.json()
    except Exception:
        raise RuntimeError(f"Whisper non-JSON response: {response.text[:300]}")

    if not data.get("success", True):
        raise RuntimeError(f"Whisper failure: {data.get('error', 'unknown')}")

    text = (
        data.get("english")
        or data.get("hindi")
        or data.get("text")
        or data.get("transcript")
        or ""
    ).strip()

    if not text:
        raise RuntimeError(f"Whisper empty transcript. Fields: {list(data.keys())}")

    logger.info(f"Transcript received: {len(text)} chars")
    return text, "hi-IN"


HINDI_CORRECTION_PROMPT = """You are an expert Hindi language editor specializing in correcting speech-to-text transcripts from Indian phone call recordings for a pharmaceutical delivery software company.

BUSINESS CONTEXT:
- This is a call between delivery staff and a support/manager
- The company manages medicine delivery orders for medical stores
- Callers speak Hindi mixed with Gujarati words
- Common topics: order assignments, delivery areas, medical stores, staff issues

KNOWN VOCABULARY — always correct to these exact spellings when phonetically similar:
- विजयनगर (area name) ← fixes: विजेइनगर, उजे नगर, विजेईगर, विजेनगर, वीजेनगर
- बातकर मेडिकल (store name) ← fixes: बाजकर, वाजकर, बातकर, बाटकर
- तरुण मेडिकल (store name) ← fixes: तरुन, तरुण, टरुन
- पासकर मेडिकल (store name) ← fixes: पास कर, पास्कर, पाशकर
- प्रशांत (person name) ← fixes: प्रसांत, प्रसांत्, परशांत
- ऑर्डर ← fixes: ओडर, ओर्डर, ओड़र
- मेडिकल स्टोर ← fixes: मेडिकल स्ट्रायलेक्शन, मेडिकल स्टायर, मेडिकल सटोर
- ओके ओके ← keep as is (agreement)
- हाँ हाँ ← fixes: गैस गैस, यस यस (Gujarati "haa haa" misheard)

CORRECTION RULES:
1. Fix all words from KNOWN VOCABULARY list above — this is your highest priority
2. Fix obvious transcription errors using surrounding context
3. ONLY fix spelling mistakes. Do NOT change sentence structure.
4. If input is repetitive or broken, keep it as-is.
5. Add basic punctuation: । for sentence end, ? for questions, ... for hesitation pauses
6. Keep English words that were spoken in English (e.g. "OK", "order")
7. Do NOT translate anything to English
8. Do NOT add any explanation — return ONLY the corrected Hindi text
9. If unsure about a word, keep the original

Return ONLY the corrected Hindi transcript."""


def correct_hindi_transcript(raw_hindi: str) -> str:
    if not raw_hindi or len(raw_hindi.strip()) < 10:
        return raw_hindi
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": HINDI_CORRECTION_PROMPT},
                {
                    "role": "user",
                    "content": f"Correct this Hindi transcript:\n\n{raw_hindi}",
                },
            ],
            temperature=0.1,
            max_completion_tokens=2048,
        )
        corrected = response.choices[0].message.content.strip()
        if len(corrected) < len(raw_hindi) * 0.3:
            logger.warning("Hindi correction suspiciously short, using raw")
            return raw_hindi
        logger.info("Hindi transcript corrected successfully")
        return corrected
    except Exception as e:
        logger.warning(f"Hindi correction failed, using raw: {e}")
        return raw_hindi


def transcribe_audio(file_path: str, language_code: str = "unknown"):
    """
    Unified entry point. Returns (raw_transcript, final_transcript, detected_lang).
    When Whisper: final = Groq-corrected version of raw.
    When Sarvam:  raw == final.
    """
    if USE_WHISPER:
        raw, lang = _transcribe_whisper(file_path, language_code)
        final = correct_hindi_transcript(raw)
        return raw, final, lang
    else:
        transcript, lang = _transcribe_sarvam(file_path, language_code)
        return transcript, transcript, lang


# ─────────────────────────────────────────────
# EXOTEL API HELPER
# ─────────────────────────────────────────────


def _exotel_auth():
    """Return (auth_tuple, base_url_for_account)."""
    return (
        (EXOTEL_API_KEY, EXOTEL_API_TOKEN),
        f"{EXOTEL_BASE_URL}/{EXOTEL_ACCOUNT_SID}",
    )


def fetch_exotel_calls(
    date_from: str = None,
    date_to: str = None,
    page_size: int = 50,
    page_uri: str = None,
) -> dict:
    """
    Fetch call list from Exotel API.
    Returns the full parsed JSON (Metadata + Calls).

    Args:
        date_from : ISO datetime string  e.g. "2026-03-14 17:21:14"
        date_to   : ISO datetime string  e.g. "2026-04-14 17:21:13"
        page_size : records per page (max 50)
        page_uri  : pass NextPageUri / PrevPageUri to paginate
    """
    auth, base = _exotel_auth()

    if page_uri:
        # Exotel returns relative URIs — prepend the host
        host = EXOTEL_BASE_URL.replace("/v1/Accounts", "")
        url = f"{host}{page_uri}"
    else:
        url = f"{base}/Calls.json"

    params = {"PageSize": page_size, "SortBy": "DateCreated:desc"}
    if date_from and date_to:
        params["DateCreated"] = f"gte:{date_from};lte:{date_to}"

    response = req.get(
        url, auth=auth, params=params if not page_uri else {}, timeout=30
    )
    response.raise_for_status()
    return response.json()


def download_recording(recording_url: str, dest_path: str):
    """Download an Exotel recording MP3 to dest_path."""
    auth, _ = _exotel_auth()
    r = req.get(recording_url, auth=auth, timeout=120, stream=True)
    r.raise_for_status()
    with open(dest_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)


def process_call(call: dict, call_type: str = "general") -> dict:
    """
    Download, transcribe, analyse and save a single Exotel call dict.
    Returns a result summary dict.
    """
    call_sid = call["Sid"]
    recording_url = call.get("RecordingUrl", "")
    from_number = call.get("From", "")
    direction = call.get("Direction", "")
    duration = call.get("Duration", 0)

    if not recording_url:
        return {"call_sid": call_sid, "skipped": True, "reason": "no_recording"}

    tmp_path = os.path.join(UPLOAD_FOLDER, f"{call_sid}.mp3")
    try:
        # 1. Download
        download_recording(recording_url, tmp_path)
        logger.info(f"Downloaded recording for {call_sid} ({duration}s)")

        # 2. Transcribe
        raw_transcript, final_transcript, detected_lang = transcribe_audio(tmp_path)
        logger.info(f"Transcribed {call_sid}: {len(final_transcript)} chars")

        # 3. Analyse
        analysis = analyze_text(final_transcript, call_type)
        logger.info(f"Analysed {call_sid}")

        # 4. Save
        save_call_analysis(
            call_sid, from_number, call_type, final_transcript, detected_lang, analysis
        )

        result = {
            "call_sid": call_sid,
            "skipped": False,
            "from": from_number,
            "direction": direction,
            "duration": duration,
            "language": detected_lang,
            "transcript": final_transcript,
            "analysis": analysis,
            "transcription_backend": "whisper" if USE_WHISPER else "sarvam",
        }
        if USE_WHISPER and final_transcript != raw_transcript:
            result["transcript_raw"] = raw_transcript

        return result

    except Exception as e:
        logger.error(f"Failed to process {call_sid}: {e}")
        return {"call_sid": call_sid, "skipped": True, "reason": str(e)}

    finally:
        _cleanup(tmp_path)


# ─────────────────────────────────────────────
# ANALYSIS PROMPTS & FUNCTIONS
# ─────────────────────────────────────────────

STANDARD_SYSTEM_PROMPT = """You are an expert multilingual call center analyst for a pharmaceutical software company (Pillo App / Evital Software).

Your job has TWO steps:
1. CLEAN & NORMALIZE the transcript (Hindi / Hinglish / noisy ASR text)
2. ANALYZE the cleaned transcript and return ONLY a valid JSON

-----------------------------------
STEP 1: TRANSCRIPT CLEANING RULES
-----------------------------------
- Fix broken, incorrect, or phonetically गलत Hindi words
- Convert Hinglish / mixed language into proper, natural Hindi
- Remove noise, filler, repetition (e.g., "hello hello", "haan bolo", etc.)
- Correct sentence structure while preserving original meaning
- Do NOT add new information
- If a word is unclear, replace it with the most logical word based on context
- Preserve important intent words like: ride, pickup, order, network, payment, etc.
- Output of this step is an internally cleaned transcript (DO NOT print it)

-----------------------------------
STEP 2: CALL ANALYSIS
-----------------------------------

SPEAKER IDENTIFICATION RULES:
- Speaker 1 = CUSTOMER side:
  * "patient" → buys medicine
  * "chemist" → pharmacy owner
  * "rider" → delivery person
- Speaker 2 = SUPPORT AGENT

STRICT RULES:
- Output MUST be valid JSON only (no extra text)
- Do NOT omit any field
- If info missing → use null
- Use ONLY allowed enum values
- Keep summaries concise (2–3 sentences)
- Do NOT hallucinate

-----------------------------------
JSON SCHEMA:
-----------------------------------

{
  "speaker_1": {
    "role": "patient" | "chemist" | "rider",
    "sentiment": "positive" | "negative" | "neutral",
    "emotion": "happy" | "frustrated" | "angry" | "confused" | "satisfied" | "neutral",
    "summary": string
  },
  "speaker_2": {
    "role": "support_agent",
    "sentiment": "positive" | "negative" | "neutral",
    "emotion": "happy" | "frustrated" | "angry" | "confused" | "satisfied" | "neutral",
    "summary": string
  },
  "overall": {
    "sentiment": "positive" | "negative" | "neutral",
    "emotion": "happy" | "frustrated" | "angry" | "confused" | "satisfied" | "neutral",
    "customer_satisfaction": integer (1-10) | null,
    "confidence_score": float (0.0-1.0),
    "summary": string,
    "issue_category": "billing" | "software_error" | "feature_request" | "training_needed" | "complaint" | "payment_issue" | "integration_issue" | "performance_slow" | "operations_issue" | "other" | null,
    "issue_subcategory": string | null,
    "urgency_level": "high" | "medium" | "low" | null,
    "follow_up_required": boolean,
    "follow_up_reason": string | null,
    "suggested_callback_time": string | null,
    "callback_requested_by_customer": boolean,
    "product_module_mentioned": array of strings (use only: "Inventory", "Billing", "Online Ordering", "Reports", "Staff Login"),
    "resolution_status": "resolved" | "partially_resolved" | "unresolved" | null,
    "resolution_summary": string | null,
    "key_customer_concern": string | null
  }
}

-----------------------------------
CONTEXT RULES:
-----------------------------------
- Rider + no orders → issue_category = "operations_issue", subcategory = "no orders assigned"
- Use "software_error" ONLY for bugs/crashes
- Use "performance_slow" ONLY for slowness
- Use "training_needed" if user doesn't know how system works
- Use "feature_request" for new features

-----------------------------------
FINAL INSTRUCTION:
-----------------------------------
- First clean the transcript internally
- Then analyze
- Return ONLY JSON

INPUT TRANSCRIPT:
{transcript}"""

LEAD_GEN_SYSTEM_PROMPT = """..."""  # Keep your full LEAD_GEN_SYSTEM_PROMPT here


def analyze_text(text: str, call_type: str = "general") -> dict:
    if call_type == "lead_gen":
        return _analyze_lead_gen(text)
    return _analyze_standard(text)


def _analyze_standard(text: str) -> dict:
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": STANDARD_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Analyze this customer call transcript:\n\n{text}",
            },
        ],
        temperature=0.1,
        max_completion_tokens=512,
        response_format={"type": "json_object"},
    )
    raw = response.choices[0].message.content
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        cleaned = raw.strip().removeprefix("```json").removesuffix("```").strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return {"error": "Failed to parse response", "raw": raw}


def _analyze_lead_gen(text: str) -> dict:
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": LEAD_GEN_SYSTEM_PROMPT},
            {"role": "user", "content": f"Call transcript:\n\n{text}"},
        ],
        temperature=0.0,
        max_completion_tokens=1024,
        response_format={"type": "json_object"},
    )
    raw = response.choices[0].message.content
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        cleaned = raw.strip().removeprefix("```json").removesuffix("```").strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return {"error": "Failed to parse response", "raw": raw}


# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────


@app.route("/")
def home():
    backend = "Whisper" if USE_WHISPER else "Sarvam AI"
    return f"{backend} + Groq API running"


# ── Manual file upload ────────────────────────
@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "Empty filename"}), 400

    url = request.form.get("url", "");
    call_id = request.form.get("call_id", 0)
    customer_id = request.form.get("customer_id", 0)
    call_type = request.form.get("call_type", "general")
    language = request.form.get("language", "unknown")

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    try:
        raw_transcript, final_transcript, detected_lang = transcribe_audio(
            file_path, language
        )
        logger.info(f"Transcribed call_id={call_id} lang={detected_lang}")

        analysis = analyze_text(final_transcript, call_type)
        logger.info(f"Analysed call_id={call_id} type={call_type}")

        save_call_analysis(
            call_id, customer_id, call_type, final_transcript, detected_lang, analysis
        )

        response_data = {
            "message": "Success",
            "call_type": call_type,
            "transcript": final_transcript,
            "language": detected_lang,
            "analysis": analysis,
            "transcription_backend": "whisper" if USE_WHISPER else "sarvam",
        }
        if USE_WHISPER and final_transcript != raw_transcript:
            response_data["transcript_raw"] = raw_transcript

        return jsonify(response_data)

    except Exception as e:
        logger.error(f"Upload failed for call_id={call_id}: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        _cleanup(file_path)


# ── Exotel real-time webhook ──────────────────
@app.route("/webhook/exotel", methods=["POST"])
def exotel_webhook():
    """Called by Exotel automatically after each call ends."""
    data = request.form.to_dict()
    logger.info(f"Exotel webhook received: {data}")

    recording_url = data.get("RecordingUrl")
    call_sid = data.get("CallSid")
    from_number = data.get("From")
    call_type = request.args.get("call_type", "general")

    if not recording_url or not call_sid:
        return jsonify({"error": "Missing RecordingUrl or CallSid"}), 400

    # Build a minimal call dict and reuse process_call()
    call = {
        "Sid": call_sid,
        "From": from_number,
        "RecordingUrl": recording_url,
        "Direction": data.get("Direction", ""),
        "Duration": data.get("Duration", 0),
    }
    result = process_call(call, call_type)

    if result.get("skipped"):
        return jsonify({"error": result.get("reason")}), 500

    return jsonify({"message": "Processed", **result}), 200


# ── Fetch + process calls from Exotel API ─────
@app.route("/exotel/process-calls", methods=["POST"])
def process_exotel_calls():
    """
    Fetch calls from Exotel API and process recordings.

    POST body (JSON) — all optional:
    {
        "date_from":  "2026-04-14 00:00:00",   // filter start (inclusive)
        "date_to":    "2026-04-14 23:59:59",   // filter end   (inclusive)
        "page_size":  50,                       // 1–50, default 50
        "page_uri":   "/v1/Accounts/...",       // for pagination
        "call_type":  "general",                // analysis type
        "min_duration": 30                      // skip calls shorter than N seconds
    }

    Returns:
    {
        "processed": [...],   // calls that were transcribed & saved
        "skipped":   [...],   // calls with no recording or errors
        "metadata":  {...}    // Exotel pagination metadata
    }
    """
    body = request.get_json(silent=True) or {}
    date_from = body.get("date_from")
    date_to = body.get("date_to")
    page_size = int(body.get("page_size", 50))
    page_uri = body.get("page_uri")
    call_type = body.get("call_type", "general")
    min_duration = int(body.get("min_duration", 0))

    if not EXOTEL_API_KEY or not EXOTEL_API_TOKEN or not EXOTEL_ACCOUNT_SID:
        return (
            jsonify(
                {
                    "error": "Exotel credentials not configured. "
                    "Set EXOTEL_ACCOUNT_SID, EXOTEL_API_KEY, EXOTEL_API_TOKEN in .env"
                }
            ),
            500,
        )

    try:
        # 1. Fetch call list from Exotel
        exotel_response = fetch_exotel_calls(date_from, date_to, page_size, page_uri)
        calls = exotel_response.get("Calls", [])
        metadata = exotel_response.get("Metadata", {})
        logger.info(f"Fetched {len(calls)} calls from Exotel")

        # 2. Filter: only calls with recordings and long enough duration
        eligible = [
            c
            for c in calls
            if c.get("RecordingUrl") and int(c.get("Duration", 0)) >= min_duration
        ]
        skipped_no_rec = [
            {"call_sid": c["Sid"], "reason": "no_recording_or_too_short"}
            for c in calls
            if c not in eligible
        ]
        logger.info(
            f"{len(eligible)} eligible for processing, {len(skipped_no_rec)} skipped"
        )

        # 3. Process each eligible call
        processed = []
        skipped = list(skipped_no_rec)

        for call in eligible:
            result = process_call(call, call_type)
            if result.get("skipped"):
                skipped.append(result)
            else:
                processed.append(result)

        return (
            jsonify(
                {
                    "message": f"Processed {len(processed)}, skipped {len(skipped)}",
                    "processed": processed,
                    "skipped": skipped,
                    "metadata": metadata,  # includes NextPageUri for pagination
                }
            ),
            200,
        )

    except req.HTTPError as e:
        logger.error(f"Exotel API error: {e}")
        return jsonify({"error": f"Exotel API error: {str(e)}"}), 502
    except Exception as e:
        logger.error(f"process_exotel_calls failed: {e}")
        return jsonify({"error": str(e)}), 500


# ── Fetch calls list only (no processing) ────
@app.route("/exotel/calls", methods=["GET"])
def list_exotel_calls():
    """
    Returns raw Exotel call list without processing.

    Query params:
      date_from, date_to, page_size, page_uri
    """
    if not EXOTEL_API_KEY or not EXOTEL_API_TOKEN or not EXOTEL_ACCOUNT_SID:
        return jsonify({"error": "Exotel credentials not configured"}), 500

    try:
        data = fetch_exotel_calls(
            date_from=request.args.get("date_from"),
            date_to=request.args.get("date_to"),
            page_size=int(request.args.get("page_size", 50)),
            page_uri=request.args.get("page_uri"),
        )
        # Annotate each call with whether it has a recording
        for c in data.get("Calls", []):
            c["has_recording"] = bool(c.get("RecordingUrl"))
        return jsonify(data), 200

    except req.HTTPError as e:
        return jsonify({"error": f"Exotel API error: {str(e)}"}), 502
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Existing listener ─────────────────────────
@app.route("/listen", methods=["POST"])
def listen():
    data = get_call_analysis()
    return jsonify({"message": "Listening", "data": data}), 200


if __name__ == "__main__":
    app.run(port=5000, debug=True)
