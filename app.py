# app.py

from flask import Flask, request, jsonify
import os
import subprocess
import tempfile
import json
import logging
from dotenv import load_dotenv
from sarvamai import SarvamAI
from groq import Groq

# Import from functions
from function.database import save_call_analysis

load_dotenv()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize clients
sarvam_client = SarvamAI(api_subscription_key=os.getenv("SARVAM_API_KEY"))
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))


# ─────────────────────────────────────────────
# AUDIO CHUNK & TRANSCRIPTION (Keep as is)
# ─────────────────────────────────────────────


def chunk_audio(file_path: str, chunk_duration: int = 29):
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


def _cleanup(path: str):
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception:
        pass


def _cleanup_dir(path: str):
    try:
        if os.path.exists(path):
            os.rmdir(path)
    except Exception:
        pass


def transcribe_audio(file_path: str, language_code: str = "unknown"):
    chunks, tmp_dir = chunk_audio(file_path)
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


# ─────────────────────────────────────────────
# ANALYSIS PROMPTS & FUNCTIONS (Keep as is)
# ─────────────────────────────────────────────

# [Your STANDARD_SYSTEM_PROMPT and LEAD_GEN_SYSTEM_PROMPT remain exactly the same]

STANDARD_SYSTEM_PROMPT = """You are an expert call center analyst for a pharmaceutical software company.

Analyze the given support/complaint call transcript and return ONLY a valid JSON object.

STRICT RULES:
- Output MUST be valid JSON (no markdown, no explanation, no extra text)
- Do NOT omit any field
- If information is missing, use null
- Follow exact data types strictly
- Use ONLY allowed enum values
- Keep summary concise (2-3 sentences)
- Do NOT hallucinate or assume facts not present

JSON SCHEMA:

{
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

CONTEXT UNDERSTANDING RULES:

- Identify caller type implicitly (customer, pharmacist, staff, delivery person, etc.)

- If caller is delivery staff / employee:
  - Complaints about "no orders", "no delivery assigned", "no work", "no tasks"
    → issue_category = "operations_issue"
    → issue_subcategory = "no orders assigned"

- If medicines/products are actually unavailable for customers
  → use "complaint" with subcategory like "product unavailable"

- Do NOT confuse:
  - "no orders assigned" ❌ with ❌ "product unavailable"
  - "no work" ❌ with ❌ "inventory issue"

- Use "software_error" ONLY for bugs, crashes, incorrect system behavior
- Use "performance_slow" ONLY when system/app is slow
- Use "training_needed" if user doesn’t know how to use system
- Use "feature_request" for new feature demands

VALIDATION RULES:

- sentiment must be one of the allowed values
- emotion must match tone of conversation
- customer_satisfaction should reflect sentiment:
  - negative → 1–4
  - neutral → 4–7
  - positive → 7–10
- follow_up_required = true if:
  - issue unresolved OR
  - callback requested OR
  - action is pending
- keep product_module_mentioned relevant only
- confidence_score should be lower (<0.5) if transcript is unclear or incomplete

Return ONLY JSON."""

LEAD_GEN_SYSTEM_PROMPT = """..."""  # Keep your full LEAD_GEN_SYSTEM_PROMPT here


def analyze_text(text: str, call_type: str = "general") -> dict:
    """Route to correct analyzer based on call_type."""
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
    return "Sarvam + Groq API running"


@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "Empty filename"}), 400

    call_id = request.form.get("call_id", 0)
    customer_id = request.form.get("customer_id", 0)
    call_type = request.form.get("call_type", "general")
    language = request.form.get("language", "unknown")

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    try:
        transcript, detected_lang = transcribe_audio(file_path, language)
        logger.info(f"Transcribed call_id={call_id} lang={detected_lang}")

        analysis = analyze_text(transcript, call_type)
        logger.info(f"Analyzed call_id={call_id} type={call_type}")

        # Save using the new database function
        save_call_analysis(
            call_id, customer_id, call_type, transcript, detected_lang, analysis
        )

        return jsonify(
            {
                "message": "Success",
                "call_type": call_type,
                "transcript": transcript,
                "language": detected_lang,
                "analysis": analysis,
            }
        )

    except Exception as e:
        logger.error(f"Upload failed for call_id={call_id}: {e}")
        return jsonify({"error": str(e)}), 500

    finally:
        _cleanup(file_path)


@app.route("/webhook/exotel", methods=["POST"])
def exotel_webhook():
    data = request.form.to_dict()
    logger.info(f"Exotel webhook received: {data}")

    recording_url = data.get("RecordingUrl")
    call_sid = data.get("CallSid")
    from_number = data.get("From")
    call_type = request.args.get("call_type", "general")

    if not recording_url or not call_sid:
        return jsonify({"error": "Missing RecordingUrl or CallSid"}), 400

    import requests as req

    tmp_path = os.path.join(UPLOAD_FOLDER, f"{call_sid}.mp3")

    try:
        r = req.get(recording_url, timeout=60)
        r.raise_for_status()
        with open(tmp_path, "wb") as f:
            f.write(r.content)

        transcript, detected_lang = transcribe_audio(tmp_path)
        analysis = analyze_text(transcript, call_type)

        save_call_analysis(
            call_sid, from_number, call_type, transcript, detected_lang, analysis
        )

        return (
            jsonify(
                {
                    "message": "Processed",
                    "call_sid": call_sid,
                    "call_type": call_type,
                    "analysis": analysis,
                }
            ),
            200,
        )

    except Exception as e:
        logger.error(f"Webhook failed for {call_sid}: {e}")
        return jsonify({"error": str(e)}), 500

    finally:
        _cleanup(tmp_path)


if __name__ == "__main__":
    app.run(port=5000, debug=True)
