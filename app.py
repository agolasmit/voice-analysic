from flask import Flask, request, jsonify
import os
import subprocess
import tempfile
import json
import logging
from sarvamai import SarvamAI
from groq import Groq
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import Json

load_dotenv()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

sarvam_client = SarvamAI(api_subscription_key=os.getenv("SARVAM_API_KEY"))
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))


# ─────────────────────────────────────────────
# DATABASE
# ─────────────────────────────────────────────


def get_db_connection():
    return psycopg2.connect(
        host=os.getenv("DB_HOST"),
        database=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        port=os.getenv("DB_PORT", 5432),
    )


def save_call_analysis(
    call_id, customer_id, call_type, transcript, detected_lang, analysis
):
    """
    Save analysis to DB. Routes to correct columns based on call_type.
    call_type: "general" | "lead_gen"
    """
    conn = get_db_connection()
    cur = conn.cursor()

    try:
        if call_type == "lead_gen":
            meta = analysis.get("call_meta", {})
            profile = analysis.get("business_profile")
            quality = analysis.get("lead_quality")

            cur.execute(
                """
                INSERT INTO call_analysis_details (
                    call_id, customer_id, call_type,
                    transcript, detected_language,
                    call_summary, emotion, sentiment,
                    customer_satisfaction, confidence_score,
                    lead_business_profile, lead_quality,
                    data_completeness, missing_fields
                ) VALUES (
                    %s, %s, %s,
                    %s, %s,
                    %s, %s, %s,
                    %s, %s,
                    %s, %s,
                    %s, %s
                )
                """,
                (
                    call_id,
                    customer_id,
                    call_type,
                    transcript,
                    detected_lang,
                    meta.get("summary"),
                    meta.get("emotion"),
                    meta.get("sentiment"),
                    meta.get("customer_satisfaction"),
                    meta.get("confidence_score"),
                    Json(profile),
                    Json(quality),
                    meta.get("data_completeness"),
                    meta.get("missing_fields", []),
                ),
            )

        else:
            cur.execute(
                """
                INSERT INTO call_analysis_details (
                    call_id, customer_id, call_type,
                    transcript, detected_language,
                    call_summary, emotion, sentiment,
                    customer_satisfaction, confidence_score
                ) VALUES (
                    %s, %s, %s,
                    %s, %s,
                    %s, %s, %s,
                    %s, %s
                )
                """,
                (
                    call_id,
                    customer_id,
                    call_type,
                    transcript,
                    detected_lang,
                    analysis.get("summary"),
                    analysis.get("emotion"),
                    analysis.get("sentiment"),
                    analysis.get("customer_satisfaction"),
                    analysis.get("confidence_score"),
                ),
            )

        conn.commit()
        logger.info(f"Saved analysis for call_id={call_id} type={call_type}")

    except Exception as e:
        conn.rollback()
        logger.error(f"DB save failed for call_id={call_id}: {e}")
        raise

    finally:
        cur.close()
        conn.close()


# ─────────────────────────────────────────────
# AUDIO — CHUNK
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


# ─────────────────────────────────────────────
# TRANSCRIPTION — SARVAM
# ─────────────────────────────────────────────


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
# ANALYSIS PROMPTS
# ─────────────────────────────────────────────

STANDARD_SYSTEM_PROMPT = """You are a call center analyst. Analyze the transcript and return ONLY a valid JSON object with exactly these fields:
{
  "sentiment": "positive" | "negative" | "neutral",
  "emotion": "happy" | "frustrated" | "angry" | "confused" | "satisfied" | "neutral",
  "customer_satisfaction": <integer 1-10>,
  "confidence_score": <float 0.0-1.0>,
  "summary": "<2-3 sentence summary of the call>"
}
No markdown. No extra text. JSON only."""

LEAD_GEN_SYSTEM_PROMPT = """You are an expert at extracting structured business information from Indian pharmaceutical sales calls.

The call transcript is in Hinglish (a mix of Hindi and English). The conversation is unstructured — the agent and chemist talk naturally, not in a fixed Q&A format. Your job is to carefully read the entire transcript and extract every piece of business information about the chemist/pharmacy.

EXTRACTION RULES:
- Extract ONLY what is explicitly said or clearly implied in the transcript
- If something is NOT mentioned, set it to null — never guess or assume
- For numbers: extract the actual number if mentioned (e.g. "teen stores" = 3, "do branches" = 2, "ek hi shop" = 1)
- For vague answers: capture the vague text as-is in a "_raw" field alongside null for the structured field
- For locations: extract city, area/locality if mentioned
- For yes/no fields: true if confirmed, false if denied, null if not discussed
- Inventory value may be mentioned in lakhs (e.g. "50 lakh ka maal" = 5000000)

Return ONLY this JSON structure, no markdown, no extra text:

{
  "business_profile": {
    "store_count": <integer or null>,
    "store_count_raw": "<exact words used or null>",
    "locations": [
      {
        "city": "<city name or null>",
        "area": "<area/locality or null>",
        "is_main_store": <true/false/null>
      }
    ],
    "branch_count": <integer or null>,
    "branch_count_raw": "<exact words used or null>",
    "staff_count": <integer or null>,
    "staff_count_raw": "<exact words used or null>",
    "is_doing_online_orders": <true/false/null>,
    "online_platforms": ["<platform name>"],
    "counter_sales_per_day": <integer or null>,
    "counter_sales_raw": "<exact words used or null>",
    "inventory_quantity": <integer or null>,
    "inventory_quantity_raw": "<exact words used or null>",
    "inventory_value_inr": <integer or null>,
    "inventory_value_raw": "<exact words used or null>",
    "years_in_business": <integer or null>,
    "years_raw": "<exact words used or null>"
  },
  "lead_quality": {
    "interest_level": "high" | "medium" | "low" | "not_interested",
    "interest_reason": "<why you assessed this level, 1 sentence>",
    "callback_requested": <true/false/null>,
    "best_time_to_call": "<mentioned time/day or null>"
  },
  "call_meta": {
    "sentiment": "positive" | "negative" | "neutral",
    "emotion": "happy" | "frustrated" | "angry" | "confused" | "satisfied" | "neutral",
    "customer_satisfaction": <integer 1-10>,
    "confidence_score": <float 0.0-1.0>,
    "summary": "<2-3 sentence summary of what was discussed>",
    "data_completeness": <float 0.0-1.0>,
    "missing_fields": ["<list of important fields not mentioned in the call>"]
  }
}"""


# ─────────────────────────────────────────────
# ANALYSIS — GROQ
# ─────────────────────────────────────────────


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
    call_type = request.form.get("call_type", "general")  # "general" | "lead_gen"
    language = request.form.get("language", "unknown")

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    try:
        # 1. Transcribe
        transcript, detected_lang = transcribe_audio(file_path, language)
        logger.info(f"Transcribed call_id={call_id} lang={detected_lang}")

        # 2. Analyze
        analysis = analyze_text(transcript, call_type)
        logger.info(f"Analyzed call_id={call_id} type={call_type}")

        # 3. Save to DB
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
    """Exotel calls this automatically when a call ends."""
    data = request.form.to_dict()
    logger.info(f"Exotel webhook: {data}")

    recording_url = data.get("RecordingUrl")
    call_sid = data.get("CallSid")
    from_number = data.get("From")
    duration = data.get("Duration", 0)

    # Pass call_type as query param in your Exotel passthru URL:
    # https://yourdomain.com/webhook/exotel?call_type=lead_gen
    call_type = request.args.get("call_type", "general")

    if not recording_url or not call_sid:
        return jsonify({"error": "Missing RecordingUrl or CallSid"}), 400

    import requests as req
    import uuid

    tmp_path = os.path.join(UPLOAD_FOLDER, f"{call_sid}.mp3")

    try:
        # Download recording from Exotel
        r = req.get(recording_url, timeout=60)
        r.raise_for_status()
        with open(tmp_path, "wb") as f:
            f.write(r.content)

        # Transcribe + analyze
        transcript, detected_lang = transcribe_audio(tmp_path)
        analysis = analyze_text(transcript, call_type)

        # Save — use CallSid as call_id, from_number as customer_id
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
        logger.error(f"Webhook processing failed for {call_sid}: {e}")
        return jsonify({"error": str(e)}), 500

    finally:
        _cleanup(tmp_path)


if __name__ == "__main__":
    app.run(port=5000, debug=True)
