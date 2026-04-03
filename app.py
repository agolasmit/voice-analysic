from flask import Flask, request, jsonify
import os
import subprocess
import tempfile
from sarvamai import SarvamAI
from groq import Groq
import json
from dotenv import load_dotenv
import psycopg2


def get_db_connection():
    return psycopg2.connect(
        host=os.getenv("DB_HOST"),
        database=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        port=os.getenv("DB_PORT", 5432),
    )


load_dotenv()

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

sarvam_client = SarvamAI(api_subscription_key=os.getenv("SARVAM_API_KEY"))

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def save_call_analysis(call_id, customer_id, analysis):
    conn = get_db_connection()
    cur = conn.cursor()

    try:
        cur.execute(
            """
            INSERT INTO call_analysic_details
            (call_id, customer_id, call_summary, emotion, sentiment)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (
                call_id,
                customer_id,
                analysis.get("summary", ""),
                analysis.get("emotion", ""),
                analysis.get("sentiment", ""),
            ),
        )
        conn.commit()

    except Exception as e:
        conn.rollback()
        print("DB Error:", e)
        raise e

    finally:
        cur.close()
        conn.close()


# ---------------- AUDIO CHUNK ----------------
def chunk_audio(file_path, chunk_duration=29):
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


# ---------------- TRANSCRIPTION ----------------
def transcribe_audio(file_path, language_code="unknown"):
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
            if os.path.exists(f):
                os.remove(f)
        if os.path.exists(tmp_dir):
            os.rmdir(tmp_dir)

    return " ".join(full_transcript).strip(), detected_language


# ---------------- ANALYSIS (GROQ) ----------------
def analyze_text(text: str) -> dict:
    system_prompt = """You are a call center analyst. Analyze customer calls and return ONLY a valid JSON object with exactly these fields:
{
  "sentiment": "positive" | "negative" | "neutral",
  "emotion": "happy" | "frustrated" | "angry" | "confused" | "satisfied" | "neutral",
  "customer_satisfaction": 1-10 (integer),
  "confidence_score": 0.0-1.0 (float),
  "summary": "2-3 sentence summary of the call"
}
No extra text. No markdown. Just the JSON object."""

    user_prompt = f"Analyze this customer call transcript:\n\n{text}"

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,  # low = more consistent JSON
        max_completion_tokens=512,
        response_format={"type": "json_object"},  # guarantees valid JSON output
    )

    raw = response.choices[0].message.content

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"error": "Failed to parse response", "raw": raw}


# ---------------- API ----------------
@app.route("/")
def home():
    return "Sarvam + Groq API running"


@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]

    call_id = request.form.get("call_id", 0)
    customer_id = request.form.get("customer_id", 0)

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    try:
        transcript, detected_lang = transcribe_audio(file_path)

        analysis = analyze_text(transcript)

        # ✅ SAVE TO DB
        save_call_analysis(call_id, customer_id, analysis)

        return jsonify({"message": "Success", "analysis": analysis})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        if os.path.exists(file_path):
            os.remove(file_path)


if __name__ == "__main__":
    app.run(port=5000, debug=True)
