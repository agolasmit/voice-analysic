

from app import chunk_audio


class trnascibe:
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

