# functions/database.py

import os
import json
import logging
import psycopg2
from psycopg2.extras import Json
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


def get_db_connection():
    host = os.getenv("DB_HOST")
    db = os.getenv("DB_NAME")
    user = os.getenv("DB_USER")
    port = os.getenv("DB_PORT", 5432)

    print("DEBUG DB CONFIG →", host, db, user, port)

    if not host:
        raise Exception("DB_HOST is missing")

    return psycopg2.connect(
        host=host.strip(),
        database=db,
        user=user,
        password=os.getenv("DB_PASSWORD"),
        port=port,
    )


def save_call_analysis(
    call_id,
    customer_id,
    call_type: str,
    transcript: str,
    detected_lang: str,
    analysis: dict,
):
    """
    Update automated_call_schedule with transcript and analysis data.
    - response        : full payload JSON (existing behaviour)
    - transcript      : plain transcript text
    - analysis_details: full analysis JSON
    call_id = id of the automated_call_schedule row.
    """

    logger.info(
        f"🔍 save_call_analysis called | call_id={call_id!r} "
        f"(type={type(call_id).__name__}) | customer_id={customer_id!r} | call_type={call_type!r}"
    )

    # ── Guard: call_id must be a valid non-zero integer ───────────────────────
    try:
        call_id_int = int(call_id)
    except (ValueError, TypeError):
        logger.error(
            f"❌ Invalid call_id={call_id!r} — cannot convert to int. Skipping DB save."
        )
        return

    if call_id_int <= 0:
        logger.error(
            f"❌ call_id={call_id_int} is 0 or negative — no matching row possible. Skipping DB save."
        )
        return

    conn = get_db_connection()
    cur = conn.cursor()

    try:
        # ── Check row exists ──────────────────────────────────────────────────
        cur.execute(
            "SELECT id, is_triggered FROM automated_call_schedule WHERE id = %s",
            (call_id_int,),
        )
        existing = cur.fetchone()
        if existing is None:
            logger.error(
                f"❌ No row found in automated_call_schedule with id={call_id_int}. Nothing to update."
            )
            return
        logger.info(
            f"✅ Row found: id={existing[0]}, current is_triggered={existing[1]}"
        )

        # ── Build response payload (full combined object) ─────────────────────
        overall = analysis.get("overall", analysis)

        response_payload = {
            "transcript": transcript,
            "detected_language": detected_lang,
            "call_type": call_type,
            "analysis": analysis,
            "summary": overall.get("summary"),
            "sentiment": overall.get("sentiment"),
            "emotion": overall.get("emotion"),
            "customer_satisfaction": overall.get("customer_satisfaction"),
            "confidence_score": overall.get("confidence_score"),
            "issue_category": overall.get("issue_category"),
            "issue_subcategory": overall.get("issue_subcategory"),
            "urgency_level": overall.get("urgency_level"),
            "follow_up_required": overall.get("follow_up_required"),
            "follow_up_reason": overall.get("follow_up_reason"),
            "resolution_status": overall.get("resolution_status"),
            "resolution_summary": overall.get("resolution_summary"),
            "key_customer_concern": overall.get("key_customer_concern"),
            "product_module_mentioned": overall.get("product_module_mentioned", []),
            "callback_requested_by_customer": overall.get(
                "callback_requested_by_customer"
            ),
            "suggested_callback_time": overall.get("suggested_callback_time"),
            "speaker_1": analysis.get("speaker_1"),
            "speaker_2": analysis.get("speaker_2"),
        }

        # ── UPDATE all three columns ──────────────────────────────────────────
        cur.execute(
            """
            UPDATE automated_call_schedule
            SET
                response           = %s,
                transcript         = %s,
                analysis_details   = %s,
                is_triggered       = 'yes',
                triggered_datetime = CURRENT_TIMESTAMP,
                updated_date       = CURRENT_TIMESTAMP
            WHERE id = %s
            """,
            (
                Json(response_payload),  # response      (json)
                transcript,  # transcript    (text)
                Json(analysis),  # analysis_details (json)
                call_id_int,
            ),
        )

        logger.info(f"🔍 UPDATE rowcount={cur.rowcount}")

        if cur.rowcount == 0:
            logger.warning(f"⚠️  UPDATE affected 0 rows for id={call_id_int}.")
        else:
            logger.info(
                f"✅ Analysis saved | call_id={call_id_int} | type={call_type} "
                f"| sentiment={overall.get('sentiment')} | rows_updated={cur.rowcount}"
            )

        conn.commit()
        logger.info(f"✅ COMMIT done for call_id={call_id_int}")

    except Exception as e:
        conn.rollback()
        logger.error(f"❌ DB save failed for call_id={call_id_int}: {str(e)}")
        raise
    finally:
        cur.close()
        conn.close()


def get_call_analysis():
    conn = get_db_connection()
    cur = conn.cursor()

    try:
        cur.execute("SELECT * FROM automated_call_schedule")

        rows = cur.fetchall()
        colnames = [desc[0] for desc in cur.description]
        results = [dict(zip(colnames, row)) for row in rows]

        return results

    except Exception as e:
        logger.error(f"❌ DB fetch failed: {str(e)}")
        raise
    finally:
        cur.close()
        conn.close()

def get_existing_analysis(call_id) -> dict | None:
    """
    Returns the saved analysis dict if this call_id already has
    a completed analysis (is_triggered='yes' and analysis_details is not null).
    Returns None if not found or not yet analysed.
    """
    try:
        call_id_int = int(call_id)
    except (ValueError, TypeError):
        return None

    if call_id_int <= 0:
        return None

    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            """
            SELECT transcript, analysis_details, response
            FROM automated_call_schedule
            WHERE id = %s
              AND is_triggered = 'yes'
              AND analysis_details IS NOT NULL
            """,
            (call_id_int,),
        )
        row = cur.fetchone()
        if row is None:
            return None

        transcript, analysis_details, response = row
        return {
            "transcript": transcript,
            "analysis": analysis_details,  # already a dict (psycopg2 parses json)
            "response": response,
            "cached": True,
        }
    except Exception as e:
        logger.error(f"get_existing_analysis failed for call_id={call_id_int}: {e}")
        return None
    finally:
        cur.close()
        conn.close()
