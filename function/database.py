# functions/database.py

import os
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
    analysis: dict
):
    """
    Save call analysis to database.
    Matches your current table schema (uses 'summary' column, not 'call_summary').
    """
    conn = get_db_connection()
    cur = conn.cursor()

    try:
        if call_type == "lead_gen":
            meta = analysis.get("call_meta", {}) if isinstance(analysis.get("call_meta"), dict) else {}
            profile = analysis.get("business_profile")
            quality = analysis.get("lead_quality")

            cur.execute(
                """
                INSERT INTO call_analysis_details (
                    call_id, customer_id, call_type, transcript, detected_language,
                    summary, sentiment, emotion, customer_satisfaction, confidence_score,
                    raw_analysis
                ) VALUES (
                    %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s,
                    %s
                )
                ON CONFLICT (call_id) DO UPDATE SET
                    updated_at = CURRENT_TIMESTAMP,
                    summary = EXCLUDED.summary,
                    sentiment = EXCLUDED.sentiment,
                    emotion = EXCLUDED.emotion,
                    customer_satisfaction = EXCLUDED.customer_satisfaction,
                    raw_analysis = EXCLUDED.raw_analysis;
                """,
                (
                    str(call_id),
                    str(customer_id),
                    call_type,
                    transcript,
                    detected_lang,
                    meta.get("summary"),
                    meta.get("sentiment"),
                    meta.get("emotion"),
                    meta.get("customer_satisfaction"),
                    meta.get("confidence_score"),
                    Json(analysis),                    # Store full analysis
                ),
            )

        else:
            # Support / General calls - using your current table columns
            cur.execute(
                """
                INSERT INTO call_analysis_details (
                    call_id, customer_id, call_type, transcript, detected_language,
                    summary, sentiment, emotion, customer_satisfaction, confidence_score,
                    issue_category, issue_subcategory, urgency_level,
                    follow_up_required, follow_up_reason, suggested_callback_time,
                    callback_requested_by_customer, resolution_status,
                    resolution_summary, key_customer_concern, product_modules,
                    raw_analysis
                ) VALUES (
                    %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s,
                    %s, %s, %s,
                    %s, %s, %s,
                    %s, %s,
                    %s, %s, %s,
                    %s
                )
                ON CONFLICT (call_id) DO UPDATE SET
                    updated_at = CURRENT_TIMESTAMP,
                    summary = EXCLUDED.summary,
                    sentiment = EXCLUDED.sentiment,
                    emotion = EXCLUDED.emotion,
                    customer_satisfaction = EXCLUDED.customer_satisfaction,
                    issue_category = EXCLUDED.issue_category,
                    urgency_level = EXCLUDED.urgency_level,
                    follow_up_required = EXCLUDED.follow_up_required,
                    resolution_status = EXCLUDED.resolution_status,
                    raw_analysis = EXCLUDED.raw_analysis;
                """,
                (
                    str(call_id),
                    str(customer_id),
                    call_type,
                    transcript,
                    detected_lang,
                    analysis.get("summary"),
                    analysis.get("sentiment"),
                    analysis.get("emotion"),
                    analysis.get("customer_satisfaction"),
                    analysis.get("confidence_score"),
                    analysis.get("issue_category"),
                    analysis.get("issue_subcategory"),
                    analysis.get("urgency_level"),
                    analysis.get("follow_up_required"),
                    analysis.get("follow_up_reason"),
                    analysis.get("suggested_callback_time"),
                    analysis.get("callback_requested_by_customer"),
                    analysis.get("resolution_status"),
                    analysis.get("resolution_summary"),
                    analysis.get("key_customer_concern"),
                    Json(analysis.get("product_module_mentioned", [])),
                    Json(analysis),                    # Full raw analysis for future use
                ),
            )

        conn.commit()
        logger.info(f"✅ Analysis saved | call_id={call_id} | type={call_type} | summary={bool(analysis.get('summary'))}")

    except Exception as e:
        conn.rollback()
        logger.error(f"❌ DB save failed for call_id={call_id}: {str(e)}")
        raise
    finally:
        cur.close()
        conn.close()
