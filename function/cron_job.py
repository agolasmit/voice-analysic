import sys
import os
import time  # ✅ FIXED

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app, fetch_exotel_calls, process_call
import logging

logging.basicConfig(level=logging.INFO)


def run_hourly_job():
    with app.app_context():
        print("Running hourly cron job...")

        response = fetch_exotel_calls(page_size=50)
        calls = response.get("Calls", [])

        for call in calls:
            if call.get("RecordingUrl") and int(call.get("Duration", 0)) >= 30:
                result = process_call(call, call_type="general")
                print(result)

                # ⏱ wait 30 seconds before next call
                time.sleep(30)


if __name__ == "__main__":
    run_hourly_job()
