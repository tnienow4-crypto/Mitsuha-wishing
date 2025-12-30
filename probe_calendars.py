import datetime
import os
import urllib.parse

import requests
from dotenv import load_dotenv

load_dotenv()

key = (
    os.environ.get("GOOGLE_API_KEY")
    or os.environ.get("GOOGLE_CALENDAR_API_KEY")
    or os.environ.get("GOOGLE_CALENDER_API_KEY")
)
print("Key present:", bool(key))

candidates = [
    # Likely patterns for Google public holiday calendars
    "en.za#holiday@group.v.calendar.google.com",
    "en.southafrican#holiday@group.v.calendar.google.com",
    "en.southafrica#holiday@group.v.calendar.google.com",
    "en.nigerian#holiday@group.v.calendar.google.com",
    "en.egyptian#holiday@group.v.calendar.google.com",
    "en.kenyan#holiday@group.v.calendar.google.com",
]

# Small window; we only care whether the calendar ID exists (non-404).
date = datetime.date.today()
time_min = f"{date.isoformat()}T00:00:00Z"
time_max = f"{date.isoformat()}T23:59:59Z"

for cid in candidates:
    encoded = urllib.parse.quote(cid, safe="")
    url = f"https://www.googleapis.com/calendar/v3/calendars/{encoded}/events"
    try:
        r = requests.get(
            url,
            params={
                "key": key,
                "timeMin": time_min,
                "timeMax": time_max,
                "singleEvents": True,
            },
            timeout=20,
        )
        print(cid, r.status_code)
    except Exception as exc:
        print(cid, "ERROR", exc)
