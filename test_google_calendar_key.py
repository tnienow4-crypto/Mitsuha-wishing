import argparse
import datetime
import os
from urllib.parse import quote

import requests
from dotenv import load_dotenv


def _env(name: str) -> str | None:
    val = os.environ.get(name)
    if val is None or val.strip() == "":
        return None
    return val


def get_google_key() -> str:
    key = _env("GOOGLE_API_KEY") or _env("GOOGLE_CALENDAR_API_KEY") or _env("GOOGLE_CALENDER_API_KEY")
    if not key:
        raise SystemExit(
            "Missing Google API key. Set GOOGLE_API_KEY (or GOOGLE_CALENDAR_API_KEY / GOOGLE_CALENDER_API_KEY)."
        )
    return key


def get_google_key_source_name() -> str | None:
    for name in ("GOOGLE_API_KEY", "GOOGLE_CALENDAR_API_KEY", "GOOGLE_CALENDER_API_KEY"):
        if _env(name):
            return name
    return None


def _iso_utc(dt: datetime.datetime) -> str:
    return dt.replace(tzinfo=datetime.timezone.utc).isoformat().replace("+00:00", "Z")


def main() -> int:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Test Google Calendar API key against a public calendar")
    parser.add_argument(
        "--calendar-id",
        default="en.usa.official#holiday@group.v.calendar.google.com",
        help="Public Google Calendar ID to query (default: US official holidays)",
    )
    parser.add_argument(
        "--skip-metadata",
        action="store_true",
        help="Skip the calendars.get metadata call and only test events.list (closer to how the bot runs).",
    )
    parser.add_argument(
        "--date",
        default=None,
        help="Date to query (YYYY-MM-DD). Defaults to today (UTC).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=20,
        help="HTTP timeout seconds (default: 20)",
    )
    args = parser.parse_args()

    key = get_google_key()
    key_source = get_google_key_source_name() or "(unknown env var)"
    calendar_id = args.calendar_id

    if args.date:
        try:
            day = datetime.date.fromisoformat(args.date)
        except Exception:
            raise SystemExit("Invalid --date. Use YYYY-MM-DD")
    else:
        day = datetime.datetime.now(datetime.timezone.utc).date()

    start = datetime.datetime.combine(day, datetime.time.min, tzinfo=datetime.timezone.utc)
    end = start + datetime.timedelta(days=1)

    encoded_id = quote(calendar_id, safe="")

    meta_url = f"https://www.googleapis.com/calendar/v3/calendars/{encoded_id}"
    events_url = f"https://www.googleapis.com/calendar/v3/calendars/{encoded_id}/events"

    def get(url: str, params: dict) -> requests.Response:
        return requests.get(url, params=params, timeout=args.timeout)

    print(f"Testing calendar: {calendar_id}")
    print(f"Query date (UTC): {day.isoformat()}")
    print(f"Key source: {key_source}")

    # 1) Events list (small window)
    ev_params = {
        "key": key,
        "timeMin": _iso_utc(start),
        "timeMax": _iso_utc(end),
        "singleEvents": True,
        "orderBy": "startTime",
        "maxResults": 10,
    }
    ev_resp = get(events_url, ev_params)
    print(f"Events status: {ev_resp.status_code}")
    if ev_resp.status_code != 200:
        try:
            print(ev_resp.json())
        except Exception:
            print(ev_resp.text)
        return 3

    data = ev_resp.json() if ev_resp.headers.get("content-type", "").startswith("application/json") else {}
    items = data.get("items") or []
    print(f"Events returned: {len(items)} (showing up to 3)")
    for item in items[:3]:
        print(f"- {(item.get('summary') or '').strip()} | start={item.get('start')} | end={item.get('end')}")

    # 2) Optional: Calendar metadata
    if not args.skip_metadata:
        meta_resp = get(meta_url, {"key": key})
        print(f"Metadata status: {meta_resp.status_code}")
        if meta_resp.status_code == 200:
            meta = (
                meta_resp.json()
                if meta_resp.headers.get("content-type", "").startswith("application/json")
                else {}
            )
            cal_summary = meta.get("summary")
            cal_tz = meta.get("timeZone")
            print(f"Calendar metadata OK: summary={cal_summary!r}, timeZone={cal_tz!r}")
        else:
            try:
                print(meta_resp.json())
            except Exception:
                print(meta_resp.text)
            print("Note: events.list worked, but calendars.get metadata failed.")

    print("\nResult: API key works for events.list on a public Google Calendar.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
