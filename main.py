import asyncio
import argparse
import datetime
import json
import os
from dataclasses import dataclass
from typing import Iterable, List, Optional, Set, Tuple
from urllib.parse import quote

import discord
import requests
from dotenv import load_dotenv
from ollama import Client

try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover
    ZoneInfo = None  # type: ignore


IST_FIXED_TZ = datetime.timezone(datetime.timedelta(hours=5, minutes=30))

load_dotenv()

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
DM_DISABLED_PATH = os.path.join(DATA_DIR, "dm_disabled.json")


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    return value


def _env_int(*names: str) -> int:
    for name in names:
        value = _env(name)
        if value is not None:
            return int(value)
    raise ValueError(f"Missing required int env var (tried: {', '.join(names)})")


def _split_csv(value: Optional[str]) -> List[str]:
    if not value:
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


@dataclass(frozen=True)
class Config:
    discord_token: str
    guild_id: int
    fallback_channel_id: int
    sticker_id: Optional[int]
    sticker_prefix: str
    sticker_pick_mode: str
    google_api_key: str
    google_calendar_ids: Tuple[str, ...]
    ollama_api_key: Optional[str]
    ollama_host: str
    ollama_model: str


def load_config() -> Config:
    discord_token = _env("DISCORD_TOKEN") or _env("DISCORD_BOT_API_KEY")
    if not discord_token:
        raise ValueError("DISCORD_TOKEN is required")

    guild_id = _env_int("DISCORD_GUILD_ID", "GUILD_ID")
    fallback_channel_id = _env_int("DISCORD_FALLBACK_CHANNEL_ID", "CHANNEL_ID")

    sticker_id: Optional[int] = None
    sticker_raw = _env("DISCORD_STICKER_ID")
    if sticker_raw:
        try:
            sticker_id = int(sticker_raw)
        except Exception:
            raise ValueError("DISCORD_STICKER_ID must be an integer sticker ID")

    sticker_prefix = _env("DISCORD_STICKER_PREFIX", "CSD") or "CSD"
    sticker_pick_mode = (_env("DISCORD_STICKER_PICK_MODE", "daily") or "daily").strip().lower()
    if sticker_pick_mode not in {"daily", "ai"}:
        raise ValueError("DISCORD_STICKER_PICK_MODE must be 'daily' or 'ai'")

    google_api_key = (
        _env("GOOGLE_API_KEY")
        or _env("GOOGLE_CALENDAR_API_KEY")
        or _env("GOOGLE_CALENDER_API_KEY")
    )
    if not google_api_key:
        raise ValueError("GOOGLE_API_KEY is required")

    calendar_ids = _split_csv(_env("GOOGLE_CALENDAR_IDS"))
    if not calendar_ids:
        raise ValueError(
            "GOOGLE_CALENDAR_IDS is required (comma-separated Google Calendar IDs for public holiday calendars)"
        )

    ollama_host = _env("OLLAMA_HOST", "https://ollama.com")
    ollama_model = _env("OLLAMA_MODEL", "gpt-oss:120b") or "gpt-oss:120b"

    return Config(
        discord_token=discord_token,
        guild_id=guild_id,
        fallback_channel_id=fallback_channel_id,
        sticker_id=sticker_id,
        sticker_prefix=sticker_prefix,
        sticker_pick_mode=sticker_pick_mode,
        google_api_key=google_api_key,
        google_calendar_ids=tuple(calendar_ids),
        ollama_api_key=_env("OLLAMA_API_KEY") or _env("OLLAMA_CLOUD_API_KEY"),
        ollama_host=ollama_host,
        ollama_model=ollama_model,
    )


def _stable_daily_index(date_ist: datetime.date, count: int) -> int:
    # Stable for a given date; avoids sending a different sticker each rerun.
    return (hash(date_ist.isoformat()) & 0x7FFFFFFF) % max(1, count)


async def resolve_sticker(
    client_: discord.Client,
    guild: discord.Guild,
    *,
    sticker_id: Optional[int],
    sticker_prefix: str,
    date_ist: datetime.date,
) -> Optional[discord.Sticker]:
    # If an explicit sticker id is configured, use it.
    if sticker_id:
        try:
            return await client_.fetch_sticker(sticker_id)
        except Exception as exc:
            print(f"Could not fetch sticker {sticker_id}: {exc}")
            return None

    prefix = (sticker_prefix or "").strip()
    if not prefix:
        return None

    try:
        stickers = await guild.fetch_stickers()
    except Exception as exc:
        print(f"Could not fetch guild stickers: {exc}")
        return None

    matches = [s for s in stickers if (getattr(s, "name", "") or "").lower().startswith(prefix.lower())]
    if not matches:
        print(f"No guild stickers found with prefix '{prefix}'.")
        return None

    return matches[_stable_daily_index(date_ist, len(matches))]


def _normalize_sticker_name(name: str) -> str:
    return "".join(ch for ch in name.strip().lower() if ch.isalnum() or ch in {"_", "-"})


def _suffix_after_prefix(name: str, prefix: str) -> str:
    if not name.lower().startswith(prefix.lower()):
        return name
    suffix = name[len(prefix):]
    return suffix.lstrip(" _-:")


def pick_sticker_by_ai(
    config: Config,
    *,
    stickers: List[discord.Sticker],
    prefix: str,
    date_ist: datetime.date,
    special_days: List[str],
) -> Optional[discord.Sticker]:
    """Ask Ollama to pick the best sticker (by name) for today's special day(s)."""
    # Keep prompt bounded.
    max_candidates = 100
    candidates = stickers[:max_candidates]

    # Present both full name and suffix so the model has semantically useful tokens.
    lines = []
    for s in candidates:
        name = getattr(s, "name", None) or ""
        if not name:
            continue
        lines.append(f"- {name}  (suffix: '{_suffix_after_prefix(name, prefix)}')")

    if not lines:
        return None

    joined_days = ", ".join(special_days)
    date_str = datetime.datetime.combine(date_ist, datetime.time.min).strftime("%d %b %Y")

    prompt = (
        "You are selecting ONE Discord sticker to attach to a server greeting.\n"
        "Pick the sticker whose NAME best matches the vibe for today's holiday/special day(s).\n"
        "Rules:\n"
        "- Output ONLY the sticker name (exactly as listed), nothing else.\n"
        "- If none fit, output 'NONE'.\n\n"
        f"Date (IST): {date_str}\n"
        f"Holiday/special day(s): {joined_days}\n\n"
        "Sticker candidates:\n"
        + "\n".join(lines)
    )

    try:
        ollama_client = Client(
            host=config.ollama_host,
            headers={"Authorization": "Bearer " + config.ollama_api_key} if config.ollama_api_key else None,
        )
        response = ollama_client.chat(
            model=config.ollama_model,
            messages=[{"role": "user", "content": prompt}],
        )
        text = ((response or {}).get("message") or {}).get("content")
        if not isinstance(text, str):
            return None
        choice = text.strip().strip('"').strip("'")
        if not choice:
            return None
        if choice.upper() == "NONE":
            return None

        choice_norm = _normalize_sticker_name(choice)
        for s in candidates:
            name = getattr(s, "name", None) or ""
            if _normalize_sticker_name(name) == choice_norm:
                return s
        # Also allow returning just the suffix.
        for s in candidates:
            name = getattr(s, "name", None) or ""
            if _normalize_sticker_name(_suffix_after_prefix(name, prefix)) == choice_norm:
                return s
    except Exception as exc:
        print(f"AI sticker selection failed: {exc}")

    return None

intents = discord.Intents.default()
intents.members = True
client = discord.Client(intents=intents)


def ist_now() -> datetime.datetime:
    # Prefer zoneinfo when available, but gracefully fall back on Windows
    # environments that don't ship tzdata.
    if ZoneInfo is not None:
        try:
            return datetime.datetime.now(tz=ZoneInfo("Asia/Kolkata"))
        except Exception:
            pass
    return datetime.datetime.now(tz=IST_FIXED_TZ)


def ist_day_bounds_utc(date_ist: datetime.date) -> Tuple[str, str]:
    tzinfo: datetime.tzinfo = IST_FIXED_TZ
    if ZoneInfo is not None:
        try:
            tzinfo = ZoneInfo("Asia/Kolkata")
        except Exception:
            tzinfo = IST_FIXED_TZ

    start_ist = datetime.datetime.combine(date_ist, datetime.time.min).replace(tzinfo=tzinfo)
    end_ist = datetime.datetime.combine(date_ist, datetime.time.max).replace(tzinfo=tzinfo)

    start_utc = start_ist.astimezone(datetime.timezone.utc)
    end_utc = end_ist.astimezone(datetime.timezone.utc)
    return start_utc.isoformat().replace("+00:00", "Z"), end_utc.isoformat().replace("+00:00", "Z")


def _parse_rfc3339_datetime(value: str) -> Optional[datetime.datetime]:
    try:
        # Python's fromisoformat doesn't accept 'Z'.
        value = value.replace("Z", "+00:00")
        dt = datetime.datetime.fromisoformat(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=datetime.timezone.utc)
        return dt
    except Exception:
        return None


def _event_occurs_on_ist_date(item: dict, date_ist: datetime.date) -> bool:
    start = item.get("start") or {}
    if isinstance(start, dict):
        date_only = start.get("date")
        if isinstance(date_only, str) and date_only:
            return date_only == date_ist.isoformat()

        date_time = start.get("dateTime")
        if isinstance(date_time, str) and date_time:
            dt = _parse_rfc3339_datetime(date_time)
            if dt is None:
                return False
            return dt.astimezone(IST_FIXED_TZ).date() == date_ist
    return False


def load_dm_disabled() -> Set[int]:
    try:
        if not os.path.exists(DM_DISABLED_PATH):
            return set()
        with open(DM_DISABLED_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        user_ids = data.get("user_ids", [])
        return {int(x) for x in user_ids}
    except Exception:
        return set()


def save_dm_disabled(user_ids: Set[int]) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    payload = {
        "updated_at_utc": datetime.datetime.utcnow().replace(tzinfo=datetime.timezone.utc).isoformat(),
        "user_ids": sorted(user_ids),
    }
    with open(DM_DISABLED_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

def fetch_special_days_for_ist_date(config: Config, date_ist: datetime.date) -> List[str]:
    """Fetches holiday/special day names for the provided IST date from one or more public Google Calendars."""
    # Query a slightly wider window to avoid edge cases with calendar timezone boundaries,
    # then filter by actual occurrence on the IST date.
    wide_start_utc = datetime.datetime.combine(date_ist - datetime.timedelta(days=1), datetime.time.min).replace(
        tzinfo=datetime.timezone.utc
    )
    wide_end_utc = datetime.datetime.combine(date_ist + datetime.timedelta(days=1), datetime.time.max).replace(
        tzinfo=datetime.timezone.utc
    )
    time_min = wide_start_utc.isoformat().replace("+00:00", "Z")
    time_max = wide_end_utc.isoformat().replace("+00:00", "Z")

    found: List[str] = []
    seen: Set[str] = set()

    for calendar_id in config.google_calendar_ids:
        encoded_calendar_id = quote(calendar_id, safe="")
        url = f"https://www.googleapis.com/calendar/v3/calendars/{encoded_calendar_id}/events"
        params = {
            "key": config.google_api_key,
            "timeMin": time_min,
            "timeMax": time_max,
            "singleEvents": True,
            "orderBy": "startTime",
        }
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
        except Exception as exc:
            print(f"Error fetching calendar '{calendar_id}': {exc}")
            continue

        for item in data.get("items", []) or []:
            if not _event_occurs_on_ist_date(item, date_ist):
                continue
            summary = (item.get("summary") or "").strip()
            if not summary:
                continue
            key = summary.casefold()
            if key in seen:
                continue
            seen.add(key)
            found.append(summary)

    return found


def fetch_today_special_days(config: Config) -> List[str]:
    return fetch_special_days_for_ist_date(config, ist_now().date())

def generate_wish(config: Config, special_days: List[str], *, date_ist: Optional[datetime.date] = None) -> str:
    """Generates one universal Mitsuha-style wish body for DMs using Ollama."""
    joined = ", ".join(special_days)
    date_ist = date_ist or ist_now().date()
    date_str = datetime.datetime.combine(date_ist, datetime.time.min).strftime("%d %b %Y")

    prompt = (
        "You are Mitsuha — a cute little school girl from CSD (Chaos Show Down). "
        "You want to make friends with everyone in the Discord server. "
        "Write ONE heartwarming, elaborate, human-sounding DM wish that can be sent to any member. "
        "Make it feel like you're talking directly to one person, warmly and a bit shy, like you're trying to make a new friend. "
        "Style: short poem / lyrical message (not rhymes required), with cute decorative lines. "
        "No hashtags. No @everyone. Use more emojis and decorations, but keep it tasteful (about 5–10 emojis total). "
        "Prefer festival/holiday-relevant emojis (based on the holiday names). "
        "End with a tiny signature like '— Mitsuha (CSD)'.\n\n"
        f"Date (IST): {date_str}\n"
        f"Today's globally relevant holiday/special day(s): {joined}\n\n"
        "Message requirements:\n"
        "- Mention the holiday name(s) naturally\n"
        "- Make the reader feel seen and special (but keep it wholesome)\n"
        "- Invite them to say hi / be friends\n"
        "- 8–14 short lines (not one huge paragraph)\n"
        "- Include 1–2 decorative separators like '⋆｡°✩' or '╰(*´︶`*)╯' (ASCII/Unicode ok)\n"
        "- Keep it under 1800 characters\n"
    )

    try:
        ollama_client = Client(
            host=config.ollama_host,
            headers={"Authorization": "Bearer " + config.ollama_api_key} if config.ollama_api_key else None,
        )
        response = ollama_client.chat(
            model=config.ollama_model,
            messages=[{"role": "user", "content": prompt}],
        )
        text = ((response or {}).get("message") or {}).get("content")
        if isinstance(text, str) and text.strip():
            return text.strip()
    except Exception as exc:
        print(f"Error generating wish via Ollama: {exc}")

    return f"Happy {joined}! Hope your day feels a little brighter. — Mitsuha (CSD)"


def generate_channel_wish(
    config: Config,
    special_days: List[str],
    *,
    date_ist: Optional[datetime.date] = None,
) -> str:
    """Generates a channel announcement style wish that includes @everyone."""
    joined = ", ".join(special_days)
    date_ist = date_ist or ist_now().date()
    date_str = datetime.datetime.combine(date_ist, datetime.time.min).strftime("%d %b %Y")

    prompt = (
        "You are Mitsuha — a cute little school girl from CSD (Chaos Show Down). "
        "Write ONE channel announcement message for the whole Discord server audience. "
        "It must start with '@everyone' on the first line. "
        "Make it warm, inclusive, and make everyone feel special together. "
        "No hashtags. Keep emojis to 0–3 max. "
        "End with a tiny signature like '— Mitsuha (CSD)'.\n\n"
        f"Date (IST): {date_str}\n"
        f"Today's globally relevant holiday/special day(s): {joined}\n\n"
        "Message requirements:\n"
        "- Mention the holiday name(s) naturally\n"
        "- Speak to the whole community (plural: everyone / you all / friends)\n"
        "- 4–8 short lines (not one huge paragraph)\n"
        "- Keep it under 1200 characters\n"
    )

    try:
        ollama_client = Client(
            host=config.ollama_host,
            headers={"Authorization": "Bearer " + config.ollama_api_key} if config.ollama_api_key else None,
        )
        response = ollama_client.chat(
            model=config.ollama_model,
            messages=[{"role": "user", "content": prompt}],
        )
        text = ((response or {}).get("message") or {}).get("content")
        if isinstance(text, str) and text.strip():
            out = text.strip()
            if not out.splitlines()[0].strip().startswith("@everyone"):
                out = "@everyone\n" + out
            return out
    except Exception as exc:
        print(f"Error generating channel wish via Ollama: {exc}")

    return f"@everyone\nHappy {joined}! Wishing you all a bright day together. — Mitsuha (CSD)"


def personalize_dm_message(wish_body: str, display_name: str) -> str:
    name = (display_name or "").strip() or "there"
    # Same wish for everyone; only the name changes.
    return (
        f"Hey {name} ✨\n"
        f"⋆｡°✩⋆｡°✩⋆｡°✩\n\n"
        f"{wish_body}\n\n"
        f"⋆｡°✩⋆｡°✩⋆｡°✩\n"
        f"P.S. {name}… if you want, say hi back — I'd really love to be friends 🌸"
    )


async def get_guild(client_: discord.Client, guild_id: int) -> Optional[discord.Guild]:
    guild = client_.get_guild(guild_id)
    if guild is not None:
        return guild
    for g in client_.guilds:
        if g.id == guild_id:
            return g
    return None


async def iter_human_members(guild: discord.Guild) -> Iterable[discord.Member]:
    # Prefer HTTP fetch to avoid relying on member cache.
    try:
        async for member in guild.fetch_members(limit=None):
            if not member.bot:
                yield member
        return
    except Exception as exc:
        print(f"fetch_members failed (falling back to cached members): {exc}")

    for member in guild.members:
        if not member.bot:
            yield member


@client.event
async def on_ready():
    print(f'Logged in as {client.user}')
    try:
        config = load_config()

        date_ist = ist_now().date()

        special_days = fetch_today_special_days(config)
        if not special_days:
            print("No globally relevant holiday/special day found today (based on configured calendars).")
            return

        print(f"Found special day(s): {special_days}")
        dm_wish_body = generate_wish(config, special_days, date_ist=date_ist)
        channel_wish = generate_channel_wish(config, special_days, date_ist=date_ist)
        print(f"Generated DM wish length: {len(dm_wish_body)}")
        print(f"Generated channel wish length: {len(channel_wish)}")

        guild = await get_guild(client, config.guild_id)
        if guild is None:
            print("Guild not found. Check DISCORD_GUILD_ID/GUILD_ID.")
            return

        sticker: Optional[discord.Sticker] = None
        if config.sticker_id:
            sticker = await resolve_sticker(
                client,
                guild,
                sticker_id=config.sticker_id,
                sticker_prefix=config.sticker_prefix,
                date_ist=date_ist,
            )
        else:
            # Fetch guild stickers once.
            try:
                all_stickers = await guild.fetch_stickers()
            except Exception as exc:
                print(f"Could not fetch guild stickers: {exc}")
                all_stickers = []

            prefix = (config.sticker_prefix or "").strip()
            candidates = [
                s
                for s in all_stickers
                if prefix and (getattr(s, "name", "") or "").lower().startswith(prefix.lower())
            ]

            if config.sticker_pick_mode == "ai" and candidates:
                sticker = pick_sticker_by_ai(
                    config,
                    stickers=candidates,
                    prefix=prefix,
                    date_ist=date_ist,
                    special_days=special_days,
                )

            if sticker is None and candidates:
                sticker = candidates[_stable_daily_index(date_ist, len(candidates))]

        # Always post the wish in the configured channel (stickers are allowed in-channel).
        try:
            channel = await client.fetch_channel(config.fallback_channel_id)
            allowed = discord.AllowedMentions(everyone=True, users=False, roles=False)
            if sticker is not None:
                try:
                    await channel.send(channel_wish, stickers=[sticker], allowed_mentions=allowed)
                except discord.HTTPException as exc:
                    print(f"Sticker send failed in channel: {exc}")
                    await channel.send(channel_wish, allowed_mentions=allowed)
            else:
                await channel.send(channel_wish, allowed_mentions=allowed)
        except Exception as exc:
            print(f"Failed to post wish in channel {config.fallback_channel_id}: {exc}")

        dm_disabled_known = load_dm_disabled()
        dm_disabled_updated = set(dm_disabled_known)
        newly_dm_disabled_to_mention: List[str] = []

        async for member in iter_human_members(guild):
            try:
                # DMs are always text-only. (Guild stickers are not permitted in DMs.)
                dm_text = personalize_dm_message(dm_wish_body, member.display_name)
                await member.send(dm_text)
                if member.id in dm_disabled_updated:
                    dm_disabled_updated.discard(member.id)
                await asyncio.sleep(0.6)
            except discord.Forbidden:
                # DM is disabled or blocked.
                if member.id not in dm_disabled_known:
                    newly_dm_disabled_to_mention.append(member.mention)
                dm_disabled_updated.add(member.id)
                await asyncio.sleep(0.2)
            except Exception as exc:
                print(f"Error DMing {member.id}: {exc}")
                await asyncio.sleep(0.2)

        save_dm_disabled(dm_disabled_updated)

        if newly_dm_disabled_to_mention:
            mentions_str = " ".join(newly_dm_disabled_to_mention)
            base = (
                "I tried to DM you today's wish but couldn't. "
                "Please enable DMs (or add me as a friend) so I can DM you next time:\n\n"
            )
            full_message = f"{base}{mentions_str}"

            try:
                channel = await client.fetch_channel(config.fallback_channel_id)
            except Exception:
                print("Fallback channel not found. Check DISCORD_FALLBACK_CHANNEL_ID/CHANNEL_ID.")
                return

            if len(full_message) > 2000:
                # Keep it to one message to avoid channel mess: mention as many as fit, then summarize.
                remaining = 2000 - len(base)
                clipped_mentions = mentions_str[: max(0, remaining - 40)]
                omitted_count = max(0, len(newly_dm_disabled_to_mention) - clipped_mentions.count("<@"))
                if omitted_count > 0:
                    clipped_mentions = clipped_mentions.rstrip() + f"\n(+{omitted_count} more)"
                await channel.send(base + clipped_mentions)
            else:
                await channel.send(full_message)

    finally:
        await client.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Daily CSD Mitsuha wish bot")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not connect to Discord; only fetch special days and generate the wish.",
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Override IST date for --dry-run in YYYY-MM-DD format (example: 2026-01-01).",
    )
    args = parser.parse_args()

    try:
        cfg = load_config()
    except Exception as exc:
        print(f"Configuration error: {exc}")
        raise SystemExit(2)

    if args.dry_run:
        date_ist: Optional[datetime.date] = None
        if args.date:
            try:
                date_ist = datetime.date.fromisoformat(args.date)
            except Exception:
                print("Invalid --date. Use YYYY-MM-DD, e.g. 2026-01-01")
                raise SystemExit(2)

        special_days = (
            fetch_special_days_for_ist_date(cfg, date_ist)
            if date_ist is not None
            else fetch_today_special_days(cfg)
        )
        if not special_days:
            print("No globally relevant holiday/special day found today (based on configured calendars).")
            raise SystemExit(0)
        wish = generate_wish(cfg, special_days, date_ist=date_ist)
        print("--- Special day(s) ---")
        for day in special_days:
            print(f"- {day}")
        print("\n--- Generated wish ---")
        print(wish)
        raise SystemExit(0)

    client.run(cfg.discord_token)
