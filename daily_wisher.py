import argparse
import asyncio
import datetime
import inspect
import json
import os
import re
from typing import Any

import discord
import pytz
from dotenv import load_dotenv

from main import CHANNEL_ID, GUILD_ID, query_gemini_raw


load_dotenv()


parser = argparse.ArgumentParser(description="Daily Wish Bot")
parser.add_argument(
    "--time",
    type=str,
    help="Time of day string (e.g. Morning, Evening). If not provided, auto-detected.",
)
parser.add_argument("--test", action="store_true", help="Run in test mode (prints to console)")
args = parser.parse_args()


IST = pytz.timezone("Asia/Kolkata")


def get_time_of_day() -> str:
    if args.time:
        return args.time

    now = datetime.datetime.now(IST)
    hour = now.hour

    if 5 <= hour < 11:
        return "Morning"
    if 11 <= hour < 15:
        return "Noon"
    if 15 <= hour < 18:
        return "Afternoon"
    if 18 <= hour < 21:
        return "Evening"
    return "Night"


TIME_OF_DAY = get_time_of_day().strip('"').strip("'")
IS_TEST = args.test

print(f"Starting Daily Wisher. Time: {TIME_OF_DAY}, Test Mode: {IS_TEST}")


intents = discord.Intents.default()
intents.members = True
intents.message_content = True
client = discord.Client(intents=intents)


_MENTION_RE = re.compile(r"<@&?\d+>|<@!?\d+>")
_CUSTOM_EMOJI_RE = re.compile(r"<a?:[A-Za-z0-9_~]+:\d+>")


def strip_discord_mentions(text: str) -> str:
    if not text:
        return ""
    text = text.replace("@everyone", "everyone").replace("@here", "here")
    text = _MENTION_RE.sub("", text)
    text = text.replace("@", "")
    return text.strip()


def strip_unicode_emojis(text: str) -> str:
    if not text:
        return ""

    # Best-effort removal of most emoji/pictograph ranges.
    # Intentionally keeps server custom emojis like <:name:id>.
    out: list[str] = []
    for ch in text:
        code = ord(ch)
        if (
            0x1F000 <= code <= 0x1FAFF
            or 0x2600 <= code <= 0x26FF
            or 0x2700 <= code <= 0x27BF
            or 0xFE00 <= code <= 0xFE0F
            or 0x1F1E6 <= code <= 0x1F1FF
        ):
            continue
        out.append(ch)
    return "".join(out)


def enforce_allowed_custom_emojis(text: str, allowed_tokens: list[str]) -> str:
    if not text:
        return ""
    allowed = set(t for t in allowed_tokens if isinstance(t, str) and t.strip())

    def repl(match: re.Match) -> str:
        token = match.group(0)
        return token if token in allowed else ""

    return _CUSTOM_EMOJI_RE.sub(repl, text)


async def fetch_guild_emojis_and_stickers(guild: discord.Guild):
    try:
        emojis = await guild.fetch_emojis()
    except Exception:
        emojis = list(getattr(guild, "emojis", []))

    try:
        stickers = await guild.fetch_stickers()
    except Exception:
        stickers = list(getattr(guild, "stickers", []))

    return emojis, stickers


def _extract_first_json_object(text: str) -> dict:
    if not text:
        return {}
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {}
    candidate = text[start : end + 1]
    try:
        return json.loads(candidate)
    except Exception:
        return {}


async def pick_decorations_with_ai(
    *,
    time_of_day: str,
    wish_text: str,
    emojis: list[Any],
    stickers: list[Any],
):
    emoji_names = [e.name for e in emojis[:25] if getattr(e, "name", None)]
    sticker_names = [s.name for s in stickers[:10] if getattr(s, "name", None)]

    if not emoji_names and not sticker_names:
        return [], None

    prompt = (
        "Pick up to THREE distinct custom emoji names and optionally ONE sticker name to match a daily server wish. "
        "You must choose only from the provided lists. If none fit, return an empty list for emojis and null for sticker. "
        "Never output @everyone, @here, or any mentions.\n\n"
        f"Time of day: {time_of_day}\n"
        f"Wish text: {wish_text}\n\n"
        f"Available custom emoji names: {emoji_names}\n"
        f"Available sticker names: {sticker_names}\n\n"
        "Return STRICT JSON ONLY in this shape: {\"emojis\": [string, ...], \"sticker\": string|null}"
    )

    raw = await asyncio.to_thread(query_gemini_raw, prompt)
    raw = strip_discord_mentions(raw)
    raw = strip_unicode_emojis(raw)
    data = _extract_first_json_object(raw)

    emoji_picks = data.get("emojis") if isinstance(data, dict) else None
    sticker_pick = data.get("sticker") if isinstance(data, dict) else None

    picked_emojis: list[Any] = []
    if isinstance(emoji_picks, list):
        for name in emoji_picks:
            if not isinstance(name, str):
                continue
            obj = next((e for e in emojis if getattr(e, "name", None) == name), None)
            if obj is not None and obj not in picked_emojis:
                picked_emojis.append(obj)
            if len(picked_emojis) >= 3:
                break

    picked_sticker = (
        next((s for s in stickers if getattr(s, "name", None) == sticker_pick), None)
        if isinstance(sticker_pick, str) and sticker_pick
        else None
    )

    return picked_emojis, picked_sticker


async def rewrite_wish_with_custom_emojis(
    *,
    time_of_day: str,
    wish_text: str,
    allowed_emoji_tokens: list[str],
) -> str:
    wish_text = strip_discord_mentions(wish_text)
    wish_text = strip_unicode_emojis(wish_text)

    allowed_emoji_tokens = [t for t in allowed_emoji_tokens if isinstance(t, str) and t.strip()]
    if not allowed_emoji_tokens:
        return wish_text.strip()

    prompt = (
        "Rewrite this Discord server wish to be decorative using ONLY the provided custom emoji tokens. "
        "Do not use any Unicode emojis at all. "
        "Do not mention any users or roles. Do not use @everyone or @here. "
        "Avoid addressing the message as 'everyone'. Keep it friendly and not annoying.\n\n"
        f"Time of day: {time_of_day}\n\n"
        f"Allowed custom emoji tokens (use only these): {allowed_emoji_tokens}\n\n"
        f"Wish to rewrite:\n{wish_text}\n\n"
        "Return ONLY the rewritten wish text."
    )

    rewritten = await asyncio.to_thread(query_gemini_raw, prompt)
    rewritten = strip_discord_mentions(rewritten)
    rewritten = strip_unicode_emojis(rewritten)
    rewritten = enforce_allowed_custom_emojis(rewritten, allowed_emoji_tokens)
    return rewritten.strip()


def format_decorated_wish(*, time_of_day: str, wish_text: str, custom_emojis: list[str]) -> str:
    wish_text = strip_discord_mentions(wish_text)
    wish_text = strip_unicode_emojis(wish_text)

    unique = [e for e in (custom_emojis or []) if isinstance(e, str) and e.strip()]
    unique = list(dict.fromkeys(unique))

    if len(unique) >= 2:
        header = f"{unique[0]} **Good {time_of_day}!** {unique[1]}"
    elif len(unique) == 1:
        header = f"{unique[0]} **Good {time_of_day}!**"
    else:
        header = f"**Good {time_of_day}!**"

    footer = " ".join(unique[:3]).strip()
    if footer:
        return f"{header}\n{wish_text}\n{footer}".strip()
    return f"{header}\n{wish_text}".strip()


# Time-of-day theme configurations for embeds
TIME_THEMES = {
    "Morning": {
        "color": 0xFFD700,  # Golden sunrise
        "emoji": "🌅",
        "greeting": "Rise & Shine!",
        "quote_prompt": "morning motivation",
        "decorative_line": "═══════════════════════════",
    },
    "Noon": {
        "color": 0xFF6B35,  # Bright orange sun
        "emoji": "☀️",
        "greeting": "Midday Vibes!",
        "quote_prompt": "afternoon energy",
        "decorative_line": "═══════════════════════════",
    },
    "Afternoon": {
        "color": 0xFFA500,  # Warm orange
        "emoji": "🌤️",
        "greeting": "Afternoon Bliss!",
        "quote_prompt": "afternoon relaxation",
        "decorative_line": "═══════════════════════════",
    },
    "Evening": {
        "color": 0x9B59B6,  # Purple twilight
        "emoji": "🌆",
        "greeting": "Evening Serenity!",
        "quote_prompt": "evening peace",
        "decorative_line": "═══════════════════════════",
    },
    "Night": {
        "color": 0x2C3E50,  # Deep night blue
        "emoji": "🌙",
        "greeting": "Sweet Dreams!",
        "quote_prompt": "goodnight wishes",
        "decorative_line": "═══════════════════════════",
    },
}


def create_wish_embed(
    *,
    guild: discord.Guild,
    time_of_day: str,
    wish_text: str,
    custom_emojis: list[str],
    image_filename: str | None = None,
) -> discord.Embed:
    """Create a beautiful professional embed for daily wishes."""
    
    theme = TIME_THEMES.get(time_of_day, TIME_THEMES["Morning"])
    unique_emojis = [e for e in (custom_emojis or []) if isinstance(e, str) and e.strip()]
    unique_emojis = list(dict.fromkeys(unique_emojis))
    
    # Build decorative emoji string
    emoji_decoration = " ".join(unique_emojis[:3]) if unique_emojis else ""
    
    # Create embed with theme color
    embed = discord.Embed(
        color=theme["color"],
        timestamp=datetime.datetime.now(IST),
    )
    
    # Set author with server info and icon
    server_icon_url = guild.icon.url if guild.icon else None
    embed.set_author(
        name=f"✨ {guild.name} ✨",
        icon_url=server_icon_url,
    )
    
    # Build the title with emojis
    title_emojis_left = unique_emojis[0] if len(unique_emojis) >= 1 else theme["emoji"]
    title_emojis_right = unique_emojis[1] if len(unique_emojis) >= 2 else theme["emoji"]
    
    embed.title = f"{title_emojis_left} 𝐆𝐨𝐨𝐝 {time_of_day}! {title_emojis_right}"
    
    # Build description with decorative elements
    decorative_top = f"╭{'─' * 25}╮"
    decorative_bottom = f"╰{'─' * 25}╯"
    
    # Format the wish text nicely
    wish_text = strip_discord_mentions(wish_text)
    wish_text = strip_unicode_emojis(wish_text)
    
    description_parts = [
        f"```ansi\n\u001b[1;33m{theme['greeting']}\u001b[0m\n```",
        "",
        f"{theme['emoji']} {wish_text}",
        "",
    ]
    
    # Add custom emojis decoration if available
    if emoji_decoration:
        description_parts.append(f"\n{emoji_decoration}")
    
    embed.description = "\n".join(description_parts)
    
    # Add inspirational quote field
    embed.add_field(
        name=f"━━━━━━━━━━━━━━━━━",
        value=f"*Wishing you all the best today!* {theme['emoji']}",
        inline=False,
    )
    
    # Set image (the wish image)
    if image_filename:
        embed.set_image(url=f"attachment://{image_filename}")
    
    # Set thumbnail to server icon for extra flair
    if server_icon_url:
        embed.set_thumbnail(url=server_icon_url)
    
    # Set footer with bot signature and custom emojis
    footer_emoji = unique_emojis[-1] if unique_emojis else "💫"
    embed.set_footer(
        text=f"From Sazami with love {theme['emoji']} • Have a wonderful {time_of_day.lower()}!",
        icon_url=server_icon_url,
    )
    
    return embed


def create_premium_wish_embed(
    *,
    guild: discord.Guild,
    time_of_day: str,
    wish_text: str,
    custom_emojis: list[str],
    image_filename: str | None = None,
) -> list[discord.Embed]:
    """Create a premium-style beautiful embed with more decorations.
    
    Returns a list of embeds (banner embed + main wish embed if banner exists).
    """
    
    theme = TIME_THEMES.get(time_of_day, TIME_THEMES["Morning"])
    unique_emojis = [e for e in (custom_emojis or []) if isinstance(e, str) and e.strip()]
    unique_emojis = list(dict.fromkeys(unique_emojis))
    
    server_icon_url = guild.icon.url if guild.icon else None
    server_banner_url = guild.banner.url if guild.banner else None
    
    embeds: list[discord.Embed] = []
    
    # Create banner embed if server has a banner
    if server_banner_url:
        banner_embed = discord.Embed(color=theme["color"])
        banner_embed.set_image(url=server_banner_url)
        embeds.append(banner_embed)
    
    # Main wish embed
    main_embed = discord.Embed(
        color=theme["color"],
        timestamp=datetime.datetime.now(IST),
    )
    
    # Author with server branding
    main_embed.set_author(
        name=f"『 {guild.name} 』",
        icon_url=server_icon_url,
    )
    
    # Stylized title with custom emojis
    left_deco = unique_emojis[0] if len(unique_emojis) >= 1 else "✦"
    right_deco = unique_emojis[1] if len(unique_emojis) >= 2 else "✦"
    center_emoji = theme["emoji"]
    
    main_embed.title = f"{left_deco} ─ {center_emoji} 𝑮𝒐𝒐𝒅 {time_of_day} {center_emoji} ─ {right_deco}"
    
    # Clean the wish text
    wish_text = strip_discord_mentions(wish_text)
    wish_text = strip_unicode_emojis(wish_text)
    
    # Build beautiful description
    sparkle_line = "･ﾟ✧ ━━━━━━━━━━━━━━ ✧ﾟ･"
    emoji_row = " ".join(unique_emojis[:4]) if unique_emojis else f"{theme['emoji']} ✨ 💫 🌟"
    
    description = f"""
{sparkle_line}

{theme['emoji']} **{theme['greeting']}** {theme['emoji']}

{wish_text}

{sparkle_line}

{emoji_row}
"""
    
    main_embed.description = description.strip()
    
    # Add a motivational field
    main_embed.add_field(
        name=f"💝 ── Today's Blessing ── 💝",
        value=f"> *May your {time_of_day.lower()} be filled with joy and positivity!*",
        inline=False,
    )
    
    # Set main image (wish image)
    if image_filename:
        main_embed.set_image(url=f"attachment://{image_filename}")
    
    # Set thumbnail to server icon
    if server_icon_url:
        main_embed.set_thumbnail(url=server_icon_url)
    
    # Beautiful footer
    footer_text = f"✿ Sazami • {guild.name} ✿ • Happy {time_of_day}!"
    main_embed.set_footer(
        text=footer_text,
        icon_url=server_icon_url,
    )
    
    embeds.append(main_embed)
    return embeds


@client.event
async def on_ready():
    print(f"Logged in as {client.user}")

    try:
        guild = client.get_guild(GUILD_ID)
        if not guild:
            print(f"CRITICAL: Guild with ID {GUILD_ID} not found.")
            await client.close()
            return

        channel = guild.get_channel(CHANNEL_ID)
        if not channel:
            print(f"WARNING: Channel with ID {CHANNEL_ID} not found.")

        async def generate_with_retry(prompt: str, fallback: str) -> str:
            for i in range(3):
                msg = await asyncio.to_thread(query_gemini_raw, prompt)
                if "Gemini API Error" not in msg:
                    return msg
                print(f"Generation failed ({msg}). Retrying {i + 1}/3...")
                await asyncio.sleep(5)
            return fallback

        print("Generating daily server wish...")
        base_wish = await generate_with_retry(
            (
                f"Write a cheerful {TIME_OF_DAY} wish for a Discord server. "
                "Keep it friendly and short (1-2 paragraphs max). Do not add emojis (they will be added later). "
                "Do NOT mention or tag any users or roles. "
                "Do NOT use @everyone or @here. "
                "Do NOT include any user names. "
                "Avoid addressing it as 'everyone'."
            ),
            f"Wishing you a lovely {TIME_OF_DAY}! Stay safe, stay strong, and have a beautiful rest ahead.",
        )
        base_wish = strip_discord_mentions(base_wish)
        base_wish = strip_unicode_emojis(base_wish)

        image_filename = f"good-{TIME_OF_DAY.lower()}.png"
        image_path = os.path.join("assets", image_filename)
        has_image = os.path.exists(image_path)
        if not has_image:
            print(f"WARNING: Image not found at {image_path}. Sending text only.")

        emojis, stickers = await fetch_guild_emojis_and_stickers(guild)
        allowed_emoji_tokens = [str(e) for e in emojis[:25]]

        # Replace any in-message emojis with server custom emojis only.
        base_wish = await rewrite_wish_with_custom_emojis(
            time_of_day=TIME_OF_DAY,
            wish_text=base_wish,
            allowed_emoji_tokens=allowed_emoji_tokens,
        )

        picked_emojis, picked_sticker = await pick_decorations_with_ai(
            time_of_day=TIME_OF_DAY,
            wish_text=base_wish,
            emojis=emojis,
            stickers=stickers,
        )

        # Get custom emoji strings for the embed
        custom_emoji_strings = [str(e) for e in (picked_emojis or [])]

        # Create the beautiful embeds (returns list with banner + main embed)
        wish_embeds = create_premium_wish_embed(
            guild=guild,
            time_of_day=TIME_OF_DAY,
            wish_text=base_wish,
            custom_emojis=custom_emoji_strings,
            image_filename=image_filename if has_image else None,
        )

        if not channel:
            return

        if IS_TEST:
            print(
                f"[TEST] Channel Embed to #{channel.name}:\n"
                f"  Embeds Count: {len(wish_embeds)}\n"
                f"  Title: {wish_embeds[-1].title}\n"
                f"  Description: {wish_embeds[-1].description[:100]}...\n"
                f"  [Sticker: {getattr(picked_sticker, 'name', None)}] "
                f"[Image: {image_filename if has_image else 'None'}]"
            )
            return

        send_kwargs: dict[str, Any] = {
            "embeds": wish_embeds,
            "allowed_mentions": discord.AllowedMentions.none(),
        }

        # Try to suppress notifications if supported.
        try:
            if "silent" in inspect.signature(channel.send).parameters:
                send_kwargs["silent"] = True
        except Exception:
            pass

        if has_image:
            send_kwargs["file"] = discord.File(image_path)

        if picked_sticker is not None:
            try:
                if "stickers" in inspect.signature(channel.send).parameters:
                    send_kwargs["stickers"] = [picked_sticker]
            except Exception:
                pass

        await channel.send(**send_kwargs)
        print(f"Sent server wish embed to #{channel.name}")

    except Exception as e:
        print(f"An error occurred during execution: {e}")
    finally:
        print("Done. Closing client.")
        await client.close()


if __name__ == "__main__":
    client.run(os.getenv("BOT_TOKEN"))
