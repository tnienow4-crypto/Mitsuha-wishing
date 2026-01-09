import argparse
import asyncio
import datetime
from typing import Any

import discord

from main import (
    _stable_daily_index,
    create_premium_occasion_embed,
    fetch_guild_emojis_and_stickers,
    fetch_special_days_for_ist_date,
    generate_channel_wish,
    get_time_of_day_from_ist,
    load_config,
    pick_random_emojis,
    pick_sticker_by_ai,
)


async def run(*, date_ist: datetime.date, channel_id: int | None) -> int:
    config = load_config()
    target_channel_id = channel_id or config.fallback_channel_id

    intents = discord.Intents.none()
    async with discord.Client(intents=intents) as client:

        @client.event
        async def on_ready():
            try:
                special_days = fetch_special_days_for_ist_date(config, date_ist)
                if not special_days:
                    print("No holiday/special day found for that date; not sending.")
                    return

                wish_message = generate_channel_wish(config, special_days, date_ist=date_ist)
                # Clean message for embed (remove @everyone as we'll add it to content)
                wish_message_clean = wish_message.replace("@everyone", "").strip()
                if wish_message_clean.startswith("\n"):
                    wish_message_clean = wish_message_clean[1:]

                # Fetch guild for embed creation
                try:
                    guild = await client.fetch_guild(config.guild_id)
                except Exception as exc:
                    print(f"Failed to fetch guild: {exc}")
                    return

                # Fetch emojis for embed decoration
                emojis, stickers_list = await fetch_guild_emojis_and_stickers(guild)
                picked_emojis = pick_random_emojis(emojis, count=4, date_ist=date_ist)
                custom_emoji_strings = [str(e) for e in picked_emojis]
                print(f"Picked {len(picked_emojis)} custom emojis for embed decoration")

                # Resolve a CSD sticker the same way main.py does.
                sticker = None
                if config.sticker_id:
                    try:
                        sticker = await client.fetch_sticker(config.sticker_id)
                        print(
                            f"Sticker: using explicit ID {config.sticker_id} ({getattr(sticker, 'name', '')})"
                        )
                    except Exception as exc:
                        print(f"Sticker: failed to fetch ID {config.sticker_id}: {exc}")
                        sticker = None
                else:
                    try:
                        stickers = await guild.fetch_stickers()
                    except Exception as exc:
                        print(f"Sticker: failed to fetch guild stickers: {exc}")
                        stickers = []

                    prefix = (config.sticker_prefix or "").strip()
                    candidates = [
                        s
                        for s in stickers
                        if prefix and (getattr(s, "name", "") or "").lower().startswith(prefix.lower())
                    ]

                    if candidates:
                        if config.sticker_pick_mode == "ai":
                            sticker = pick_sticker_by_ai(
                                config,
                                stickers=candidates,
                                prefix=prefix,
                                date_ist=date_ist,
                                special_days=special_days,
                            )
                            if sticker is not None:
                                print(f"Sticker: AI picked '{getattr(sticker, 'name', '')}'")

                        if sticker is None:
                            sticker = candidates[_stable_daily_index(date_ist, len(candidates))]
                            print(f"Sticker: daily picked '{getattr(sticker, 'name', '')}'")
                    else:
                        print(f"Sticker: no guild stickers found with prefix '{prefix}'")

                # Create beautiful embed
                time_of_day = get_time_of_day_from_ist()
                wish_embeds = create_premium_occasion_embed(
                    guild=guild,
                    special_days=special_days,
                    wish_text=wish_message_clean,
                    custom_emojis=custom_emoji_strings,
                    time_of_day=time_of_day,
                )
                print(f"Created {len(wish_embeds)} embeds for channel message")

                channel = await client.fetch_channel(target_channel_id)
                channel_name = getattr(channel, "name", None)
                print(f"Channel: {target_channel_id}{' (' + channel_name + ')' if channel_name else ''}")

                allowed = discord.AllowedMentions(everyone=True, users=False, roles=False)
                
                # Prepare send kwargs with embed
                send_kwargs: dict[str, Any] = {
                    "content": "@everyone",  # Mention in content
                    "embeds": wish_embeds,
                    "allowed_mentions": allowed,
                }

                if sticker is not None:
                    try:
                        send_kwargs["stickers"] = [sticker]
                        await channel.send(**send_kwargs)
                        print("Channel embed message sent (with sticker).")
                    except discord.HTTPException as exc:
                        print(f"Channel sticker send failed, sending embed-only. Reason: {exc}")
                        del send_kwargs["stickers"]
                        await channel.send(**send_kwargs)
                        print("Channel embed message sent (without sticker).")
                else:
                    await channel.send(**send_kwargs)
                    print("Channel embed message sent (no sticker resolved).")
            finally:
                await client.close()

        await client.start(config.discord_token)

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Send a single test wish into a guild channel for a given date (optionally with a CSD sticker)"
    )
    parser.add_argument(
        "--date",
        type=str,
        default="2026-01-01",
        help="IST date to simulate (YYYY-MM-DD). Default: 2026-01-01",
    )
    parser.add_argument(
        "--channel-id",
        type=int,
        default=None,
        help="Override channel id (defaults to CHANNEL_ID/DISCORD_FALLBACK_CHANNEL_ID from .env)",
    )
    args = parser.parse_args()

    try:
        date_ist = datetime.date.fromisoformat(args.date)
    except Exception:
        raise SystemExit("Invalid --date, expected YYYY-MM-DD")

    raise SystemExit(asyncio.run(run(date_ist=date_ist, channel_id=args.channel_id)))
