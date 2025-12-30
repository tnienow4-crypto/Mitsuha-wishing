import argparse
import asyncio
import datetime

import discord

from main import (
    _stable_daily_index,
    fetch_special_days_for_ist_date,
    generate_channel_wish,
    load_config,
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
                        guild = await client.fetch_guild(config.guild_id)
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

                channel = await client.fetch_channel(target_channel_id)
                channel_name = getattr(channel, "name", None)
                print(f"Channel: {target_channel_id}{' (' + channel_name + ')' if channel_name else ''}")

                allowed = discord.AllowedMentions(everyone=True, users=False, roles=False)

                if sticker is not None:
                    try:
                        await channel.send(wish_message, stickers=[sticker], allowed_mentions=allowed)
                        print("Channel message sent (with sticker).")
                    except discord.HTTPException as exc:
                        print(f"Channel sticker send failed, sending text-only. Reason: {exc}")
                        await channel.send(wish_message, allowed_mentions=allowed)
                        print("Channel message sent (text-only).")
                else:
                    await channel.send(wish_message, allowed_mentions=allowed)
                    print("Channel message sent (text-only, no sticker resolved).")
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
