import argparse
import datetime
import asyncio

import discord

from main import (
    _stable_daily_index,
    fetch_special_days_for_ist_date,
    generate_wish,
    load_config,
    pick_sticker_by_ai,
)


async def run(user_id: int, date_ist: datetime.date) -> int:
    config = load_config()

    intents = discord.Intents.none()
    async with discord.Client(intents=intents) as client:

        @client.event
        async def on_ready():
            special_days = fetch_special_days_for_ist_date(config, date_ist)
            if not special_days:
                # For a test DM, don't send anything if no special day is found.
                print("No holiday/special day found for that date; not sending.")
                await client.close()
                return

            wish_message = generate_wish(config, special_days, date_ist=date_ist)

            # Resolve a CSD sticker the same way main.py does.
            sticker = None
            if config.sticker_id:
                try:
                    sticker = await client.fetch_sticker(config.sticker_id)
                    print(f"Sticker: using explicit ID {config.sticker_id} ({getattr(sticker, 'name', '')})")
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

            try:
                user = await client.fetch_user(user_id)
                if sticker is not None:
                    try:
                        await user.send(wish_message, stickers=[sticker])
                        print("DM sent (with sticker).")
                    except discord.HTTPException as exc:
                        print(f"DM sticker rejected by Discord, sending text-only. Reason: {exc}")
                        await user.send(wish_message)
                        print("DM sent (text-only).")
                else:
                    await user.send(wish_message)
                    print("DM sent (text-only, no sticker resolved).")
            except discord.Forbidden:
                print("DM failed: user has DMs disabled or blocked the bot.")
            finally:
                await client.close()

        await client.start(config.discord_token)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Send a single test DM for a given date")
    parser.add_argument("user_id", type=int, help="Discord user ID to DM")
    parser.add_argument(
        "--date",
        type=str,
        default="2026-01-01",
        help="IST date to simulate (YYYY-MM-DD). Default: 2026-01-01",
    )
    args = parser.parse_args()

    try:
        date_ist = datetime.date.fromisoformat(args.date)
    except Exception:
        raise SystemExit("Invalid --date, expected YYYY-MM-DD")

    raise SystemExit(asyncio.run(run(args.user_id, date_ist)))
