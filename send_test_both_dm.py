"""Send BOTH the personal DM wish AND the server-style embed to a single user's DM for preview."""
import argparse
import asyncio
import datetime
from typing import Any

import discord

from main import (
    _naruto_font_available,
    _stable_daily_index,
    build_naruto_font_heading,
    create_premium_occasion_embed,
    fetch_guild_emojis_and_stickers,
    fetch_special_days_for_ist_date,
    generate_channel_wish,
    generate_day_description,
    generate_wish,
    get_time_of_day_from_ist,
    load_config,
    personalize_dm_message,
    pick_random_emojis,
    pick_sticker_by_ai,
)


async def run(user_id: int, date_ist: datetime.date) -> int:
    config = load_config()

    intents = discord.Intents.none()
    async with discord.Client(intents=intents) as client:

        @client.event
        async def on_ready():
            try:
                special_days = fetch_special_days_for_ist_date(config, date_ist)
                if not special_days:
                    print("No holiday/special day found for that date; not sending.")
                    return

                print(f"Special day(s): {special_days}")

                # ── Generate all content ──
                dm_wish = generate_wish(config, special_days, date_ist=date_ist)
                channel_wish = generate_channel_wish(config, special_days, date_ist=date_ist)
                channel_wish_clean = channel_wish.replace("@everyone", "").strip()
                if channel_wish_clean.startswith("\n"):
                    channel_wish_clean = channel_wish_clean[1:]

                day_description = generate_day_description(config, special_days, date_ist=date_ist)
                print(f"Day description: {day_description[:80]}...")

                # ── Fetch guild, emojis, stickers ──
                guild = await client.fetch_guild(config.guild_id)
                emojis, stickers_list = await fetch_guild_emojis_and_stickers(guild)
                picked_emojis = pick_random_emojis(emojis, count=4, date_ist=date_ist)
                custom_emoji_strings = [str(e) for e in picked_emojis]
                print(f"Picked {len(picked_emojis)} custom emojis")

                # ── Build NarutoFonts heading ──
                naruto_heading = ""
                if _naruto_font_available(emojis):
                    day_title = special_days[0] if special_days else "Special Day"
                    naruto_heading = build_naruto_font_heading(day_title, emojis)
                    print(f"NarutoFonts heading built ({len(naruto_heading)} chars)")
                else:
                    print("NarutoFonts emojis not available. Using bold fallback.")

                # ── Resolve sticker ──
                sticker = None
                prefix = (config.sticker_prefix or "").strip()
                candidates = [
                    s for s in stickers_list
                    if prefix and (getattr(s, "name", "") or "").lower().startswith(prefix.lower())
                ]
                if config.sticker_id:
                    try:
                        sticker = await client.fetch_sticker(config.sticker_id)
                    except Exception:
                        pass
                elif candidates:
                    if config.sticker_pick_mode == "ai":
                        sticker = pick_sticker_by_ai(
                            config, stickers=candidates, prefix=prefix,
                            date_ist=date_ist, special_days=special_days,
                        )
                    if sticker is None:
                        sticker = candidates[_stable_daily_index(date_ist, len(candidates))]
                if sticker:
                    print(f"Sticker: {getattr(sticker, 'name', '???')}")

                # ── Fetch user ──
                user = await client.fetch_user(user_id)
                display_name = (
                    getattr(user, "display_name", None)
                    or getattr(user, "global_name", None)
                    or getattr(user, "name", None)
                    or "there"
                )
                print(f"Sending to: {display_name} ({user_id})")

                # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                # 1) PERSONAL DM WISH
                # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                dm_text = personalize_dm_message(
                    dm_wish,
                    str(display_name),
                    naruto_heading=naruto_heading,
                    day_description=day_description,
                    special_days=special_days,
                )
                try:
                    await user.send(dm_text)
                    print("✓ Personal DM wish sent!")
                except discord.HTTPException as exc:
                    print(f"✗ Personal DM failed: {exc}")

                await asyncio.sleep(1)  # small pause so they arrive in order

                # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                # 2) SERVER-STYLE EMBED (sent as DM for preview)
                # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                time_of_day = get_time_of_day_from_ist()
                wish_embeds = create_premium_occasion_embed(
                    guild=guild,
                    special_days=special_days,
                    wish_text=channel_wish_clean,
                    custom_emojis=custom_emoji_strings,
                    time_of_day=time_of_day,
                    day_description=day_description,
                    naruto_heading=naruto_heading,
                )

                send_kwargs: dict[str, Any] = {
                    "content": "**[Server Wish Preview]** — this is how the channel embed would look:",
                    "embeds": wish_embeds,
                }
                if sticker:
                    try:
                        send_kwargs["stickers"] = [sticker]
                        await user.send(**send_kwargs)
                        print("✓ Server embed DM sent (with sticker)!")
                    except discord.HTTPException:
                        del send_kwargs["stickers"]
                        await user.send(**send_kwargs)
                        print("✓ Server embed DM sent (without sticker).")
                else:
                    await user.send(**send_kwargs)
                    print("✓ Server embed DM sent!")

            except discord.Forbidden:
                print("✗ Cannot DM user — DMs disabled or bot blocked.")
            except Exception as exc:
                print(f"✗ Error: {exc}")
                import traceback; traceback.print_exc()
            finally:
                await client.close()

        await client.start(config.discord_token)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Send both DM wish + server embed preview to a user's DM"
    )
    parser.add_argument("user_id", type=int, help="Discord user ID to DM")
    parser.add_argument(
        "--date", type=str, default=None,
        help="IST date to simulate (YYYY-MM-DD). Default: today",
    )
    args = parser.parse_args()

    if args.date:
        try:
            date_ist = datetime.date.fromisoformat(args.date)
        except Exception:
            raise SystemExit("Invalid --date, expected YYYY-MM-DD")
    else:
        date_ist = datetime.date.today()

    raise SystemExit(asyncio.run(run(args.user_id, date_ist)))
