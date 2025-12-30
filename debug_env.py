from dotenv import load_dotenv
import os

print("Current CWD:", os.getcwd())
print("Files in CWD:", os.listdir('.'))
loaded = load_dotenv()
print("Dotenv loaded:", loaded)
print("DISCORD_TOKEN present:", 'DISCORD_TOKEN' in os.environ)
if 'DISCORD_TOKEN' in os.environ:
    print("DISCORD_TOKEN length:", len(os.environ['DISCORD_TOKEN']))
else:
    print("Env vars keys:", list(os.environ.keys()))
