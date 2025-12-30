import os

if os.path.exists('.env'):
    print("--- .env keys start ---")
    with open('.env', 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if '=' in line:
                key = line.split('=', 1)[0].strip()
                print(f"Key Found: {key}")
    print("--- .env keys end ---")
