import os

if os.path.exists('.env'):
    print(".env found")
    with open('.env', 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if '=' in line:
                key = line.split('=')[0].strip()
                val_len = len(line.split('=', 1)[1].strip())
                print(f"Line {i+1}: Key='{key}', Value_Length={val_len}")
            else:
                print(f"Line {i+1}: SKIPPED (no = found): {line}")
else:
    print(".env NOT found")
