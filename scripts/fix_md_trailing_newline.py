import glob

fixed = []
for path in glob.glob('**/*.md', recursive=True):
    try:
        with open(path, 'rb') as f:
            data = f.read()
        if not data:
            continue
        if data.endswith(b'\n'):
            continue
        # Append a single newline
        with open(path, 'ab') as f:
            f.write(b'\n')
        fixed.append(path)
    except Exception as e:
        print(f"ERR {path}: {e}")

if fixed:
    print('Fixed trailing newlines for:')
    for p in fixed:
        print(' -', p)
else:
    print('No trailing newline fixes required')
