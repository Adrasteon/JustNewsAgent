import glob
import re

for f in glob.glob('**/*.md', recursive=True):
    try:
        with open(f, encoding='utf-8') as fh:
            lines = fh.readlines()
        if not lines:
            continue
        if lines[0].strip() == '---':
            second = ''
            for l in lines[1:6]:
                if l.strip():
                    second = l.strip()
                    break
            if not re.match(r'^[A-Za-z0-9_\-]+:\s+', second):
                print(f"{f}: second=\"{second}\"")
    except Exception as e:
        print(f"ERR {f}: {e}")
