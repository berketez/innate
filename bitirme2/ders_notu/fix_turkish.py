#!/usr/bin/env python3
"""Convert LaTeX Turkish escape sequences to UTF-8 characters."""
import re
import glob
import os

# Replacement map: LaTeX escape -> UTF-8
REPLACEMENTS = [
    # Must do multi-char sequences first (longer patterns first)
    (r'\"{o}', 'ö'),
    (r'\"{O}', 'Ö'),
    (r'\"{u}', 'ü'),
    (r'\"{U}', 'Ü'),
    (r'\"{a}', 'ä'),  # unlikely but just in case
    (r'\"{A}', 'Ä'),
    (r'\c{c}', 'ç'),
    (r'\c{C}', 'Ç'),
    (r'\c{s}', 'ş'),
    (r'\c{S}', 'Ş'),
    (r'\u{g}', 'ğ'),
    (r'\u{G}', 'Ğ'),
    (r'\.{I}', 'İ'),
    (r'\^{a}', 'â'),
    (r'\^{A}', 'Â'),
    (r'\^{i}', 'î'),
    (r'\^{u}', 'û'),
]

# {\i} is special - dotless i
# Need regex because it could appear in various contexts
DOTLESS_I_PATTERN = re.compile(r'\{\\i\}')

def convert_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    original = content

    # Apply simple replacements
    for old, new in REPLACEMENTS:
        content = content.replace(old, new)

    # Handle {\i} -> ı (dotless i)
    content = DOTLESS_I_PATTERN.sub('ı', content)

    if content != original:
        count = sum(original.count(old) for old, _ in REPLACEMENTS)
        count += len(DOTLESS_I_PATTERN.findall(original))
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"  {os.path.basename(filepath)}: {count} replacements")
        return count
    else:
        print(f"  {os.path.basename(filepath)}: no changes")
        return 0

def main():
    chapters_dir = os.path.join(os.path.dirname(__file__), 'chapters')
    tex_files = sorted(glob.glob(os.path.join(chapters_dir, '*.tex')))

    total = 0
    for f in tex_files:
        total += convert_file(f)

    print(f"\nTotal: {total} replacements across {len(tex_files)} files")

if __name__ == '__main__':
    main()
