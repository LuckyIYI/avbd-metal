#!/usr/bin/env python3
"""Fetch a pinned CC0 example. No network access is needed by tests or renderer."""
import hashlib
import pathlib
import sys
import urllib.request

root = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else '/tmp/avbd-material-example')
root.mkdir(parents=True, exist_ok=True)
files = {
    'diff': '56652a7339b16a76afad20d3585f66ea',
    'rough': '05b60bd20a1cc4355c5d73a99b245d44',
    'nor_gl': '6360e09967add8baecaa5ecc55fb5e03',
}
for kind, expected in files.items():
    name = f'wood_table_001_{kind}_1k.jpg'
    data = urllib.request.urlopen(f'https://dl.polyhaven.org/file/ph-assets/Textures/jpg/1k/wood_table_001/{name}', timeout=60).read()
    if hashlib.md5(data).hexdigest() != expected:
        raise RuntimeError(f'Asset integrity check failed: {name}')
    (root/name).write_bytes(data)
print(root)
