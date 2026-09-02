#!/usr/bin/env python3
"""Edit JEDI YAML to turn on the non-Gaussian observation cost function.

Two edits are needed and they live in DIFFERENT files:

    density   -> the obtype template, e.g.
                 validated_yamls/templates/obtype_config/adpsfc_airTemperature_181.yaml
                 (patch / patch_file)
    switch on -> `jo type: evolving gaussian` under `observations:` in the basic
                 config, e.g. basic_config/mpasjedi_hybrid3denvar.yaml
                 (ensure_jo_type / ensure_jo_type_file)

Without the second the density is inert and the run stays Gaussian.

Editing is textual, so comments and everything else in the file survive, which a
YAML load/dump round-trip would not manage. A backup is written alongside unless
suppressed.

    python3 patch_yaml.py template.yaml block.txt            # replace/insert
    python3 patch_yaml.py template.yaml block.txt --dry-run
"""

import argparse
import shutil
import sys

KEY = "non gaussian cost:"
JO_KEY = "jo type:"


def _indent_of(line):
    return len(line) - len(line.lstrip(" "))


def patch(text, block):
    """Insert or replace the `non gaussian cost` block.

    The existing key and every following line indented more deeply are removed,
    then the new block is inserted at the same place, re-indented to match. If
    the key is absent the block is appended.

    Returns (new_text, action) with action 'replaced' or 'appended'.
    """
    lines = text.splitlines()
    start = None
    for i, ln in enumerate(lines):
        if ln.strip() == KEY:
            start = i
            break

    if start is None:
        out = lines + ([""] if lines and lines[-1].strip() else []) \
            + block.splitlines()
        return "\n".join(out) + "\n", "appended"

    base = _indent_of(lines[start])
    end = start + 1
    while end < len(lines):
        ln = lines[end]
        if ln.strip() and _indent_of(ln) <= base:
            break
        end += 1

    blines = block.splitlines()
    shift = base - _indent_of(blines[0])
    if shift > 0:
        blines = [(" " * shift) + b if b.strip() else b for b in blines]
    elif shift < 0:
        blines = [b[-shift:] if b.startswith(" " * -shift) else b.lstrip()
                  for b in blines]

    out = lines[:start] + blines + lines[end:]
    return "\n".join(out) + "\n", "replaced"


def ensure_jo_type(text, value="evolving gaussian"):
    """Set `jo type` under the `observations:` mapping, which is what actually
    switches the cost function on.

    Returns (new_text, action) with action 'already set', 'updated' or 'added'.
    """
    lines = text.splitlines()
    obs_i = None
    for i, ln in enumerate(lines):
        if ln.strip() == "observations:":
            obs_i = i
            break
    if obs_i is None:
        raise ValueError("no `observations:` key found; is this the basic config?")

    base = _indent_of(lines[obs_i])
    j = obs_i + 1
    while j < len(lines):
        ln = lines[j]
        if ln.strip() and _indent_of(ln) <= base:
            break
        if ln.strip().startswith(JO_KEY):
            if ln.split(":", 1)[1].strip() == value:
                return text, "already set"
            lines[j] = " " * _indent_of(ln) + f"{JO_KEY} {value}"
            return "\n".join(lines) + "\n", "updated"
        j += 1

    child = base + 2
    for k in range(obs_i + 1, len(lines)):
        if lines[k].strip():
            child = _indent_of(lines[k])
            break
    lines.insert(obs_i + 1, " " * child + f"{JO_KEY} {value}")
    return "\n".join(lines) + "\n", "added"


def _write(path, new, backup):
    if backup:
        shutil.copy2(path, path + ".bak")
    with open(path, "w") as f:
        f.write(new)


def patch_file(path, block, dry_run=False, backup=True):
    new, action = patch(open(path).read(), block)
    if not dry_run:
        _write(path, new, backup)
    return action, new


def ensure_jo_type_file(path, value="evolving gaussian", dry_run=False,
                        backup=True):
    new, action = ensure_jo_type(open(path).read(), value)
    if not dry_run and action != "already set":
        _write(path, new, backup)
    return action, new


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("yaml_file")
    ap.add_argument("block_file",
                    help="file containing the `non gaussian cost` block")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-backup", action="store_true")
    a = ap.parse_args()
    block = open(a.block_file).read().rstrip("\n")
    action, new = patch_file(a.yaml_file, block, a.dry_run, not a.no_backup)
    if a.dry_run:
        print(f"--- would be {action}; result ---")
        print(new)
    else:
        print(f"{action} `{KEY}` in {a.yaml_file}"
              + ("" if a.no_backup else f" (backup at {a.yaml_file}.bak)"))


if __name__ == "__main__":
    sys.exit(main())
