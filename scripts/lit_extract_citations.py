"""List every citation in paper/main.tex with the sentence in which it is used (input for the citation audit)."""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    tex = (ROOT / "paper" / "main.tex").read_text(encoding="utf-8")
    tex = re.sub(r"(?<!\\)%.*", "", tex)
    body = re.sub(r"\s+", " ", tex)
    sents = re.split(r"(?<=[.;])\s+(?=[A-Z\\])", body)
    uses = {}
    for s in sents:
        for m in re.finditer(r"\\cite[a-z]*\*?(?:\[[^\]]*\])*\{([^}]+)\}", s):
            for k in m.group(1).split(","):
                uses.setdefault(k.strip(), []).append(s.strip()[:500])
    bib = (ROOT / "paper" / "refs.bib").read_text(encoding="utf-8")
    keys = re.findall(r"@\w+\{(\w+),", bib)
    print("cited but not in bib:", sorted(set(uses) - set(keys)))
    print("in bib but not cited:", sorted(set(keys) - set(uses)))
    out = ROOT / "docs" / "citation_uses.json"
    out.write_text(json.dumps(uses, indent=1, ensure_ascii=False), encoding="utf-8")
    if "-v" in sys.argv:
        for k, v in uses.items():
            print(f"\n### {k} ({len(v)})")
            for s in v:
                print("  -", s)
