"""Citation audit, step 1: compare every bibliography entry with OpenAlex metadata and fetch abstracts.

For each entry: resolve by DOI (or by title search when there is no DOI), then compare title, first author, year,
volume, issue and pages; report the venue, its type (journal / conference / repository) and retraction status.
Output: docs/lit/metadata_audit.csv and docs/lit/abstracts.json.
"""
import difflib
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "lit"
OUT.mkdir(parents=True, exist_ok=True)
UA = {"User-Agent": "thesis-citation-audit (mailto:mkaszubowski750@gmail.com)"}


def parse_bib(text):
    entries = {}
    text = re.sub(r"(?m)^\s*%.*$", "", text)
    for m in re.finditer(r"@(\w+)\{(\w+),(.*?)(?=\n\s*@|\Z)", text, flags=re.S):
        typ, key, body = m.groups()
        fields = {k.lower(): v.strip() for k, v in re.findall(r"(\w+)\s*=\s*\{((?:[^{}]|\{[^{}]*\})*)\}", body)}
        fields["type"] = typ
        entries[key] = fields
    return entries


def get(url):
    for _ in range(3):
        try:
            return json.load(urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30))
        except Exception:
            time.sleep(2)
    return None


def clean(s):
    s = re.sub(r"\\[a-zA-Z]+|[{}\\$]", "", s or "")
    return re.sub(r"\s+", " ", s).strip().lower()


def abstract(inv):
    if not inv:
        return ""
    pos = sorted((p, w) for w, ps in inv.items() for p in ps)
    return " ".join(w for _, w in pos)


if __name__ == "__main__":
    bib = parse_bib((ROOT / "paper" / "refs.bib").read_text(encoding="utf-8"))
    rows, abstracts = [], {}
    for key, f in bib.items():
        w = None
        if "doi" in f:
            w = get("https://api.openalex.org/works/doi:" + f["doi"])
        if w is None:
            q = urllib.parse.quote(clean(f.get("title", ""))[:200])
            res = get(f"https://api.openalex.org/works?search={q}&per-page=5")
            cands = (res or {}).get("results", [])
            if cands:
                w = max(cands, key=lambda c: difflib.SequenceMatcher(None, clean(c.get("title")), clean(f.get("title"))).ratio())
        if w is None:
            rows.append(dict(key=key, status="NOT FOUND"))
            continue
        loc = w.get("primary_location") or {}
        src = loc.get("source") or {}
        bibl = w.get("biblio") or {}
        pages_oa = "--".join(x for x in (bibl.get("first_page"), bibl.get("last_page")) if x)
        first_author_oa = (w.get("authorships") or [{}])[0].get("author", {}).get("display_name", "")
        first_author_bib = clean(f.get("author", "").split(" and ")[0].split(",")[0])
        rows.append(dict(
            key=key,
            title_sim=round(difflib.SequenceMatcher(None, clean(w.get("title")), clean(f.get("title"))).ratio(), 2),
            author_ok=first_author_bib.split()[-1] in clean(first_author_oa) if first_author_bib else None,
            year_bib=f.get("year"), year_oa=w.get("publication_year"),
            vol_bib=f.get("volume"), vol_oa=bibl.get("volume"),
            iss_bib=f.get("number"), iss_oa=bibl.get("issue"),
            pages_bib=f.get("pages", "").replace("-", "").replace(" ", ""), pages_oa=pages_oa.replace("-", ""),
            venue_oa=src.get("display_name"), venue_type=src.get("type"), work_type=w.get("type"),
            retracted=w.get("is_retracted"), doi_oa=(w.get("doi") or "").replace("https://doi.org/", ""),
            cited_by=w.get("cited_by_count"), oa_id=w.get("id"),
        ))
        abstracts[key] = dict(title=w.get("title"), abstract=abstract(w.get("abstract_inverted_index")))
        time.sleep(0.2)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "metadata_audit.csv", index=False)
    (OUT / "abstracts.json").write_text(json.dumps(abstracts, indent=1, ensure_ascii=False), encoding="utf-8")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(df.drop(columns=["oa_id", "doi_oa"]).to_string(index=False))
