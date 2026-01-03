import re
import os
import math
import json
import time
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Tuple, Optional

import streamlit as st
from dotenv import load_dotenv

# --- Load .env from project root reliably (works no matter where you run from) ---
ROOT_DIR = Path(__file__).resolve().parents[1]   # scripts/ -> project root
load_dotenv(ROOT_DIR / ".env")

# --- LLM config (Streamlit Secrets > env vars > defaults) ---
DEFAULT_BASE_URL = "https://litellm.oit.duke.edu/v1"
DEFAULT_MODEL = "GPT 4.1 Mini"  # allowed by Duke LiteLLM
bronze_dir = "data/bronze/ncbc"

DUKE_LLM_BASE_URL = st.secrets.get(
    "DUKE_LLM_BASE_URL",
    os.getenv("DUKE_LLM_BASE_URL", DEFAULT_BASE_URL),
)
DUKE_LLM_API_KEY = st.secrets.get(
    "DUKE_LLM_API_KEY",
    os.getenv("DUKE_LLM_API_KEY", ""),
)
DUKE_LLM_MODEL = st.secrets.get(
    "DUKE_LLM_MODEL",
    os.getenv("DUKE_LLM_MODEL", DEFAULT_MODEL),
)

REQUIRE_LLM_KEY = True
if REQUIRE_LLM_KEY and not str(DUKE_LLM_API_KEY).strip():
    st.error("Missing DUKE_LLM_API_KEY in Streamlit Secrets or .env.")
    st.stop()

ALLOWED_MODELS = {
    "GPT 4.1",
    "GPT 4.1 Mini",
    "GPT 4.1 Nano",
    "Mistral on-site",
}

if DUKE_LLM_MODEL not in ALLOWED_MODELS:
    st.error(
        f"Model '{DUKE_LLM_MODEL}' is not allowed for your team. "
        f"Set DUKE_LLM_MODEL to one of: {sorted(ALLOWED_MODELS)}"
    )
    st.stop()

# -------------------------
# Phase 1 Extractor (facts-only, no legal conclusions)
# -------------------------

WS = re.compile(r"\s+")

def norm(s: str) -> str:
    return WS.sub(" ", (s or "").strip())

def uniq_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in items:
        x = norm(x)
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out

def build_retrieval_query(facts: dict) -> str:
    """
    Build a high-signal lexical query for Phase 2 retrieval.
    Intentionally simple and transparent.
    """
    bits = []
    bits += facts.get("key_phrases", [])
    bits += facts.get("roles", [])
    bits += facts.get("actions", [])

    # Drop procedural noise unless explicitly needed later
    drop = {
        "procedural_motion",
        "motion_to_dismiss",
        "summary_judgment",
        "prelim_injunction",
    }
    bits = [b for b in bits if b not in drop]

    # Deduplicate while preserving order
    seen = set()
    out = []
    for b in bits:
        if b and b not in seen:
            seen.add(b)
            out.append(b)

    return " ".join(out)
# ---- Roles ----
ROLE_PATTERNS: List[tuple[str, str]] = [
    ("shareholder", r"\b(minority\s+)?shareholder(s)?\b"),
    ("member", r"\b(llc\s+)?member(s)?\b"),
    ("manager", r"\bmanager(s)?\b"),
    ("director", r"\bdirector(s)?\b"),
    ("officer", r"\bofficer(s)?\b"),
    ("corporation", r"\bcorporation\b|\bcorp\.\b|\binc\.\b|\bcompany\b"),
    ("llc", r"\bllc\b|\blimited\s+liability\s+company\b"),
    ("partnership", r"\bpartnership\b|\blimited\s+partnership\b|\b\bLP\b"),
    ("employee", r"\bemployee(s)?\b"),
    ("employer", r"\bemployer(s)?\b"),
    ("lender", r"\blender(s)?\b|\bbank\b|\bcreditor(s)?\b"),
    ("borrower", r"\bborrower(s)?\b|\bdebtor(s)?\b"),
    ("plaintiff", r"\bplaintiff(s)?\b"),
    ("defendant", r"\bdefendant(s)?\b"),
]
ROLE_RX = [(name, re.compile(pat, re.I)) for name, pat in ROLE_PATTERNS]


# ---- Actions / triggers ----
ACTION_PATTERNS: List[tuple[str, str]] = [
    # books & records style
    ("demanded_access_records",
     r"\bdemand(ed|s)?\b.*\b(books?\s+and\s+records|records|inspection)\b|\binspect(ion)?\b"),
    ("refused_access_records",
     r"\brefus(ed|es|al)\b.*\b(inspect(ion)?|books?\s+and\s+records|records)\b|\bdeni(ed|es)\b.*\binspection\b"),
    ("filed_action",
     r"\bfiled\b.*\b(action|lawsuit|complaint|petition)\b|\bcommenced\b.*\b(action|proceeding)\b|\bbrought\b.*\b(action|claim)\b"),

    # procedural signals (kept, but not treated as “the law”)
    ("motion_to_dismiss", r"\bmotion\s+to\s+dismiss\b|\brule\s*12\b|\b12\(b\)\(?6\)?\b"),
    ("summary_judgment", r"\bsummary\s+judgment\b|\brule\s*56\b"),
    ("prelim_injunction", r"\bpreliminary\s+injunction\b|\btemporary\s+restraining\s+order\b|\bTRO\b|\brule\s*65\b"),

    # allegation-type signals
    ("fraud_alleged", r"\bfraud\b|\bmisrepresent(ation)?\b|\bomission\b|\bconceal(ment)?\b"),
    ("udtpa_alleged", r"\bunfair\s+and\s+deceptive\b|\b75-1\.1\b|\bUDTPA\b"),
    ("trade_secret_alleged", r"\btrade\s+secret(s)?\b|\bmisappropriat"),
    ("noncompete_alleged", r"\bnon[-\s]*compete\b|\brestrictive\s+covenant\b|\bnon[-\s]*solicit\b"),
    ("fiduciary_duty_alleged", r"\bfiduciary\s+dut(y|ies)\b|\bduty\s+of\s+loyalty\b|\bduty\s+of\s+care\b"),
    ("fraudulent_transfer_alleged", r"\bfraudulent\s+transfer\b|\bvoidable\s+transaction\b|\bUVTA\b|\b39-23\.\d\b"),
]
ACTION_RX = [(name, re.compile(pat, re.I | re.S)) for name, pat in ACTION_PATTERNS]


# ---- Remedies / relief sought ----
REMEDY_PATTERNS: List[tuple[str, str]] = [
    ("compel_inspection", r"\bcompel\b.*\binspection\b|\baction\b.*\bto\s+compel\b.*\binspect"),
    ("damages", r"\bdamages\b|\bcompensatory\b|\bpunitive\b"),
    ("injunction", r"\binjunction\b|\bTRO\b|\btemporary\s+restraining\s+order\b"),
    ("declaratory_judgment", r"\bdeclaratory\s+judgment\b|\bdeclare\b.*\bright(s)?\b"),
    ("attorneys_fees", r"\battorney'?s?\s+fees\b|\bfee[-\s]*shifting\b"),
    ("specific_performance", r"\bspecific\s+performance\b"),
]
REMEDY_RX = [(name, re.compile(pat, re.I | re.S)) for name, pat in REMEDY_PATTERNS]


# ---- Caption heuristic (if user includes “X v. Y”) ----
VS_RX = re.compile(r"([A-Z][\w&.,' -]{2,80})\s+v\.?\s+([A-Z][\w&.,' -]{2,80})")


# ---- Procedural posture heuristic ----
POSTURE_PATTERNS: List[tuple[str, str]] = [
    ("pre_filing", r"\bconsidering\b|\bplanning\b|\bthreaten(ed|ing)?\b"),
    ("filed_complaint", r"\bcomplaint\b|\bfiled\b.*\b(action|lawsuit|complaint|petition)\b|\bfiled\s+an?\s+action\b|\bcommenced\b.*\b(action|proceeding)\b"),
    ("motion_stage", r"\bmotion\b|\brule\s*12\b|\bsummary\s+judgment\b|\brule\s*56\b|\bpreliminary\s+injunction\b|\bTRO\b"),
    ("appeal_stage", r"\bappeal\b|\bappellate\b"),
]
POSTURE_RX = [(name, re.compile(pat, re.I | re.S)) for name, pat in POSTURE_PATTERNS]


@dataclass
class Phase1Facts:
    parties: Dict[str, Any]
    roles: List[str]
    actions: List[str]
    dispute_triggers: List[str]
    relief_sought: List[str]
    procedural_posture: str
    key_phrases: List[str]
    warnings: List[str]


def extract_phase1_facts(user_text: str) -> Dict[str, Any]:
    text = norm(user_text)
    warnings: List[str] = []

    # Roles
    roles_found: List[str] = []
    for name, rx in ROLE_RX:
        if rx.search(text):
            roles_found.append(name)

    # Actions
    actions_found: List[str] = []
    for name, rx in ACTION_RX:
        if rx.search(text):
            actions_found.append(name)

    # Remedies
    remedies_found: List[str] = []
    for name, rx in REMEDY_RX:
        if rx.search(text):
            remedies_found.append(name)

    # Dispute triggers subset
    trigger_set = {
        "refused_access_records",
        "fraud_alleged",
        "udtpa_alleged",
        "trade_secret_alleged",
        "noncompete_alleged",
        "fiduciary_duty_alleged",
        "fraudulent_transfer_alleged",
        "motion_to_dismiss",
        "summary_judgment",
    }
    dispute_triggers = [a for a in actions_found if a in trigger_set]

    # Procedural posture (most advanced)
    # Procedural posture (most advanced)
    posture = "unknown"
    detected = []
    for name, rx in POSTURE_RX:
        if rx.search(text):
            detected.append(name)

    # ✅ Strong rule: if we detected "filed_action", posture is at least filed_complaint
    if "filed_action" in actions_found:
        detected.append("filed_complaint")

    precedence = ["appeal_stage", "motion_stage", "filed_complaint", "pre_filing"]
    for p in precedence:
        if p in detected:
            posture = p
            break

    # Parties (caption only if present)
    parties = {"caption": "", "plaintiff": "", "defendant": "", "party_names": []}
    m = VS_RX.search(text)
    if m:
        p1, p2 = norm(m.group(1)), norm(m.group(2))
        parties["caption"] = f"{p1} v. {p2}"
        parties["plaintiff"] = p1
        parties["defendant"] = p2
        parties["party_names"] = [p1, p2]

    # Key phrases (anchors)
    key_phrases = []
    for rx in [
        re.compile(r"\bbooks?\s+and\s+records\b", re.I),
        re.compile(r"\bminority\s+shareholder\b", re.I),
        re.compile(r"\bcompel\b.*\binspection\b", re.I),
        re.compile(r"\bmotion\s+to\s+dismiss\b", re.I),
        re.compile(r"\btrade\s+secret(s)?\b", re.I),
        re.compile(r"\bnon[-\s]*compete\b", re.I),
        re.compile(r"\bfraud(ulent)?\b", re.I),
        re.compile(r"\bunfair\s+and\s+deceptive\b", re.I),
        re.compile(r"\bpersonal\s+jurisdiction\b", re.I),
    ]:
        mm = rx.search(text)
        if mm:
            key_phrases.append(norm(mm.group(0)))

    # Normalize lists
    roles_found = uniq_keep_order(roles_found)
    actions_found = uniq_keep_order(actions_found)
    remedies_found = uniq_keep_order(remedies_found)
    dispute_triggers = uniq_keep_order(dispute_triggers)
    key_phrases = uniq_keep_order(key_phrases)

    # Warnings
    if not roles_found:
        warnings.append("No clear party roles detected (shareholder/member/employee/lender/etc.).")
    if not actions_found:
        warnings.append("No clear actions detected. Add what happened (demand/refusal/transfer/termination).")
    if posture == "unknown":
        warnings.append("Procedural posture unclear (filed? motion stage? appeal?).")

    facts = Phase1Facts(
        parties=parties,
        roles=roles_found,
        actions=actions_found,
        dispute_triggers=dispute_triggers,
        relief_sought=remedies_found,
        procedural_posture=posture,
        key_phrases=key_phrases,
        warnings=warnings,
    )
    return asdict(facts)


#--------------------------
#Phase 2: High Recall
#__________________________
WORD = re.compile(r"[a-z0-9]+", re.I)
def tokenize(text: str) -> List[str]:
    # Conservative tokenizer for recall
    return WORD.findall((text or "").lower())

def add_ngrams(tokens: List[str], n_min: int = 1, n_max: int = 3) -> List[str]:
    out = []
    L = len(tokens)
    for n in range(n_min, n_max + 1):
        if n == 1:
            out.extend(tokens)
            continue
        for i in range(L - n + 1):
            out.append("_".join(tokens[i:i+n]))
    return out

@dataclass
class Doc:
    doc_id: str
    text: str
    tokens: List[str]
    title: str = ""
    source_file: str = ""

@dataclass
class Hit:
    doc_id: str
    score: float
    method: str          # "bm25" or "tfidf"
    snippet: str

CASE_TITLE_RX = re.compile(r"([A-Z][\w&.,' -]{2,80}\s+v\.?\s+[A-Z][\w&.,' -]{2,80})")

def guess_case_title(raw_text: str, fallback: str) -> str:
    t = (raw_text or "").strip()
    if not t:
        return fallback
    m = CASE_TITLE_RX.search(t[:1500])  # look near the top
    if m:
        return norm(m.group(1))
    # fallback: sometimes opinions have "X v. Y, 2018 NCBC 27." style on first line
    first_line = norm(t.splitlines()[0]) if t else ""
    return first_line[:120] if first_line else fallback

def load_bronze_docs(bronze_dir: str, max_chars: int = 750_000) -> List[Doc]:
    docs = []
    for fn in os.listdir(bronze_dir):
        if not fn.lower().endswith(".json"):
            continue

        path = os.path.join(bronze_dir, fn)
        try:
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
        except Exception:
            continue

        txt = obj.get("raw_text", "")
        if not txt.strip():
            continue

        txt = txt[:max_chars]
        toks = add_ngrams(tokenize(txt), 1, 2)  # 1–2 grams (massively smaller)
        docs.append(Doc(
            doc_id=fn,
            text=txt,
            tokens=toks,
            title=guess_case_title(txt, fn),
            source_file=path,  # ✅ record where it came from
        ))
    return docs

# -------- BM25 (primary) --------

@dataclass
class BM25Index:
    docs: List[Doc]
    df: Dict[str, int]
    doc_len: List[int]
    avgdl: float
    tf: List[Dict[str, int]]
    k1: float = 1.5
    b: float = 0.75

def build_bm25_index(docs: List[Doc], k1: float = 1.5, b: float = 0.75) -> BM25Index:
    df: Dict[str, int] = {}
    tf: List[Dict[str, int]] = []
    doc_len: List[int] = []

    for d in docs:
        counts: Dict[str, int] = {}
        for t in d.tokens:
            counts[t] = counts.get(t, 0) + 1
        tf.append(counts)
        doc_len.append(len(d.tokens))

        # DF: count each term once per doc
        for t in counts.keys():
            df[t] = df.get(t, 0) + 1

    avgdl = (sum(doc_len) / max(1, len(doc_len))) if doc_len else 0.0
    return BM25Index(docs=docs, df=df, doc_len=doc_len, avgdl=avgdl, tf=tf, k1=k1, b=b)

def bm25_idf(N: int, df_t: int) -> float:
    # standard BM25 with +0.5 smoothing
    return math.log(1.0 + (N - df_t + 0.5) / (df_t + 0.5))

# -------- Snippet helper (must be defined before bm25_search) --------

def make_snippet(text: str, query: str, window: int = 260) -> str:
    """
    Find first match of any strong query token and return a nearby snippet.
    """
    if not text:
        return ""
    q_tokens = uniq_keep_order(tokenize(query))[:12]  # a few anchors
    best_pos: Optional[int] = None
    lower = text.lower()
    for t in q_tokens:
        pos = lower.find(t.lower())
        if pos != -1:
            best_pos = pos
            break
    if best_pos is None:
        # fallback: first chunk
        return norm(text[:window]) + ("…" if len(text) > window else "")
    start = max(0, best_pos - window // 2)
    end = min(len(text), start + window)
    return "…" + norm(text[start:end]) + ("…" if end < len(text) else "")

def bm25_search(index: BM25Index, query: str, top_k: int = 100) -> List[Hit]:
    q_tokens = add_ngrams(tokenize(query), 1, 2)
    if not q_tokens or not index.docs:
        return []

    N = len(index.docs)
    hits: List[Hit] = []

    for i, d in enumerate(index.docs):
        score = 0.0
        dl = index.doc_len[i]
        denom_base = index.k1 * (1.0 - index.b + index.b * (dl / (index.avgdl or 1.0)))

        tf_i = index.tf[i]
        for t in q_tokens:
            f = tf_i.get(t, 0)
            if f == 0:
                continue
            idf = bm25_idf(N, index.df.get(t, 0))
            score += idf * (f * (index.k1 + 1.0)) / (f + denom_base)

        if score > 0:
            hits.append(Hit(
                doc_id=d.doc_id,
                score=score,
                method="bm25",
                snippet=make_snippet(d.text, query)
            ))

    hits.sort(key=lambda h: h.score, reverse=True)
    return hits[:top_k]

# -------- TF-IDF fallback (secondary) --------
# Lightweight TF-IDF (no sklearn), cosine on sparse dicts

def build_tfidf(docs: List[Doc]) -> Tuple[List[Dict[str, float]], Dict[str, float]]:
    """
    Returns:
      doc_vecs: list of tf-idf sparse vectors
      idf: term -> idf
    """
    N = len(docs)
    if N == 0:
        return [], {}

    # DF
    df: Dict[str, int] = {}
    for d in docs:
        seen = set(d.tokens)
        for t in seen:
            df[t] = df.get(t, 0) + 1

    idf: Dict[str, float] = {}
    for t, df_t in df.items():
        idf[t] = math.log((N + 1.0) / (df_t + 1.0)) + 1.0

    doc_vecs: List[Dict[str, float]] = []
    for d in docs:
        counts: Dict[str, int] = {}
        for t in d.tokens:
            counts[t] = counts.get(t, 0) + 1
        # tf-idf
        vec: Dict[str, float] = {}
        for t, c in counts.items():
            vec[t] = (1.0 + math.log(c)) * idf.get(t, 0.0)
        # L2 normalize
        norm2 = math.sqrt(sum(v*v for v in vec.values())) or 1.0
        for t in list(vec.keys()):
            vec[t] /= norm2
        doc_vecs.append(vec)

    return doc_vecs, idf
@st.cache_resource(show_spinner=False)
def get_phase2_assets(bronze_dir: str):
    """
    Load bronze docs once and build retrieval artifacts once per Streamlit process.
    Returns:
      docs_by_id: doc_id -> Doc (contains text + tokens)
      bm25_idx: BM25Index
      tfidf_pack: (doc_vecs, idf)
    """
    docs = load_bronze_docs(bronze_dir)
    docs_by_id = {d.doc_id: d for d in docs}

    bm25_idx = build_bm25_index(docs)

    doc_vecs, idf = build_tfidf(docs)
    tfidf_pack = (doc_vecs, idf)

    return docs_by_id, bm25_idx, tfidf_pack

def tfidf_query_vec(query: str, idf: Dict[str, float]) -> Dict[str, float]:
    qt = add_ngrams(tokenize(query), 1, 3)
    counts: Dict[str, int] = {}
    for t in qt:
        counts[t] = counts.get(t, 0) + 1
    vec: Dict[str, float] = {}
    for t, c in counts.items():
        if t in idf:
            vec[t] = (1.0 + math.log(c)) * idf[t]
    norm2 = math.sqrt(sum(v*v for v in vec.values())) or 1.0
    for t in list(vec.keys()):
        vec[t] /= norm2
    return vec

def cosine_sparse(a: Dict[str, float], b: Dict[str, float]) -> float:
    # iterate smaller dict
    if len(a) > len(b):
        a, b = b, a
    return sum(v * b.get(t, 0.0) for t, v in a.items())

def tfidf_search(docs: List[Doc], doc_vecs: List[Dict[str, float]], idf: Dict[str, float],
                 query: str, top_k: int = 100) -> List[Hit]:
    qv = tfidf_query_vec(query, idf)
    if not qv:
        return []

    hits: List[Hit] = []
    for d, dv in zip(docs, doc_vecs):
        s = cosine_sparse(qv, dv)
        if s > 0:
            hits.append(Hit(
                doc_id=d.doc_id,
                score=s,
                method="tfidf",
                snippet=make_snippet(d.text, query)
            ))
    hits.sort(key=lambda h: h.score, reverse=True)
    return hits[:top_k]

# -------- Snippet helper --------

# -------- Combined search --------

def phase2_retrieve(bronze_dir: str, query: str, top_k: int = 150) -> List[Hit]:
    """
    High-recall strategy (cached):
      1) BM25 primary
      2) If BM25 yields too few, TF-IDF fallback
    """
    if not query.strip():
        return []

    docs_by_id, bm25_idx, (doc_vecs, idf) = get_phase2_assets(bronze_dir)
    docs = bm25_idx.docs  # same docs list used to build bm25

    # BM25
    bm25_hits = bm25_search(bm25_idx, query, top_k=top_k)

    if len(bm25_hits) >= min(50, top_k):
        return bm25_hits

    # TF-IDF backup
    tfidf_hits = tfidf_search(docs, doc_vecs, idf, query, top_k=top_k)

    # Merge (prefer BM25 order; fill from TF-IDF)
    seen = set(h.doc_id for h in bm25_hits)
    out = bm25_hits[:]
    for h in tfidf_hits:
        if h.doc_id not in seen:
            out.append(h)
            seen.add(h.doc_id)
        if len(out) >= top_k:
            break
    return out

# -------------------------
# Phase 3 — Structured Re-Ranking (LLM w/ evidence only)
# From many candidates -> a few truly similar ones.
# -------------------------
import time
import requests

STATUTE_RX = re.compile(
    r"""
    (?:N\.?\s*C\.?\s*Gen\.?\s*Stat\.?|N\.?\s*C\.?\s*G\.?\s*S\.?)   # NC Gen Stat / NCGS variants
    \s*§+\s*
    [0-9A-Za-z\-\.()]+                                            # section like 55-16-01(b)
    """,
    re.IGNORECASE | re.VERBOSE,
)

OUTCOME_CUES = [
    "DENIES", "GRANTS", "ALLOWS", "DISMISSES", "DISMISSED", "DENIED", "GRANTED",
    "WITHOUT PREJUDICE", "WITH PREJUDICE", "COMPELS", "ORDERED", "SO ORDERED",
]

POSTURE_CUES_RX = re.compile(
    r"\b(motion\s+to\s+\w+|summary\s+judgment|rule\s*12|12\(b\)\(?6\)?|rule\s*56|"
    r"preliminary\s+injunction|temporary\s+restraining\s+order|TRO|"
    r"action\s+to\s+compel|compel\s+inspection|inspection\s+demand|books?\s+and\s+records)\b",
    re.I
)

def extract_statutes_from_excerpt(excerpt: str) -> List[str]:
    if not excerpt:
        return []
    found = [norm(m.group(0)) for m in STATUTE_RX.finditer(excerpt)]
    return uniq_keep_order(found)

def normalize_outcome(label: str, excerpt: str) -> str:
    if not label or not isinstance(label, str):
        return extract_outcome_from_excerpt(excerpt)

    s = label.strip().lower()

    # map common phrases
    if "grant" in s and "deny" in s:
        return "granted_in_part_denied_in_part"
    if "deny" in s:
        return "denied"
    if "grant" in s:
        return "granted"
    if "dismiss" in s:
        return "dismissed"
    if "without prejudice" in s:
        return "without_prejudice"
    if "with prejudice" in s:
        return "with_prejudice"
    if "compel" in s:
        return "compelled"
    if "order" in s:
        return "ordered"

    return extract_outcome_from_excerpt(excerpt)


def extract_outcome_from_excerpt(excerpt: str) -> str:
    """
    Deterministic, conservative outcome guess FROM EXCERPT ONLY.
    Used as fallback if LLM output is empty or malformed.
    """
    if not excerpt:
        return "unknown"
    up = excerpt.upper()
    for cue in OUTCOME_CUES:
        if cue in up:
            # Return a short label rather than long text
            if "DENY" in cue:
                return "denied"
            if "GRANT" in cue:
                return "granted"
            if "DISMISS" in cue:
                return "dismissed"
            if "WITHOUT PREJUDICE" in cue:
                return "without_prejudice"
            if "WITH PREJUDICE" in cue:
                return "with_prejudice"
            if "COMPEL" in cue:
                return "compelled"
            if "ORDER" in cue:
                return "ordered"
    return "unknown"

def make_evidence_excerpt(raw_text: str, max_chars: int = 2400) -> str:
    """
    Build a short evidence excerpt:
      - intro (front of doc)
      - posture/issue neighborhood (first posture cue found)
      - holding-ish neighborhood (near "ORDER"/"CONCLUSION"/"DENIES/GRANTS" etc.)
    """
    t = raw_text or ""
    t = t.replace("\x00", " ")
    t = norm(t)
    if not t:
        return ""

    # 1) Intro chunk
    intro = t[:900]

    # 2) Posture neighborhood (first cue match)
    posture_chunk = ""
    m = POSTURE_CUES_RX.search(t)
    if m:
        start = max(0, m.start() - 400)
        end = min(len(t), m.end() + 600)
        posture_chunk = t[start:end]

    # 3) Holding neighborhood: search for strong holding words
    hold_chunk = ""
    hold_m = re.search(r"\b(CONCLUSION|SO ORDERED|ORDER|IT IS ORDERED|DENIES|GRANTS|DISMISSES)\b", t, re.I)
    if hold_m:
        start = max(0, hold_m.start() - 500)
        end = min(len(t), hold_m.start() + 1000)
        hold_chunk = t[start:end]

    # Combine, dedup-ish
    parts = [intro]
    if posture_chunk and posture_chunk not in intro:
        parts.append(posture_chunk)
    if hold_chunk and hold_chunk not in intro and hold_chunk not in posture_chunk:
        parts.append(hold_chunk)

    excerpt = "\n\n---\n\n".join(parts)
    excerpt = excerpt[:max_chars]
    return excerpt

def llm_rerank_one(
    *,
    base_url: str,
    api_key: str,
    model: str,
    user_scenario: str,
    phase1_facts: Dict[str, Any],
    candidate_doc_id: str,
    candidate_source_file: str,
    excerpt: str,
    timeout: int = 45,
) -> Dict[str, Any]:
    """
    OpenAI-compatible Chat Completions call:
      POST {base_url}/chat/completions
    """
    rubric = {
        "factual_similarity": "Is this about the same kind of dispute as the user scenario?",
        "legal_mechanism": "Same type of claim/remedy mechanism (e.g., action to compel inspection / books & records demand/refusal)?",
        "procedural_posture": "Similar stage (pre-filing vs filed vs motion stage)?",
        "outcome_relevance": "Does the holding in the excerpt help answer the user's situation?"
    }

    system = (
        "You are re-ranking legal candidates using ONLY the provided excerpt. "
        "No outside knowledge. No assumptions. If it is not in the excerpt, you must say unknown.\n\n"
        "Hard constraint: You may ONLY list statutes that appear verbatim in the excerpt.\n"
        "Hard constraint: Outcome must be derived ONLY from the excerpt text.\n"
        "Return STRICT JSON only (no markdown)."
    )

    user = {
        "task": "Phase 3 structured rerank (evidence-only).",
        "user_scenario": user_scenario,
        "phase1_facts": phase1_facts,
        "candidate": {
            "doc_id": candidate_doc_id,
            "source_file": candidate_source_file,
            "excerpt": excerpt
        },
        "scoring_rubric": rubric,
        "required_output_json_schema": {
            "doc_id": "string",
            "relevance_score": "number (0 to 1)",
            "why_relevant": "string (one sentence)",
            "extracted_statutes": "array of strings (ONLY statutes appearing in excerpt; otherwise [])",
            "outcome": "string (from excerpt only; e.g., granted/denied/dismissed/unknown)",
            "rubric_scores": {
                "factual_similarity": "number (0 to 1)",
                "legal_mechanism": "number (0 to 1)",
                "procedural_posture": "number (0 to 1)",
                "outcome_relevance": "number (0 to 1)"
            }
        },
        "instructions": [
            "If the excerpt does not mention a statute explicitly, extracted_statutes must be [].",
            "If the excerpt does not clearly state an outcome, outcome must be 'unknown'.",
            "why_relevant must be one sentence, grounded in excerpt + scenario only.",
        ]
    }

    payload = {
        "model": model,
        "temperature": 0.0,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
        ],
        "response_format": {"type": "json_object"},
    }

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    u = base_url.rstrip("/")

    # Accept either ".../v1" OR full ".../v1/chat/completions"
    if u.endswith("/chat/completions"):
        url = u
    else:
        url = u + "/chat/completions"

    r = requests.post(url, headers=headers, json=payload, timeout=timeout)

    # Surface server error text to UI (instead of silent HTTPError)
    try:
        r.raise_for_status()
    except requests.HTTPError as e:
        raise RuntimeError(f"LLM HTTP {r.status_code} from {url}: {r.text[:800]}") from e

    data = r.json()
    content = data["choices"][0]["message"]["content"]
    return json.loads(content)

def extract_statutes_from_fulltext(full_text: str) -> List[str]:
    if not full_text:
        return []
    found = [norm(m.group(0)) for m in STATUTE_RX.finditer(full_text)]
    return uniq_keep_order(found)
def phase3_rerank(
    *,
    hits: List[Hit],
    docs_by_id: Dict[str, Doc],
    user_scenario: str,
    phase1_facts: Dict[str, Any],
    top_n: int = 20,
    per_case_chars: int = 2400,
    base_url: str = "",
    api_key: str = "",
    model: str = "",
    max_candidates: int = 150,
    sleep_s: float = 0.0,
    _ui_progress=None,
) -> List[Dict[str, Any]]:
    """
    Rerank top candidates from Phase 2.
    - Only reranks first max_candidates for cost control.
    - Returns list of dict results sorted by relevance_score desc.
    """
    results: List[Dict[str, Any]] = []
    shortlist = hits[:max_candidates]
    total = max(1, len(shortlist))

    for i, h in enumerate(shortlist, start=1):
        if _ui_progress:
            prog, status = _ui_progress
            status.write(f"LLM reranking {i}/{total}: {h.doc_id}")
            prog.progress(i / total)
        d = docs_by_id.get(h.doc_id)
        if not d:
            continue

        excerpt = make_evidence_excerpt(d.text, max_chars=per_case_chars)
        statutes_excerpt = extract_statutes_from_excerpt(excerpt)
        statutes_fulltext = extract_statutes_from_fulltext(d.text)
        if not excerpt.strip():
            continue

        # LLM call
        try:
            out = llm_rerank_one(
                base_url=base_url,
                api_key=api_key,
                model=model,
                user_scenario=user_scenario,
                phase1_facts=phase1_facts,
                candidate_doc_id=h.doc_id,
                candidate_source_file=getattr(d, "source_file", ""),
                excerpt=excerpt,
            )
        except Exception as e:
            # Fallback: still produce a structured record without LLM
            out = {
                "doc_id": h.doc_id,
                "relevance_score": 0.0,
                "why_relevant": f"LLM error; fallback used ({type(e).__name__}): {str(e)[:220]}",
                "extracted_statutes": extract_statutes_from_excerpt(excerpt),  # <-- show what is actually in excerpt
                "outcome": extract_outcome_from_excerpt(excerpt),
                "rubric_scores": {
                    "factual_similarity": 0.0,
                    "legal_mechanism": 0.0,
                    "procedural_posture": 0.0,
                    "outcome_relevance": 0.0,
                }
            }

        # -------- Guardrails: enforce statutes-from-excerpt ONLY --------
        # Statutes actually present in the excerpt
        excerpt_list = extract_statutes_from_excerpt(excerpt)  # list, de-duped, in order
        excerpt_set = set(excerpt_list)  # for membership test

        # Statutes the LLM claimed (may be empty / hallucinated)
        llm_statutes = out.get("extracted_statutes", []) or []
        llm_statutes = [norm(s) for s in llm_statutes if isinstance(s, str)]

        # HARD FILTER: keep only what truly appears in the excerpt
        llm_statutes = [s for s in llm_statutes if s in excerpt_set]
        llm_statutes = uniq_keep_order(llm_statutes)

        # ✅ If LLM gave nothing, still show what is in the excerpt (truthful + useful)
        out["extracted_statutes"] = uniq_keep_order(llm_statutes)  # excerpt-only after hard-filter
        # Enforce basic fields
        out["doc_id"] = h.doc_id
        out["case_title"] = getattr(d, "title", "") or h.doc_id
        out["source_file"] = getattr(d, "source_file", "")
        out["phase2_method"] = h.method
        out["phase2_score"] = h.score
        out["excerpt"] = excerpt  # keep for UI transparency
        out["outcome"] = normalize_outcome(out.get("outcome", ""), excerpt)
        out["statutes_excerpt"] = statutes_excerpt
        out["statutes_fulltext"] = statutes_fulltext

        # Clamp relevance_score
        try:
            rs = float(out.get("relevance_score", 0.0))
        except Exception:
            rs = 0.0
        out["relevance_score"] = max(0.0, min(1.0, rs))

        # If outcome missing, fallback from excerpt
        if not out.get("outcome") or not isinstance(out["outcome"], str):
            out["outcome"] = extract_outcome_from_excerpt(excerpt)

        results.append(out)

        if sleep_s:
            time.sleep(sleep_s)
    if _ui_progress:
        prog, status = _ui_progress
        prog.progress(1.0)
        status.write("LLM reranking complete.")

    results.sort(key=lambda x: x.get("relevance_score", 0.0), reverse=True)
    return results[:top_n]


def build_phase4_evidence_pack(
    reranked: List[Dict[str, Any]],
    max_cases: int = 8,
    max_total_chars: int = 18_000,
) -> Dict[str, Any]:
    """
    Build a compact evidence pack for Phase 4 from Phase 3 outputs.
    Evidence-only: include excerpts + excerpt-statutes + outcomes.
    """
    pack = []
    total = 0

    for r in reranked[:max_cases]:
        excerpt = (r.get("excerpt") or "").strip()
        if not excerpt:
            continue

        item = {
            "doc_id": r.get("doc_id", ""),
            "case_title": r.get("case_title", r.get("doc_id", "")),
            "relevance_score": r.get("relevance_score", 0.0),
            "outcome": r.get("outcome", "unknown"),
            "phase2_method": r.get("phase2_method", ""),
            "phase2_score": r.get("phase2_score", 0.0),
            # Evidence
            "statutes_excerpt": r.get("statutes_excerpt") or [],
            "excerpt": excerpt,
        }

        item_chars = len(item["excerpt"])
        if total + item_chars > max_total_chars:
            break

        pack.append(item)
        total += item_chars

    # Also provide a deduped statute list (excerpt-only, truthy)
    statutes = []
    for it in pack:
        statutes += it.get("statutes_excerpt", [])
    statutes = uniq_keep_order(statutes)

    return {
        "cases_used": pack,
        "statutes_excerpt_deduped": statutes,
        "limits": {
            "max_cases": max_cases,
            "max_total_chars": max_total_chars,
            "used_cases": len(pack),
            "used_chars": total,
        }
    }

def llm_phase4_arguments(
    *,
    base_url: str,
    api_key: str,
    model: str,
    user_scenario: str,
    phase1_facts: Dict[str, Any],
    evidence_pack: Dict[str, Any],
    timeout: int = 60,
) -> Dict[str, Any]:
    """
    Produce pro-user arguments using ONLY evidence_pack excerpts.
    """
    system = (
        "You are drafting the strongest arguments IN FAVOR of the user based ONLY on the provided evidence excerpts.\n"
        "No outside knowledge. No assumptions. If something is not supported by the excerpts, say 'unknown' or omit.\n"
        "Hard constraint: Statutes listed must appear in evidence_pack.statutes_excerpt_deduped.\n"
        "Hard constraint: Every argument must cite at least one supporting case by doc_id and a short quote (<=25 words).\n"
        "Return STRICT JSON only (no markdown)."
    )

    required_schema = {
        "overall_theory_of_case": "string (2-5 sentences, evidence-grounded)",
        "best_arguments": [
            {
                "argument_title": "string",
                "argument": "string (tight, persuasive, grounded)",
                "elements_or_prongs_if_any": "array of strings (or [])",
                "support": [
                    {
                        "doc_id": "string",
                        "case_title": "string",
                        "supporting_quote": "string (<=25 words, verbatim from excerpt)",
                        "why_supports": "string (one sentence)"
                    }
                ]
            }
        ],
        "statutes_relevant_excerpt_only": "array of strings (subset of evidence_pack.statutes_excerpt_deduped)",
        "key_counterarguments_and_responses": [
            {
                "counterargument": "string",
                "response": "string (must cite at least one case in support)"
            }
        ],
        "missing_info_to_strengthen": "array of strings",
        "safety_note": "string (brief: not legal advice)"
    }

    user_obj = {
        "task": "Phase 4 argument builder (pro-user), evidence-only.",
        "user_scenario": user_scenario,
        "phase1_facts": phase1_facts,
        "evidence_pack": evidence_pack,
        "required_output_json_schema": required_schema,
        "instructions": [
            "Use only evidence_pack.cases_used[*].excerpt.",
            "Do NOT invent holdings, standards, or statutes.",
            "Each argument must include >=1 support citation with doc_id and a short verbatim quote (<=25 words).",
            "statutes_relevant_excerpt_only must be drawn ONLY from evidence_pack.statutes_excerpt_deduped.",
            "If evidence is thin, say so and focus on what can be defended."
        ],
    }

    payload = {
        "model": model,
        "temperature": 0.1,  # slight creativity but still controlled
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user_obj, ensure_ascii=False)},
        ],
        "response_format": {"type": "json_object"},
    }

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    u = base_url.rstrip("/")
    url = u if u.endswith("/chat/completions") else (u + "/chat/completions")

    r = requests.post(url, headers=headers, json=payload, timeout=timeout)
    try:
        r.raise_for_status()
    except requests.HTTPError as e:
        raise RuntimeError(f"Phase4 LLM HTTP {r.status_code} from {url}: {r.text[:800]}") from e

    data = r.json()
    content = data["choices"][0]["message"]["content"]
    return json.loads(content)

def phase4_make_arguments(
    *,
    reranked: List[Dict[str, Any]],
    user_scenario: str,
    phase1_facts: Dict[str, Any],
    base_url: str,
    api_key: str,
    model: str,
    max_cases: int = 8,
    max_total_chars: int = 18_000,
) -> Dict[str, Any]:
    evidence_pack = build_phase4_evidence_pack(
        reranked=reranked,
        max_cases=max_cases,
        max_total_chars=max_total_chars,
    )

    out = llm_phase4_arguments(
        base_url=base_url,
        api_key=api_key,
        model=model,
        user_scenario=user_scenario,
        phase1_facts=phase1_facts,
        evidence_pack=evidence_pack,
    )

    # HARD FILTER statutes to excerpt-only set
    allowed = set(evidence_pack.get("statutes_excerpt_deduped", []))
    s = out.get("statutes_relevant_excerpt_only", []) or []
    s = [norm(x) for x in s if isinstance(x, str)]
    out["statutes_relevant_excerpt_only"] = uniq_keep_order([x for x in s if x in allowed])

    # Attach evidence pack for transparency/debug
    out["_evidence_pack_meta"] = {
        "limits": evidence_pack.get("limits", {}),
        "cases_used": [
            {
                "doc_id": c.get("doc_id",""),
                "case_title": c.get("case_title",""),
                "outcome": c.get("outcome","unknown"),
                "relevance_score": c.get("relevance_score", 0.0),
                "statutes_excerpt": c.get("statutes_excerpt", []),
            }
            for c in evidence_pack.get("cases_used", [])
        ],
        "statutes_excerpt_deduped": evidence_pack.get("statutes_excerpt_deduped", []),
    }

    return out

# -------------------------
# Streamlit UI (Display only)
# -------------------------

st.set_page_config(page_title="NCBC LegalBot — Phase 1", layout="centered")
st.title("NCBC LegalBot — Phase 1 (Facts Extraction Only)")
st.info("Phase 1 runs locally with regex only — no LLM calls.")
st.caption("Extracts roles, actions, triggers, posture, and requested relief from a user scenario. No legal conclusions.")
with st.expander("Tips for best results (what to include)", expanded=True):
    st.markdown(
        """
Include short details in this order (bullets are fine):

- **Parties + roles:** e.g., *minority shareholder*, *corporation*, *LLC member*, *employee*, *lender*
- **Object of the dispute:** e.g., *corporate books and records*, *operating agreement*, *trade secrets*
- **Action taken:** e.g., *demanded inspection*, *requested access*, *sent written demand*
- **Response/refusal + why:** e.g., *refused*, *denied*, *claimed improper purpose*
- **What you did next:** e.g., *filed an action*, *moved to compel inspection*, *sought injunction*
- **What relief you want:** e.g., *compel inspection*, *injunction*, *damages*

**Example (books & records):**
> “A minority shareholder demanded access to corporate books and records for a proper purpose.  
> The corporation refused, saying the request was improper.  
> The shareholder filed an action to compel inspection.”
        """
    )


query = st.text_area(
    "Describe your scenario (plain English)",
    placeholder="e.g., A minority shareholder demanded access to corporate books and records. The company refused. The shareholder filed an action to compel inspection.",
    height=160
)

col1, col2 = st.columns(2)
with col1:
    run = st.button("Extract facts")
with col2:
    auto = st.checkbox("Auto-run on typing", value=True)

if (auto and query.strip()) or (run and query.strip()):
    facts = extract_phase1_facts(query)
    st.session_state["phase1_facts"] = facts
    st.session_state["user_query"] = query

    st.subheader("Extracted facts (Phase 1)")
    with st.expander("Show raw extracted JSON", expanded=False):
        st.json(facts)

    # 🔹 Show the retrieval query that will be used in Phase 2
    retrieval_query = build_retrieval_query(facts)

    st.markdown("---")
    st.subheader("Phase 2 — High-Recall Retrieval (BM25 primary)")

    top_k = st.slider("How many candidate cases?", 50, 200, 150, step=10)

    if st.button("Run Phase 2 retrieval"):
        st.session_state["phase2_hits"] = phase2_retrieve(bronze_dir, retrieval_query, top_k=top_k)
        st.session_state["bronze_dir"] = bronze_dir

    # Always read hits from session_state (so it survives reruns)
    hits = st.session_state.get("phase2_hits", [])

    if not hits:
        st.warning(
            "No hits yet. Click 'Run Phase 2 retrieval'. (Also check the bronze folder and that it contains .json files.)")
    else:
        st.success(f"Retrieved {len(hits)} candidates (high recall). No statute filtering applied.")
        # Build a quick title map so we can show "X v. Y" instead of the JSON filename
        docs_by_id_titles, _, _ = get_phase2_assets(bronze_dir)

        for i, h in enumerate(hits[:3], 1):
            d0 = docs_by_id_titles.get(h.doc_id)
            case_title = (d0.title if d0 and d0.title else h.doc_id)
            with st.expander(f"{i}. {case_title} — {h.method} score={h.score:.4f}", expanded=(i <= 3)):
                st.write(h.snippet)

        if len(hits) > 3:
            st.caption(f"(Showing top 3 of {len(hits)}. Full list is still used for Phase 3.)")


    st.subheader("Phase 2 retrieval query (debug view)")
    st.code(retrieval_query, language="text")

    st.caption(
        "This string is what the search engine will use to find similar cases "
        "(BM25 / TF-IDF). It is derived only from factual anchors — not statutes."
    )

    if facts.get("warnings"):
        st.subheader("Warnings")
        for w in facts["warnings"]:
            st.warning(w)
    st.markdown("---")
    st.subheader("Phase 3 — Structured Re-Ranking (LLM, evidence-only)")

    # Build doc map for fast lookup (Doc objects from loader)
    # If you already have docs list from Phase 2, reuse it.
    # Otherwise reload once:
    docs_by_id, _, _ = get_phase2_assets(st.session_state.get("bronze_dir", bronze_dir))
    # LLM config (OpenAI-compatible endpoint)
    st.caption("LLM sees ONLY the excerpt per candidate. No outside knowledge allowed.")

    base_url = DUKE_LLM_BASE_URL
    api_key = DUKE_LLM_API_KEY
    model = DUKE_LLM_MODEL

    st.caption(f"Using model: {model}")
    st.caption(f"Using base URL: {base_url}")

    max_candidates = st.slider("How many Phase 2 candidates to rerank (cost control)?", 10, 200, 30, step=10)
    top_n = st.slider("How many final cases to show?", 5, 50, 20, step=5)

    if st.button("Run Phase 3 rerank"):
        if not hits:
            st.error("No Phase 2 candidates found. Run Phase 2 retrieval first.")
        elif not base_url.strip():
            st.error("Missing LLM base URL. Set DUKE_LLM_BASE_URL in your environment (or hardcode it in app.py).")
        elif REQUIRE_LLM_KEY and not api_key.strip():
            st.error("Missing LLM API key. Set DUKE_LLM_API_KEY in your environment.")
        else:
            prog = st.progress(0)
            status = st.empty()

            with st.spinner("Phase 3: calling LLM to rerank..."):
                reranked = phase3_rerank(
                    hits=hits,
                    docs_by_id=docs_by_id,
                    user_scenario=st.session_state.get("user_query", query),
                    phase1_facts=st.session_state.get("phase1_facts", facts),
                    top_n=top_n,
                    per_case_chars=2400,
                    base_url=base_url,
                    api_key=api_key,
                    model=model,
                    max_candidates=max_candidates,
                    sleep_s=0.0,
                    _ui_progress=(prog, status),
                )
                st.session_state["phase3_reranked"] = reranked
            st.success(f"Phase 3 complete. Showing top {len(reranked)} reranked cases.")

            for i, r in enumerate(reranked, 1):
                title = f"{i}. {r.get('case_title', r['doc_id'])} — relevance={r['relevance_score']:.2f} | outcome={r.get('outcome', 'unknown')}"
                with st.expander(title, expanded=(i <= 5)):
                    st.write("**Why relevant:** " + (r.get("why_relevant") or ""))
                    st.write("**Statutes (excerpt only):** " + (", ".join(r.get("statutes_excerpt") or []) or "[]"))
                    st.write("**Statutes (full opinion scan):** " + (", ".join(r.get("statutes_fulltext") or []) or "[]"))
                    st.write(f"**Phase 2:** {r.get('phase2_method')} score={r.get('phase2_score'):.4f}")
                    st.text_area("Evidence excerpt (what the LLM saw)", r.get("excerpt", ""), height=220)

    # ✅ Always show Phase 3 results if present (survives Streamlit reruns)
    reranked_show = st.session_state.get("phase3_reranked", [])
    if reranked_show:
        st.success(f"Phase 3 results available: {len(reranked_show)} cases.")
        for i, r in enumerate(reranked_show, 1):
            title = f"{i}. {r.get('case_title', r['doc_id'])} — relevance={r['relevance_score']:.2f} | outcome={r.get('outcome', 'unknown')}"
            with st.expander(title, expanded=(i <= 5)):
                st.write("**Why relevant:** " + (r.get("why_relevant") or ""))
                st.write("**Statutes (excerpt only):** " + (", ".join(r.get("statutes_excerpt") or []) or "[]"))
                st.write("**Statutes (full opinion scan):** " + (", ".join(r.get("statutes_fulltext") or []) or "[]"))
                st.write(f"**Phase 2:** {r.get('phase2_method')} score={r.get('phase2_score'):.4f}")
                st.text_area("Evidence excerpt (what the LLM saw)", r.get("excerpt", ""), height=220)

    st.markdown("---")
    st.subheader("Phase 4 — Build Pro-User Arguments (LLM, evidence-only)")

    reranked_ss = st.session_state.get("phase3_reranked", [])
    if not reranked_ss:
        st.info("Run Phase 3 first (rerank) to generate the evidence set for Phase 4.")
    else:
        max_cases = st.slider("Max cases to use as support (cost control)", 3, 12, 8, step=1)
        max_chars = st.slider("Max total excerpt characters to send", 6000, 30000, 18000, step=1000)

        if st.button("Run Phase 4 argument builder"):
            if not base_url.strip():
                st.error("Missing LLM base URL.")
            elif REQUIRE_LLM_KEY and not api_key.strip():
                st.error("Missing LLM API key.")
            else:
                with st.spinner("Phase 4: building arguments from evidence..."):
                    try:
                        memo = phase4_make_arguments(
                            reranked=reranked_ss,
                            user_scenario=st.session_state.get("user_query", query),
                            phase1_facts=st.session_state.get("phase1_facts", facts),
                            base_url=base_url,
                            api_key=api_key,
                            model=model,
                            max_cases=max_cases,
                            max_total_chars=max_chars,
                        )
                        st.session_state["phase4_memo"] = memo
                    except Exception as e:
                        st.error(f"Phase 4 failed: {type(e).__name__}: {str(e)[:400]}")
                        st.stop()

        memo = st.session_state.get("phase4_memo")
        if memo:
            st.success("Phase 4 memo generated.")

            st.subheader("Overall theory (evidence-grounded)")
            st.write(memo.get("overall_theory_of_case", ""))

            st.subheader("Best arguments in favor")
            for i, a in enumerate(memo.get("best_arguments", []) or [], 1):
                with st.expander(f"{i}. {a.get('argument_title', '(untitled)')}", expanded=(i <= 3)):
                    st.write(a.get("argument", ""))
                    prongs = a.get("elements_or_prongs_if_any", []) or []
                    if prongs:
                        st.write("**Elements / prongs (if any):**")
                        for p in prongs:
                            st.write(f"- {p}")

                    st.write("**Support (evidence quotes):**")
                    for s in a.get("support", []) or []:
                        st.write(f"- **{s.get('case_title', '')}** ({s.get('doc_id', '')})")
                        st.write(f"  > {s.get('supporting_quote', '')}")
                        st.write(f"  {s.get('why_supports', '')}")

            st.subheader("Relevant statutes (excerpt-only)")
            st.write(", ".join(memo.get("statutes_relevant_excerpt_only", []) or []) or "[]")

            st.subheader("Counterarguments and responses")
            for cr in memo.get("key_counterarguments_and_responses", []) or []:
                st.write(f"- **Counter:** {cr.get('counterargument', '')}")
                st.write(f"  **Response:** {cr.get('response', '')}")

            st.subheader("Missing info to strengthen")
            for mi in memo.get("missing_info_to_strengthen", []) or []:
                st.write(f"- {mi}")

            with st.expander("Debug: Evidence pack meta", expanded=False):
                st.json(memo.get("_evidence_pack_meta", {}))

st.markdown("---")
st.caption("Tip: Add who did what to whom, what was refused, and what relief you want (inspection, injunction, damages).")


