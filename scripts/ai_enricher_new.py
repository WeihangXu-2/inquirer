# scripts/phase1_extractor.py
# Phase 1: scenario understanding (NO legal conclusions)
# Deterministic extraction with transparent heuristics

from __future__ import annotations
import re
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional
import os
import math
from dataclasses import dataclass
from typing import Tuple, Optional

# ---- Basic normalizers ----
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

# ---- Role lexicon (expand as you like) ----
ROLE_PATTERNS: List[tuple[str, str]] = [
    ("shareholder", r"\b(minority\s+)?shareholder(s)?\b"),
    ("member", r"\b(llc\s+)?member(s)?\b"),
    ("manager", r"\bmanager(s)?\b"),
    ("director", r"\bdirector(s)?\b"),
    ("officer", r"\bofficer(s)?\b"),
    ("corporation", r"\bcorporation\b|\bcorp\.\b|\binc\.\b"),
    ("llc", r"\bllc\b|\blimited\s+liability\s+company\b"),
    ("partnership", r"\bpartnership\b|\blimited\s+partnership\b|\bLP\b"),
    ("employee", r"\bemployee(s)?\b"),
    ("employer", r"\bemployer(s)?\b|\bcompany\b"),
    ("lender", r"\blender(s)?\b|\bbank\b|\bcreditor(s)?\b"),
    ("borrower", r"\bborrower(s)?\b|\bdebtor(s)?\b"),
    ("plaintiff", r"\bplaintiff(s)?\b"),
    ("defendant", r"\bdefendant(s)?\b"),
]
ROLE_RX = [(name, re.compile(pat, re.I)) for name, pat in ROLE_PATTERNS]

# ---- Action / dispute triggers ----
ACTION_PATTERNS: List[tuple[str, str]] = [
    ("demanded_access_records", r"\bdemand(ed|s)?\b.*\b(books?\s+and\s+records|records|inspection)\b|\binspect(ion)?\b"),
    ("refused_access_records", r"\brefus(ed|es|al)\b.*\b(inspect(ion)?|books?\s+and\s+records|records)\b|\bdeni(ed|es)\b.*\binspection\b"),
    ("filed_action", r"\bfiled\b.*\b(action|lawsuit|complaint|petition)\b|\bcommenced\b.*\b(action|proceeding)\b"),
    ("motion_to_dismiss", r"\bmotion\s+to\s+dismiss\b|\brule\s*12\b|\b12\(b\)\(?6\)?\b"),
    ("summary_judgment", r"\bsummary\s+judgment\b|\brule\s*56\b"),
    ("prelim_injunction", r"\bpreliminary\s+injunction\b|\btemporary\s+restraining\s+order\b|\bTRO\b|\brule\s*65\b"),
    ("fraud_alleged", r"\bfraud\b|\bmisrepresent(ation)?\b|\bomission\b|\bconceal\b"),
    ("udtpa_alleged", r"\bunfair\s+and\s+deceptive\b|\b75-1\.1\b|\bUDTPA\b"),
    ("trade_secret_alleged", r"\btrade\s+secret\b|\bmisappropriat"),
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

# ---- Very lightweight "party name" heuristic ----
# (We avoid heavy NER; this just captures “X v. Y” if user supplies it)
VS_RX = re.compile(r"([A-Z][\w&.,' -]{2,80})\s+v\.?\s+([A-Z][\w&.,' -]{2,80})")

# ---- Procedural posture heuristic ----
POSTURE_PATTERNS: List[tuple[str, str]] = [
    ("pre_filing", r"\bconsidering\b|\bplanning\b|\bthreaten(ed|ing)?\b"),
    ("filed_complaint", r"\bcomplaint\b|\bfiled\b.*\b(action|lawsuit|complaint|petition)\b"),
    ("motion_stage", r"\bmotion\b|\brule\s*12\b|\bsummary\s+judgment\b|\brule\s*56\b|\bpreliminary\s+injunction\b"),
    ("appeal_stage", r"\bappeal\b|\bappellate\b"),
]
POSTURE_RX = [(name, re.compile(pat, re.I | re.S)) for name, pat in POSTURE_PATTERNS]

@dataclass
class Phase1Facts:
    parties: Dict[str, Any]          # extracted party names (if any) + roles list
    roles: List[str]                 # role labels found
    actions: List[str]               # action labels found
    dispute_triggers: List[str]      # subset of actions that are "conflict triggers"
    relief_sought: List[str]         # remedies
    procedural_posture: str          # best guess among posture labels
    key_phrases: List[str]           # short anchors copied from the user text
    warnings: List[str]              # extraction caveats

def extract_phase1_facts(user_text: str) -> Dict[str, Any]:
    text = norm(user_text)
    lower = text.lower()

    warnings: List[str] = []
    roles_found: List[str] = []
    for name, rx in ROLE_RX:
        if rx.search(text):
            roles_found.append(name)

    actions_found: List[str] = []
    for name, rx in ACTION_RX:
        if rx.search(text):
            actions_found.append(name)

    remedies_found: List[str] = []
    for name, rx in REMEDY_RX:
        if rx.search(text):
            remedies_found.append(name)

    # dispute triggers: things that represent a conflict inflection point
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

    # procedural posture: choose the "most advanced" one detected
    posture = "unknown"
    detected = []
    for name, rx in POSTURE_RX:
        if rx.search(text):
            detected.append(name)
    # precedence ordering
    precedence = ["appeal_stage", "motion_stage", "filed_complaint", "pre_filing"]
    for p in precedence:
        if p in detected:
            posture = p
            break

    # parties (only if user text includes X v. Y)
    parties = {"caption": "", "plaintiff": "", "defendant": "", "party_names": []}
    m = VS_RX.search(text)
    if m:
        p1, p2 = norm(m.group(1)), norm(m.group(2))
        parties["caption"] = f"{p1} v. {p2}"
        parties["plaintiff"] = p1
        parties["defendant"] = p2
        parties["party_names"] = [p1, p2]

    # key phrases: a few short snippets as anchors (no LLM)
    key_phrases = []
    # very basic phrase extraction (grab noun-ish chunks by regex anchors)
    for rx in [
        re.compile(r"\bbooks?\s+and\s+records\b", re.I),
        re.compile(r"\bminority\s+shareholder\b", re.I),
        re.compile(r"\bcompel\b.*\binspection\b", re.I),
        re.compile(r"\bmotion\s+to\s+dismiss\b", re.I),
        re.compile(r"\btrade\s+secret(s)?\b", re.I),
        re.compile(r"\bnon[-\s]*compete\b", re.I),
        re.compile(r"\bfraud(ulent)?\b", re.I),
        re.compile(r"\bunfair\s+and\s+deceptive\b", re.I),
    ]:
        mm = rx.search(text)
        if mm:
            key_phrases.append(norm(mm.group(0)))

    roles_found = uniq_keep_order(roles_found)
    actions_found = uniq_keep_order(actions_found)
    remedies_found = uniq_keep_order(remedies_found)
    dispute_triggers = uniq_keep_order(dispute_triggers)
    key_phrases = uniq_keep_order(key_phrases)

    if not roles_found:
        warnings.append("No clear party roles detected (shareholder/member/employee/lender/etc.).")
    if not actions_found:
        warnings.append("No clear actions detected. Consider adding what happened (demand/refusal/transfer/termination).")
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

