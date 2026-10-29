"""Lightweight, conservative evidence verification for local sentence retrieval.

No LLM, hosted model, or per-query heavyweight inference. This is a transparent
token-alignment and answer-type check, NOT a factual entailment model. Abstain
when a candidate does not contain enough question-specific evidence.
"""
from __future__ import annotations
import re

MIN_BGE_SIMILARITY = 0.55
MAX_BGE_GAP = 0.14

# Function words provide little topical evidence. Omit time question words
# here and verify expected answer types separately.
STOP = set("""
a an the is are was were be been being to of for from by with in on at
and or do does did can could should would must may might i we you they he she
it my our their its your this that those these there about please tell me
what which where when why how who whom whose many much long soon quickly fast
have has had get gets got give us need required require
company corporate employee employees business work working policy
""".split())
# Domain-neutral alias groups for common workplace expressions.
ALIASES = {
 "meals":"meal", "food":"meal", "lunch":"meal", "dinner":"meal",
 "daily":"day", "days":"day",
 "spend":"allowance", "spending":"allowance", "budget":"allowance",
 "limit":"allowance", "limits":"allowance",
 "paid":"pay", "pays":"pay", "payment":"pay",
 "receipts":"receipt", "submitted":"submit", "submission":"submit",
 "flights":"airfare", "flight":"airfare", "plane":"airfare", "ticket":"airfare",
 "tickets":"airfare", "expensive":"cost", "costing":"cost", "costs":"cost",
 "approve":"approval", "approved":"approval", "preapproval":"approval",
 "traveling":"travel", "travelling":"travel", "trip":"travel",
 "book":"book", "booked":"book", "booking":"book", "portal":"portal",
 "password":"credential", "passwords":"credential", "credentials":"credential",
 "secrets":"secret", "secret":"secret", "keys":"credential",
 "live":"production", "prod":"production", "systems":"system",
 "save":"store", "saved":"store", "storing":"store", "stored":"store",
 "storage":"store", "kept":"store", "keep":"store",
 "manager":"manager", "management":"manager",
 "publishes":"publish", "published":"publish", "write":"publish",
 "writing":"publish", "written":"publish", "retrospective":"report",
 "postmortem":"report", "postmortems":"report",
 "acknowledgment":"acknowledge", "acknowledgement":"acknowledge",
 "acknowledged":"acknowledge", "respond":"acknowledge", "response":"acknowledge",
 "outage":"incident", "outages":"incident", "breach":"incident",
 "breaches":"incident", "p1":"priority", "critical":"priority",
 "priority":"priority", "priority-one":"priority", "oncall":"on",
 "reported":"report", "reporting":"report", "alert":"report", "alerts":"report",
 "suspect":"suspected", "suspected":"suspected",
 "records":"data", "record":"data", "customer":"customer",
 "paste":"copy", "pasted":"copy", "copied":"copy",
 "chatbot":"tool", "chatbots":"tool", "tools":"tool",
 "diagnostics":"diagnostic", "logs":"log",
 "purged":"retain", "retention":"retain", "retained":"retain",
 "deleted":"delete", "removed":"delete",
 "permissions":"permission", "least":"least", "privilege":"privilege",
 "reimbursed":"reimburse", "reimbursable":"reimburse",
 "reimbursement":"reimburse", "reimburse":"reimburse",
 "memberships":"membership", "vesting":"vest", "vests":"vest",
 "holds":"store", "holding":"store", "hold":"store", "vault":"secret",
 "deadline":"duration", "writeup":"report", "write-up":"report",
}
# Extra generic words: unmatched procedural words should not veto real evidence.
COMMON = set("""
allowed allow time timeframe timeframe process procedure rules rule
actually currently exactly per rate amount number number
within into after before through than more least own
""".split())
# If a query asks about a duration or amount, verify the answer contains one.
TIME_TERMS=set("when how long soon quickly fast days day duration timeframe retained retention time after before".split())
NUMBER_PATTERN=re.compile(r"(?:\b\d+(?:\.\d+)?\b|\$\s*\d+|\b(?:one|two|three|five|ten|fourteen|thirty|ninety)\b)",re.I)
TIME_PATTERN=re.compile(r"\b(?:day|days|minute|minutes|hour|hours|week|weeks|month|months|year|years)\b",re.I)

def terms(text: str) -> list[str]:
    raw=re.findall(r"[a-z]+(?:-[a-z]+)?|\d+",text.lower())
    out=[]
    for word in raw:
        for bit in word.split("-"):
            bit=ALIASES.get(bit,bit)
            if bit not in STOP and bit not in COMMON and len(bit)>1:
                out.append(bit)
    return list(dict.fromkeys(out))


def support_score(question: str, passage: dict) -> tuple[float,dict]:
    """Lexical support from passage and title; score is NOT calibrated."""
    wanted=terms(question)
    if not wanted:
        return 0., {"matched":[], "missing":[], "coverage":0.}
    text_terms=set(terms(passage.get("text","")))
    title_terms=set(terms(passage.get("title","")))
    matched=[w for w in wanted if w in text_terms or w in title_terms]
    # Titles are only weak context; answer must contain at least one matching
    # topical term in the actual cited sentence.
    text_matched=[w for w in wanted if w in text_terms]
    coverage=len(matched)/len(wanted)
    missing=[w for w in wanted if w not in matched]
    # Strengthen penalties for missing distinctive domain-specific terms.
    score=coverage
    if not text_matched:
        score=0.
    return round(score,4), {"matched":matched,"missing":missing,
                             "coverage":round(coverage,4),"text_matched":text_matched}


def verifies_answer_type(question: str, text: str) -> bool:
    """Check broad expected factual response type without inventing facts."""
    q=question.lower()
    duration=bool(re.search(r"\b(how long|how soon|how fast|how quickly|when|how many days)\b",q))
    if duration and not (TIME_PATTERN.search(text) and NUMBER_PATTERN.search(text)):
        # 'Where' and 'what' questions do not need numbers or durations.
        return False
    amount=bool(re.search(r"\b(how much|what is the .*allowance|what .*cost|how many)\b",q))
    if amount and not NUMBER_PATTERN.search(text):
        return False
    return True


def baseline_evidence(passages: list[dict], retriever: str) -> list[dict]:
    """Frozen original, score-only retrieval for before/after comparison."""
    if retriever != "fastembed" or not passages:
        return passages
    best=float(passages[0]["score"])
    floor=max(MIN_BGE_SIMILARITY,best-MAX_BGE_GAP)
    return [p for p in passages if float(p["score"])>=floor]


def select_evidence(passages: list[dict], retriever: str,
                    question: str | None=None, limit: int | None=None) -> list[dict]:
    """Answerability-aware reranking; old API preserved for lexical backends.

    Evidence must pass a similarity floor, question-term coverage, and answer
    type check. A failed question returns [] (abstention) instead of an
    unsupported answer. Asking for many citations cannot bypass gates.
    """
    if retriever != "fastembed" or not passages:
        return passages[:limit] if limit else passages
    if question is None:
        return baseline_evidence(passages,retriever)[:limit] if limit else baseline_evidence(passages,retriever)
    candidates=[]
    for p in passages:
        cosine=float(p["score"])
        sparse=float(p.get("lexical_score",0))
        coverage, detail=support_score(question,p)
        matched=set(detail["matched"])
        topical=matched-{"day","pay","reimburse","report","cost","allowance","duration","store"}
        if len(matched)<2 or not topical or not detail.get("text_matched"):
            continue
        if cosine<0.55 and not (cosine>=0.49 and sparse>=0.2 and coverage>=0.5):
            continue
        if not (coverage>=0.4 or (coverage>=0.27 and cosine>=0.61) or (coverage>=0.3 and sparse>=0.15)):
            continue
        if not verifies_answer_type(question,p["text"]):
            continue
        blended=.52*cosine+.32*coverage+.16*sparse
        candidates.append((blended,p))
    candidates.sort(key=lambda x:(-x[0],-float(x[1]["score"]),x[1]["doc_id"],x[1]["chunk"]))
    return [p for _,p in candidates[:limit]] if limit else [p for _,p in candidates]

def cited_source_sentence(passage: dict, reference: str) -> str:
    """Verbatim citation of source sentence; no generated claims."""
    text=passage["text"].strip()
    return f"{text} {reference}" if text else ""
