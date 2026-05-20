import logging
import os
import re
import time
from typing import Optional, Literal, Iterable, Annotated

from agent.schema import OwnershipType, OccupancyType, IncomeSource, STRICT_EMAIL_REGEX

logger = logging.getLogger("voip.parsers")

# =============================================================================
# Constants & Private Helpers
# =============================================================================

_SPEECH_FILLER_RE = re.compile(
    r"^[\s.,!?]*(yeah|yep|yea|yah|okay|ok|sure|right|"
    r"uh+|u+m+|ah+|er+|well|hmm+|oh|hey|hi|like)[\s.,!?]+",
    re.IGNORECASE,
)

UNCERTAINTY_MARKERS = [
    "maybe", "i think", "i believe", "not sure", "possibly", "probably",
    "i guess", "somewhere around", "roughly", "about", "don't know",
    "can't remember", "if i recall", "or something", "something like",
    "might be", "could be", "i'm not certain", "approximately",
    "wait", "let me check",
]

_WORD_NUMS: dict[str, int] = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
    "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30,
    "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
    "eighty": 80, "ninety": 90,
    "a": 1,
}

_MONTH_NAMES: dict[str, int] = {
    "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3,
    "april": 4, "apr": 4, "may": 5, "june": 6, "jun": 6,
    "july": 7, "jul": 7, "august": 8, "aug": 8, "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10, "november": 11, "nov": 11, "december": 12, "dec": 12,
}
_MONTH_PAT = "|".join(_MONTH_NAMES.keys())

_ORDINAL_WORDS: dict[str, str] = {
    "first": "one",
    "second": "two",
    "third": "three",
    "fourth": "four",
    "fifth": "five",
    "sixth": "six",
    "seventh": "seven",
    "eighth": "eight",
    "ninth": "nine",
    "tenth": "ten",
    "eleventh": "eleven",
    "twelfth": "twelve",
    "thirteenth": "thirteen",
    "fourteenth": "fourteen",
    "fifteenth": "fifteen",
    "sixteenth": "sixteen",
    "seventeenth": "seventeen",
    "eighteenth": "eighteen",
    "nineteenth": "nineteen",
    "twentieth": "twenty",
    "thirtieth": "thirty",
}

_STATE_NAME_TO_ABBR = {
    "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR",
    "california": "CA", "colorado": "CO", "connecticut": "CT", "delaware": "DE",
    "florida": "FL", "georgia": "GA", "hawaii": "HI", "idaho": "ID",
    "illinois": "IL", "indiana": "IN", "iowa": "IA", "kansas": "KS",
    "kentucky": "KY", "louisiana": "LA", "maine": "ME", "maryland": "MD",
    "massachusetts": "MA", "michigan": "MI", "minnesota": "MN", "mississippi": "MS",
    "missouri": "MO", "montana": "MT", "nebraska": "NE", "nevada": "NV",
    "new hampshire": "NH", "new jersey": "NJ", "new mexico": "NM",
    "new york": "NY", "north carolina": "NC", "north dakota": "ND", "ohio": "OH",
    "oklahoma": "OK", "oregon": "OR", "pennsylvania": "PA", "rhode island": "RI",
    "south carolina": "SC", "south dakota": "SD", "tennessee": "TN", "texas": "TX",
    "utah": "UT", "vermont": "VT", "virginia": "VA", "washington": "WA",
    "west virginia": "WV", "wisconsin": "WI", "wyoming": "WY",
}

_GREETING_SOFT_DECLINE_PATTERNS = (
    "not now", "not a good time", "bad time", "not the right time",
    "call back", "call later", "call me later", "call me back",
    "not free", "not available", "busy right now", "i'm busy",
    "catch me later", "another time", "can i call", "can you call",
    "reach me later", "not ready", "not interested",
)

# Mid-intake "I want to stop" patterns. Specific enough that they shouldn't
# false-positive on field answers. Caller has already passed GREETING and is
# now backing out partway through. Triggers a graceful goodbye + terminate.
_MID_INTAKE_DECLINE_PATTERNS = (
    "i changed my mind",
    "i'm not interested anymore",
    "let me think about it",
    "let me check with my",  # spouse / wife / husband / partner
    "i want to stop",
    "i'd like to stop",
    "can we stop",
    "let's stop",
    "i don't want to continue",
    "i don't want to do this",
    "let me call you back",
    "let me call back",
    "i'll call back",
    "can we do this later",
    "let's do this later",
    "i need to think about",
)

# Hard Do-Not-Call patterns — caller is asking us to remove them entirely.
# Distinct from soft decline because the message is firmer and we should never
# attempt a callback.
_DNC_PATTERNS = (
    "don't call me again",
    "do not call",
    "stop calling",
    "take me off",          # "take me off your list"
    "remove me from",       # "remove me from your list"
    "no more calls",
    "stop contacting",
    "unsubscribe",
    "block this number",
)

_DECLINE_TO_SPECIFY_PATTERNS = (
    r"not sure(?!\s+about\b)", "don't know", "skip", "pass", "prefer not to say",
    "don't want to say", "not comfortable", "decline", "not saying",
    "none of your business", "unknown", "undisclosed",
    "no idea", "don't recall", "don't remember", "can't remember", "no clue",
)

# =============================================================================
# Helper Functions
# =============================================================================

def normalize_answer_scaffold(text: str) -> str:
    """Remove natural answer lead-ins + speech fillers before deterministic field inference."""
    if not text or not str(text).strip():
        return ""
    s = re.sub(r"\s+", " ", str(text).strip())
    # Strip leading speech fillers repeatedly (handles "Yeah. Um, ...")
    # Don't strip "yes"/"no" — they are valid answers for y/n questions.
    for _ in range(4):
        m = _SPEECH_FILLER_RE.match(s)
        if not m:
            break
        remainder = s[m.end():].strip()
        if not remainder:
            break
        s = remainder
    s = re.sub(
        r"^\s*(?:"
        r"it'?s|it\s+is|this\s+is|"
        r"i\s+am|i'?m|"
        r"my\s+answer\s+is|the\s+answer\s+is|"
        r"that\s+would\s+be|it\s+would\s+be|"
        r"i\s+would\s+say|probably|i\s+think"
        r")\s+",
        "",
        s,
        flags=re.IGNORECASE,
    )
    return s.strip()

def _stt_confidence_triggers_reprompt(conf: Optional[float]) -> bool:
    """True when provider confidence is in [0,1] and below env threshold."""
    if conf is None:
        return False
    try:
        f = float(conf)
    except (TypeError, ValueError):
        return False
    if f < 0.0 or f > 1.0:
        return False
    try:
        thresh = float(os.getenv("PATH2_STT_LOW_CONFIDENCE_REPROMPT_MAX", "0.52"))
    except ValueError:
        thresh = 0.52
    return f < thresh

def _use_deterministic_address_parsing() -> bool:
    return os.getenv("ENABLE_DETERMINISTIC_ADDRESS_PARSING", "1") == "1"

def detect_uncertainty(raw_input: str) -> list[str]:
    """
    Scan raw user speech for hedging / uncertainty language.
    
    Returns a list of matched markers found in the input.
    Empty list means the user sounded confident.
    """
    if not raw_input:
        return []
    
    lowered = raw_input.lower()
    return [marker for marker in UNCERTAINTY_MARKERS if marker in lowered]

def infer_occupancy_with_optional_recovery(
    text: str,
    recovery: Optional[str],
) -> tuple[str, Optional[OccupancyType]]:
    """
    Parse occupancy from final transcript, then recovery partial, then concatenations.
    Returns (effective_text_for_raw_user_input, occupancy or None).
    """
    t = (text or "").strip()
    r = (recovery or "").strip()
    seen: list[str] = []
    for cand in (t, r, f"{r} {t}".strip(), f"{t} {r}".strip()):
        if not cand or cand in seen:
            continue
        seen.append(cand)
        occ = infer_occupancy_from_transcript(cand)
        if occ is not None:
            return cand, occ
    return t, None

def _name_slot_focus_from_missing(missing_blob: str) -> Optional[str]:
    """If get_missing_fields() includes Ask next, map to first- or last-name-only intake."""
    from agent.state_manager import _ask_from_missing_line
    ask = _ask_from_missing_line(missing_blob)
    if not ask:
        return None
    q = ask.lower()
    if "first name" in q:
        return "first"
    if "last name" in q:
        return "last"
    return None

def _transcript_supports_saved_dob(raw_user_input: Optional[str], dob: str) -> bool:
    """Block LLM/tool hallucinated DOB when speech lacks month, day, and year cues."""
    if not raw_user_input or not str(raw_user_input).strip() or not dob:
        return False
    parts = dob.strip().split("/")
    if len(parts) != 3:
        return True
    try:
        mm, dd, yyyy = int(parts[0]), int(parts[1]), int(parts[2])
    except ValueError:
        return True

    # Authoritative check: if our own deterministic DOB parser extracts the
    # same date from this transcript, the speech supports it. Handles spoken
    # years ("nineteen eighty eight"), ordinals ("first"), and scaffolding
    # that the regex-based checks below miss.
    canonical = f"{mm:02d}/{dd:02d}/{yyyy}"
    if infer_dob_from_transcript(raw_user_input) == canonical:
        return True

    raw = str(raw_user_input).strip()
    scaffolded = normalize_answer_scaffold(raw) or raw
    low_sp = scaffolded.lower()
    low_sp = re.sub(r"\s+", " ", re.sub(r"[^\w\s/\.-]", " ", low_sp)).strip()

    # Digits?
    for sep in ("/", "-", "."):
        esc = re.escape(sep)
        if re.search(rf"\b{mm}\s*{esc}\s*{dd}\s*{esc}\s*{yyyy}\b", low_sp):
            return True
        if re.search(rf"\b{dd}\s*{esc}\s*{mm}\s*{esc}\s*{yyyy}\b", low_sp):
            return True

    # Words?
    month_name = [name for name, num in _MONTH_NAMES.items() if num == mm][0]
    if month_name in low_sp:
        # Day cue (digit, ordinal word, or cardinal word like "first"/"one")
        day_present = (
            str(dd) in low_sp
            or _int_to_spoken_ordinal(dd) in low_sp
            or any(
                _spoken_number_to_int(tok) == dd
                for tok in low_sp.split()
            )
        )
        # Year cue (digits OR spoken form like "nineteen eighty eight")
        year_present = str(yyyy) in low_sp or str(yyyy)[-2:] in low_sp
        if not year_present:
            tokens = low_sp.split()
            for size in (4, 3, 2):
                for i in range(len(tokens) - size + 1):
                    if _parse_spoken_year(" ".join(tokens[i : i + size])) == yyyy:
                        year_present = True
                        break
                if year_present:
                    break
        if day_present and year_present:
            return True

    # Pure space-separated digits?
    if re.search(rf"\b{mm}\s+{dd}\s+{yyyy}\b", low_sp):
        return True
    if re.search(rf"\b{dd}\s+{mm}\s+{yyyy}\b", low_sp):
        return True

    return False

def _int_to_spoken_ordinal(n: int) -> str:
    for word, val in _ORDINAL_WORDS.items():
        if _spoken_number_to_int(val) == n:
            return word
    return str(n)

def transcript_is_dnc(text: str) -> bool:
    """True when the caller is asking us to never call them again.

    Distinct from soft decline — caller wants no callback, no follow-up.
    Triggers a firm goodbye + terminate, with no callback offer.
    """
    if not text or not str(text).strip():
        return False
    low = str(text).lower()
    return any(p in low for p in _DNC_PATTERNS)


def transcript_is_intake_decline(text: str) -> bool:
    """True when the caller wants to stop the call (graceful goodbye).

    Combines:
      - Greeting-style soft decline ("not now", "not interested", "call back")
      - Mid-intake decline ("I changed my mind", "let me think about it")

    Caller will get a warm goodbye and we'll terminate the session. They may
    call back later, unlike DNC where we never contact again.
    """
    if not text or not str(text).strip():
        return False
    low = str(text).lower()
    return (
        any(p in low for p in _GREETING_SOFT_DECLINE_PATTERNS)
        or any(p in low for p in _MID_INTAKE_DECLINE_PATTERNS)
    )


def transcript_requests_intake_default(text: str) -> bool:
    """True when the user explicitly declines to provide a field."""
    if not text or not str(text).strip():
        return False
    low = str(text).strip().lower()
    return any(re.search(p, low) for p in _DECLINE_TO_SPECIFY_PATTERNS)

def transcript_is_status_check(text: str) -> bool:
    """Presence / status pings should never be parsed as applicant data."""
    if not text or not str(text).strip():
        return False
    if transcript_seeks_question_repeat(text):
        return True
    t = str(text).strip().lower()
    # Long utterances contain substantive content — only question-repeat
    # patterns (checked above) apply.  Short presence pings are ≤8 words.
    if len(t.split()) > 8:
        return False
    t = re.sub(r"[^\w\s?']", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    patterns = (
        r"\b(you there|are you there|still there|hello)\b",
        r"\b(you good|are you good|are we good|still with me)\b",
        r"\b(can you hear me|do you hear me|did you hear me)\b",
        # "go ahead" / "continue" are CONSENT, not pings — handled by yes/no parser.
        r"\b(what'?s next|next question)\b",
        r"\b(what'?s the question|what is the question|which question)\b",
        r"\b(repeat|say that again|say again|come again)\b",
        r"\b(i didn'?t hear|didn'?t catch|what did you say)\b",
        r"\b(i'?m lost|i'?m confused|what do you need|what do i do)\b",
        r"\b(pardon|excuse me|sorry what)\b",
        r"\b(thank you|thanks)\b",
    )
    return any(re.search(p, t) for p in patterns)

def transcript_seeks_question_repeat(text: str) -> bool:
    """True when the caller wants the current question repeated (vs a pure presence ping)."""
    if not text or not str(text).strip():
        return False
    t = normalize_answer_scaffold(text).lower()
    t = re.sub(r"[^\w\s']", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    if not t:
        return False
    if re.fullmatch(r"(are you there|still there|you there)(\s+there)?(\?)?", t):
        return False
    if re.fullmatch(r"(hello|hey|hi)(\s+there)?(\?)?", t):
        return False
    if re.fullmatch(r"(thank you|thanks)(\s+you)?(\?)?", t):
        return False
    patterns = (
        r"\bwhat'?s\s+the\s+question\b",
        r"\bwhat\s+is\s+the\s+question\b",
        r"\bwhich\s+question\b",
        r"\bwhat\s+question\b",
        r"\bwhat\s+(?:exactly\s+)?(?:do\s+you\s+need|are\s+you\s+asking)\b",
        r"\bwhat\s+info(?:rmation)?\s+(?:do\s+you\s+need|should\s+i\s+give)\b",
        r"\bwhat\s+details\s+do\s+you\s+need\b",
        r"\b(remind\s+me|tell\s+me\s+again)\b",
        r"^\s*repeat\s*\??\s*$",
        r"\brepeat\s+(please|that)\b",
        r"\b(repeat|say)\s+(?:that|the\s+question|it)\s+again\b",
        r"\bsay\s+again\b",
        r"\bcome\s+again\b",
        r"\bpardon\b",
        r"\bexcuse\s+me\b",
        r"\bsorry\s*,?\s*what\b",
        r"\bwhat\s+did\s+you\s+say\b",
        r"\bi\s+didn'?t\s+(?:catch|hear|get|understand)\s+(?:that|it)\b",
        r"\bi\s+m\s+(?:lost|confused)\b",
        r"\bwhat\s+do\s+i\s+(?:need\s+to\s+)?(?:say|answer)\b",
        r"\bwhat\s+should\s+i\s+say\b",
        r"\bwhat'?s\s+next\b",
        r"\bnext\s+question\b",
        # "go ahead" / "continue" / "proceed" are CONSENT to move forward,
        # not requests to repeat. Caught by infer_yes_no_from_transcript.
    )
    return any(re.search(p, t) for p in patterns)

def _is_obvious_non_answer(text: str) -> bool:
    if not text or not str(text).strip():
        return True
    t = normalize_answer_scaffold(text).lower()
    t = re.sub(r"[^\w\s']", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    if not t:
        return True
    patterns = (
        r"^(sorry|sorry about that|my bad)$",
        r"^(that'?s all|that is all|all set|never mind|nevermind)$",
        r"^(good day|bye|goodbye)$",
        r"^(thank you|thanks)$",
        r"^(love in the list|lost in the list|waste items delivered)$",
    )
    return any(re.fullmatch(p, t) for p in patterns)

def _ask_from_missing_line(missing_line: str) -> str:
    m = re.search(r'Ask next:\s*"([^"]+)"', missing_line or "")
    if not m:
        return ""
    return (m.group(1) or "").strip()

def _spoken_number_to_int(text: str) -> Optional[int]:
    """Convert English spoken number phrase to int.

    Handles: "two hundred fifty thousand", "eighty thousand",
    "one point five million", "a hundred thousand", etc.
    Returns None on failure.
    """
    tokens = re.split(r"[\s,]+", text.strip().lower())
    # Strip punctuation per-token so inputs like "million." still parse.
    tokens = [re.sub(r"[^a-z0-9]", "", t) for t in tokens if t]
    tokens = [t for t in tokens if t]
    if not tokens:
        return None

    # Handle "point" for decimals before multiplier: "one point five million"
    point_idx = None
    for i, t in enumerate(tokens):
        if t == "point":
            point_idx = i
            break

    if point_idx is not None and point_idx > 0:
        whole_part = _spoken_number_to_int(" ".join(tokens[:point_idx]))
        frac_tokens = tokens[point_idx + 1:]
        if not frac_tokens or whole_part is None:
            return None
        # fraction digit(s) before multiplier
        mult_idx = None
        for i, t in enumerate(frac_tokens):
            if t in ("thousand", "million", "billion"):
                mult_idx = i
                break
        if mult_idx is not None:
            frac_word_part = frac_tokens[:mult_idx]
            mult_word = frac_tokens[mult_idx]
            multiplier = {"thousand": 1_000, "million": 1_000_000}.get(mult_word, 1)
            frac_val = _spoken_number_to_int(" ".join(frac_word_part))
            if frac_val is None:
                return None
            frac_digits = len(str(frac_val))
            result = (whole_part + frac_val / (10 ** frac_digits)) * multiplier
            return int(result)
        return None

    total = 0
    current = 0
    has_value = False
    _TENS = frozenset({20, 30, 40, 50, 60, 70, 80, 90})

    for t in tokens:
        if t in _WORD_NUMS:
            v = _WORD_NUMS[t]
            # Shorthand: "one fifty" = 150, "two thirty" = 230.
            # When current is 1-9 and next word is a tens (20-90), treat as X*100+Y.
            if v in _TENS and 1 <= current <= 9:
                current = current * 100 + v
            else:
                current += v
            has_value = True
        elif t == "hundred":
            current = max(current, 1) * 100
            has_value = True
        elif t == "thousand":
            current = max(current, 1) * 1_000
            total += current
            current = 0
            has_value = True
        elif t == "k":
            current = max(current, 1) * 1_000
            total += current
            current = 0
            has_value = True
        elif t == "million":
            current = max(current, 1) * 1_000_000
            total += current
            current = 0
            has_value = True
        elif t in ("and", "dollars", "dollar", "bucks", "per", "year", "annually"):
            continue
        elif t.isdigit():
            current += int(t)
            has_value = True
        else:
            return None  # unknown token — bail

    if not has_value:
        return None
    total += current
    return total if total > 0 else None

def infer_ownership_from_transcript(text: str) -> Optional[OwnershipType]:
    """Map common spoken variants to OwnershipType enum."""
    if not text or not str(text).strip():
        return None
    text = normalize_answer_scaffold(text)
    if not text:
        return None
    low = text.lower()
    if re.search(
        r"\bsole\b"
        r"|\bsoul\b"
        r"|\bsol\b"
        r"|\bindividual\b"
        r"|\balone\b"
        r"|\bjust\s+me\b"
        r"|\bmyself\b"
        r"|\bonly\s+me\b"
        r"|\bsolely\b"
        r"|\bsold\b"
        r"|\bsoon\b"
        r"|\bsown\b"
        r"|\bsewn\b"
        r"|\bzone\b"
        r"|\bstore\b"
        r"|\bstored\b"
        r"|\bfull\b"
        r"|\bso\s+on\b"
        r"|\bfoam\b"
        r"|\btoll\b"
        r"|\bloner\b"
        r"|\bso\s+loner\b"
        r"|\bsool\b"
        r"|\bhonado\b"
        r"|\bsoler\b"
        r"|\bstole\b"
        r"|\bso\s+lone\b"
        r"|\bso\s+lo\b"
        r"|\bso[,\s]+lon(?:as|a|e|er)?\b"
        r"|\bone\b"
        r"|\b1\b"
        r"|\bwhen\b",
        low,
    ):
        return "Sole owner"
    if re.search(
        r"\bjoin(?:ed|t)?\s+owners?\b"
        r"|\bjoist\s+owners?\b"
        r"|\bjoynt\s+owners?\b"
        r"|\bjointed\s+owners?\b"
        r"|\bjointly\s+owned\b"
        r"|\bjoint\s+owners?\b"
        r"|\bco[\s-]?owners?\b"
        r"|\bsecondary\s+owners?\b"
        r"|\bsecond\s+owners?\b"
        r"|\bjoint\b"
        r"|\btwo\b"
        r"|\b2\b",
        low,
    ):
        return "Joint owner"
    return None

def infer_occupancy_from_transcript(text: str) -> Optional[OccupancyType]:
    """Map common STT phrasing to occupancy when LLM omits occupancy_type."""
    if not text or not str(text).strip():
        return None
    text = normalize_answer_scaffold(text)
    if not text:
        return None
    for part in re.split(r"[.!?\n]+", text):
        seg = part.strip().lower()
        if not seg:
            continue
        seg_n = seg.rstrip(".,!?")
        if re.fullmatch(r"(1|one|won|when)", seg_n):
            return "Primary property"
        if re.fullmatch(r"(2|two|too|to)", seg_n):
            return "Secondary property"
        if re.fullmatch(r"(3|three|tree|free)", seg_n):
            return "Investment property"
    low = text.lower()
    if re.fullmatch(r"\s*(1|one|won|when)\s*[.,!?]?\s*", low):
        return "Primary property"
    if re.fullmatch(r"\s*(2|two|too|to)\s*[.,!?]?\s*", low):
        return "Secondary property"
    if re.fullmatch(r"\s*(3|three|tree|free)\s*[.,!?]?\s*", low):
        return "Investment property"
    if re.search(
        r"\b(investment|investor|rental\s+property|rental\s+home|income\s+property|"
        r"non[- ]?owner\s+occupied|landlord|flip\s+(?:house|property))\b",
        low,
    ):
        return "Investment property"
    if re.search(
        r"\bsecondary\b"
        r"|\bsec+ondary\b"
        r"|\bsecond\s+home\b"
        r"|\bsecond\s+property\b"
        # Cut-off STT: "This is my second" → "my second" after scaffold strip.
        # In occupancy context (parser is gated upstream) "my second" = secondary.
        r"|\bmy\s+second\b"
        r"|\bvacation\s+home\b"
        r"|\bvacation\s+house\b"
        r"|\bweekend\s+home\b"
        r"|\bski\s+(?:house|home)\b",
        low,
    ):
        return "Secondary property"
    if re.search(
        r"\bprimary\s+property\b"
        r"|\bprimary\b"
        r"|\bprimer\b"
        r"|\bprime\s+home\b"
        r"|\bfirst\s+home\b"
        r"|\bmain\s+home\b"
        r"|\bmain\s+residence\b"
        r"|\bowner\s*[- ]?\s*occupied\b",
        low,
    ):
        return "Primary property"
    return None

def infer_occupancy_with_optional_recovery(
    text: str,
    recovery: Optional[str],
) -> tuple[str, Optional[OccupancyType]]:
    """
    Parse occupancy from final transcript, then recovery partial, then concatenations.
    Returns (effective_text_for_raw_user_input, occupancy or None).
    """
    t = (text or "").strip()
    r = (recovery or "").strip()
    seen: list[str] = []
    for cand in (t, r, f"{r} {t}".strip(), f"{t} {r}".strip()):
        if not cand or cand in seen:
            continue
        seen.append(cand)
        occ = infer_occupancy_from_transcript(cand)
        if occ is not None:
            return cand, occ
    return t, None

def infer_for_sale_from_transcript(text: str) -> Optional[bool]:
    """Map short yes/no STT to is_currently_for_sale when the model omits the tool field."""
    if not text or not str(text).strip():
        return None
    text = normalize_answer_scaffold(text)
    if not text:
        return None
    for part in re.split(r"[.!?\n]+", text):
        seg = part.strip().lower()
        if not seg:
            continue
        seg_n = seg.rstrip(".,!?")
        if re.fullmatch(r"(1|one|won|when)", seg_n):
            return True
        if re.fullmatch(r"(2|two|too|to)", seg_n):
            return False
    t = text.strip().lower().rstrip(".,!?")
    if not t:
        return None
    if re.fullmatch(r"(1|one|won|when)", t):
        return True
    if re.fullmatch(r"(2|two|too|to)", t):
        return False
    if re.fullmatch(
        r"(yes|yeah|yep|yup|correct|true|indeed|absolutely|certainly|definitely|sure|of course|it is|uh\s*-?\s*huh)",
        t,
    ):
        return True
    if re.fullmatch(r"(no|nope|nah|negative|false|naw|not|it'?s not|not really)", t):
        return False
    if len(t) <= 28 and re.match(r"^(yes|yeah|yep|yup|it is)\b", t):
        return True
    if len(t) <= 32 and re.match(r"^(no|nope|nah|it'?s not)\b", t):
        return False
    return None

def infer_income_source_from_transcript(text: str) -> Optional[IncomeSource]:
    """Map common spoken variants to IncomeSource enum."""
    if not text or not str(text).strip():
        return None
    text = normalize_answer_scaffold(text)
    if not text:
        return None
    for part in re.split(r"[.!?\n]+", text):
        seg = part.strip().lower()
        if not seg:
            continue
        seg_n = seg.rstrip(".,!?")
        if re.fullmatch(r"(1|one|won|when)", seg_n):
            return "Employed full-time"
        if re.fullmatch(r"(2|two|too|to)", seg_n):
            return "Employed part-time"
        if re.fullmatch(r"(3|three|tree|free)", seg_n):
            return "Self-employed"
        if re.fullmatch(r"(4|four|for)", seg_n):
            return "Retired"
        if re.fullmatch(r"(5|five)", seg_n):
            return "Unemployed / furloughed"
    low = text.lower()
    if re.fullmatch(r"\s*(1|one|won|when)\s*[.,!?]?\s*", low):
        return "Employed full-time"
    if re.fullmatch(r"\s*(2|two|too|to)\s*[.,!?]?\s*", low):
        return "Employed part-time"
    if re.fullmatch(r"\s*(3|three|tree|free)\s*[.,!?]?\s*", low):
        return "Self-employed"
    if re.fullmatch(r"\s*(4|four|for)\s*[.,!?]?\s*", low):
        return "Retired"
    if re.fullmatch(r"\s*(5|five)\s*[.,!?]?\s*", low):
        return "Unemployed / furloughed"

    if "full time" in low or "full-time" in low or "fulltime" in low or "employee" in low or "employment" in low:
        return "Employed full-time"
    if "part time" in low or "part-time" in low or "parttime" in low:
        return "Employed part-time"
    if "self employed" in low or "self-employed" in low or "business owner" in low or "contractor" in low:
        return "Self-employed"
    if "retired" in low or "pension" in low or "social security" in low:
        return "Retired"
    if "unemployed" in low or "furlough" in low or "not working" in low:
        return "Unemployed / furloughed"
    if "alimony" in low:
        return "Alimony"
    if "child support" in low:
        return "Child support"
    if "welfare" in low or "public assistance" in low:
        return "Public assistance program"
    if "separate maintenance" in low:
        return "Separate maintenance payments"
    if re.search(r"\bemployed\b", low):
        return "Employed full-time"
    return None

_MAX_INFERRED_ANNUAL_INCOME = 999_999_999
_INCOME_AMBIGUOUS_STT_CONFIDENCE = 0.82

def infer_annual_income_meta(text: str) -> tuple[Optional[int], bool]:
    """Return (annual income in USD or None, fragile_stt_repair)."""
    if not text or not str(text).strip():
        return (None, False)
    text = normalize_answer_scaffold(text)
    if not text:
        return (None, False)
    low = text.lower().strip()
    low = low.replace(",", " ")
    low = re.sub(r"(?<!\d)\.(?!\d)", " ", low)
    low = re.sub(r"[?!]", " ", low)
    low = re.sub(r"\s+", " ", low).strip()
    low = re.sub(r"\b(medium|millian|millon)\b", "million", low)
    low = re.sub(r"\bmanual\b", "annual", low)
    stripped = low
    stripped = re.sub(r"\b(my|the|our|this|that|it)\b", " ", stripped)
    stripped = re.sub(r"\b(annual|yearly|year)\s+(pre\s*-?\s*tax|pretax)?\s*income\b", " ", stripped)
    stripped = re.sub(
        r"\b(pre\s*-?\s*tax|pretax|income|salary|is|it'?s|about|around|"
        r"could\s+be|would\s+be|might\s+be|maybe|probably|per\s+annum|"
        r"per\s+year|a\s+year|annually)\b",
        " ",
        stripped,
    )
    stripped = re.sub(r"\s+", " ", stripped).strip()

    _millionish = r"(?=\s*(?:million|\bm\b|\bk\b))"

    if re.search(rf"\btwo\s+two\s+five\b{_millionish}", low):
        if re.search(r"\btwo\s+two\s+five\s+million\b", low) or re.search(r"\btwo\s+two\s+five\s+m\b", low):
            val = 225 * 1_000_000
        elif re.search(r"\btwo\s+two\s+five\s+k\b", low):
            val = 225 * 1_000
        else:
            val = None
        if val is not None and 0 < val <= _MAX_INFERRED_ANNUAL_INCOME:
            return (val, True)

    one_one_five_point_sub = False
    if re.search(rf"\bone\s+one\s+five\b{_millionish}", low):
        one_one_five_point_sub = True
        low = re.sub(rf"\bone\s+one\s+five\b{_millionish}", "one point five", low)

    m = re.search(r"(\d+(?:\.\d+)?)\s*m\b", low)
    if m:
        val = int(float(m.group(1)) * 1_000_000)
        if 0 < val <= _MAX_INFERRED_ANNUAL_INCOME:
            return (val, one_one_five_point_sub)
    m = re.search(r"(\d+(?:\.\d+)?)\s*k\b", low)
    if m:
        val = int(float(m.group(1)) * 1_000)
        if 0 < val <= _MAX_INFERRED_ANNUAL_INCOME:
            return (val, one_one_five_point_sub)
    m = re.fullmatch(r"([a-z\s-]+)\s+k\b", low)
    if m:
        spoken_k = _spoken_number_to_int(m.group(1).replace("-", " "))
        if spoken_k is not None:
            val = spoken_k * 1_000
            if 0 < val <= _MAX_INFERRED_ANNUAL_INCOME:
                return (val, one_one_five_point_sub)

    digits_only = re.sub(r"\D", "", low)
    if digits_only:
        try:
            val = int(digits_only)
            if 0 < val <= _MAX_INFERRED_ANNUAL_INCOME:
                return (val, one_one_five_point_sub)
        except ValueError:
            pass

    digit_word_map = {
        "zero": "0", "oh": "0", "o": "0", "one": "1", "two": "2", "three": "3",
        "four": "4", "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    }
    wtokens = [re.sub(r"[^a-z0-9]", "", t) for t in stripped.split()]
    if wtokens and all(t in digit_word_map for t in wtokens):
        seq = "".join(digit_word_map[t] for t in wtokens).lstrip("0") or "0"
        try:
            val = int(seq)
            if 0 < val <= _MAX_INFERRED_ANNUAL_INCOME:
                return (val, one_one_five_point_sub)
        except ValueError:
            pass

    spoken = _spoken_number_to_int(low) or _spoken_number_to_int(stripped)
    if spoken is not None and spoken >= 1_000 and spoken <= _MAX_INFERRED_ANNUAL_INCOME:
        return (spoken, one_one_five_point_sub)

    if spoken is not None and 1 <= spoken <= 999:
        scaled = spoken * 1_000
        if 0 < scaled <= _MAX_INFERRED_ANNUAL_INCOME:
            return (scaled, one_one_five_point_sub)

    return (None, False)

def infer_annual_income_from_transcript(text: str) -> Optional[int]:
    """Map spoken annual income to integer dollars."""
    value, _meta = infer_annual_income_meta(text)
    return value

def infer_other_income_from_transcript(text: str) -> Optional[int]:
    """Infer optional other income; map clear negatives to 0."""
    if not text or not str(text).strip():
        return None
    text = normalize_answer_scaffold(text)
    if not text:
        return None
    low = text.lower().strip()
    normalized = re.sub(r"[^\w\s]", " ", low)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    if re.fullmatch(r"(no|nope|none|nothing|zero|0)(\s+(no|nope|none|nothing|zero|0))*", normalized):
        return 0
    if normalized in {"no other income", "no any other income", "none", "nope", "no", "zero", "0", "nothing"}:
        return 0
    _other_income_negative_phrases = (
        "not at this moment", "not at the moment", "not right now", "not currently",
        "nothing right now", "nothing else right now", "no additional income",
        "dont have any other", "don't have any other", "do not have any other",
        "dont have other", "don't have other", "no other", "not any other",
    )
    if any(p in normalized for p in _other_income_negative_phrases):
        return 0
    apostrophe_fold = re.sub(r"[''`]", "", normalized)
    if any(p in apostrophe_fold for p in _other_income_negative_phrases):
        return 0
    contraction_fold = re.sub(r"\bdon\s+t\b", "dont", normalized, flags=re.IGNORECASE)
    contraction_fold = re.sub(r"\bdo\s+not\b", "donot", contraction_fold, flags=re.IGNORECASE)
    if "dont have any other" in contraction_fold or "donot have any other" in contraction_fold:
        return 0
    return infer_annual_income_from_transcript(text)


_DOB_PARTIAL_SPOKEN_ORDINALS_ALT_RE = re.compile(
    r"\b(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|"
    r"eleventh|twelfth|thirteenth|fourteenth|fifteenth|sixteenth|seventeen(?:th)?|"
    r"eighteen(?:th)?|nineteen(?:th)?|twentieth|twenty\s+first|twenty\s+second|"
    r"twenty\s+third|twenty\s+fourth|twenty\s+fifth|twenty\s+sixth|twenty\s+seventh|"
    r"twenty\s+eighth|twenty\s+ninth|thirtieth|thirty\s+first)\b",
    re.I,
)
_DOB_PARTIAL_CARDINAL_DAY_WORDS_RE = re.compile(
    r"\b(one|two|three|four|five|six|seven|eight|nine|ten|"
    r"eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|"
    r"twenty|twenty\s+one|twenty\s+two|twenty\s+three|twenty\s+four|twenty\s+five|"
    r"twenty\s+six|twenty\s+seven|twenty\s+eight|twenty\s+nine|thirty|thirty\s+one)\b",
    re.I,
)
_DOB_SPOKEN_YEAR_PREFIX_RE = re.compile(r"\b(nineteen|twenty)\b", re.I)

def _parse_spoken_day(text: str) -> Optional[int]:
    low = (text or "").strip().lower()
    if not low:
        return None
    # Check ordinals: "twenty first" -> 21
    for w, val in _ORDINAL_WORDS.items():
        if low == w:
            return _spoken_number_to_int(val)
    # Compound ordinals: "twenty-first"
    if "-" in low:
        parts = low.split("-")
        if len(parts) == 2:
            left, right = parts[0], parts[1]
            if left in ("twenty", "thirty"):
                r_val = _ORDINAL_WORDS.get(right)
                if r_val:
                    l_num = _spoken_number_to_int(left) or 0
                    r_num = _spoken_number_to_int(r_val) or 0
                    return l_num + r_num
    # Raw spoken number
    return _spoken_number_to_int(low)

def _parse_spoken_year(text: str) -> Optional[int]:
    low = (text or "").strip().lower()
    if not low:
        return None
    # "nineteen eighty four" -> 1984
    tokens = re.split(r"[\s-]+", low)
    if len(tokens) == 2:
        v1 = _spoken_number_to_int(tokens[0])
        v2 = _spoken_number_to_int(tokens[1])
        if v1 is not None and v2 is not None:
            if v1 in (19, 20) and 0 <= v2 <= 99:
                return v1 * 100 + v2
            if 1900 <= v1 <= 2020 and v2 == 0:
                return v1
    if len(tokens) == 3:
        v1 = _spoken_number_to_int(tokens[0])
        v2 = _spoken_number_to_int(" ".join(tokens[1:]))
        if v1 is not None and v2 is not None:
            if v1 in (19, 20) and 0 <= v2 <= 99:
                return v1 * 100 + v2
    # Raw spoken number: "two thousand and one"
    if len(tokens) == 1 or "thousand" in low or "hundred" in low:
        val = _spoken_number_to_int(low)
        if val and 1900 <= val <= 2010:
            return val
    return None

def infer_dob_from_transcript(text: str) -> Optional[str]:
    """Map spoken date of birth to MM/DD/YYYY format."""
    if not text or not str(text).strip():
        return None
    text = normalize_answer_scaffold(text)
    if not text:
        return None
    low = text.lower().strip()
    low = re.sub(r"(\d+)(?:st|nd|rd|th)", r"\1", low)
    low = re.sub(r"[,/.-]", " ", low)
    low = re.sub(r"\s+", " ", low).strip()
    # Convert word ordinals to cardinals so "first" scans as 1, etc.
    low = " ".join(_ORDINAL_WORDS.get(t, t) for t in low.split())

    # Strategy: Find mm, dd, yyyy by scanning for month names first,
    # then fallback to numerical scan.
    mm, dd, yyyy = None, None, None
    tokens = low.split()

    # 1. Look for month name
    month_idx = -1
    for i, t in enumerate(tokens):
        if t in _MONTH_NAMES:
            mm = _MONTH_NAMES[t]
            month_idx = i
            break
    
    if mm:
        # Day before?
        if month_idx > 0:
            prev_t = tokens[month_idx - 1]
            if tokens[month_idx - 1] == "of" and month_idx > 1:
                prev_t = tokens[month_idx - 2]
            d = _parse_spoken_day(prev_t)
            if d:
                dd = d
        
        # Day after?
        if dd is None and month_idx + 1 < len(tokens):
            next_t = tokens[month_idx + 1]
            d = _parse_spoken_day(next_t)
            if d:
                if d > 31: # Likely a year
                    yyyy = d
                else:
                    dd = d
        
        # Year?
        if yyyy is None:
            for size in (4, 3, 2, 1):
                for i in range(len(tokens) - size + 1):
                    # Skip tokens already used for month
                    if i <= month_idx < i + size:
                        continue
                    val = _parse_spoken_year(" ".join(tokens[i : i + size]))
                    if val and 1900 <= val <= 2010:
                        yyyy = val
                        break
                if yyyy: break
    else:
        # 2. No month name — numerical scan: MM DD YYYY or DD MM YYYY
        nums = []
        skip = 0
        for i, t in enumerate(tokens):
            if skip > 0:
                skip -= 1
                continue
            
            # Check for year (multi-word)
            found_y = None
            for size in (4, 3, 2, 1):
                if i + size <= len(tokens):
                    val = _parse_spoken_year(" ".join(tokens[i : i+size]))
                    if val and 1900 <= val <= 2010:
                        found_y = val
                        skip = size - 1
                        break
            if found_y:
                nums.append(found_y)
            else:
                nv = _spoken_number_to_int(t)
                if nv is not None:
                    nums.append(nv)
        
        # Look for 3 numbers in a row
        for i in range(len(nums) - 2):
            v1, v2, v3 = nums[i], nums[i+1], nums[i+2]
            # MM DD YYYY?
            if 1 <= v1 <= 12 and 1 <= v2 <= 31 and 1900 <= v3 <= 2010:
                mm, dd, yyyy = v1, v2, v3
                break
            # DD MM YYYY?
            if 1 <= v2 <= 12 and 1 <= v1 <= 31 and 1900 <= v3 <= 2010:
                mm, dd, yyyy = v2, v1, v3
                break

    if mm and dd and yyyy:
        if 1 <= mm <= 12 and 1 <= dd <= 31 and 1900 <= yyyy <= 2010:
            return f"{mm:02d}/{dd:02d}/{yyyy}"
    return None

PARTIAL_DOB_REPROMPT = (
    "Sorry, I didn't quite catch your birthday. Could you say the month, day, and year "
    "one more time? For example, January 1st, 1980."
)

def infer_partial_dob_month_day(text: str) -> bool:
    """True if speech contains month or day but not a full year (prevents premature re-prompts)."""
    if not text or not str(text).strip():
        return False
    low = text.lower().strip()
    has_month = any(m in low for m in _MONTH_NAMES)
    has_day = (
        bool(re.search(r"\b\d{1,2}\b", low))
        or bool(_DOB_PARTIAL_SPOKEN_ORDINALS_ALT_RE.search(low))
        or bool(_DOB_PARTIAL_CARDINAL_DAY_WORDS_RE.search(low))
    )
    # Year check: 4 digits starting with 19 or 20, or spoken year.
    has_year = bool(re.search(r"\b(19|20)\d{2}\b", low))
    if not has_year:
        tokens = low.split()
        for size in (1, 2, 3, 4):
            for i in range(len(tokens) - size + 1):
                if _parse_spoken_year(" ".join(tokens[i : i + size])):
                    has_year = True
                    break
            if has_year:
                break
    
    return (has_month or has_day) and not has_year

PARTIAL_MOBILE_PHONE_REPROMPT = (
    "Sorry, I missed a few digits of your phone number. Could you say your 10-digit mobile "
    "number one more time?"
)

def infer_phone_from_transcript(text: str) -> Optional[str]:
    """Extract 10-digit phone number from noisy speech."""
    if not text or not str(text).strip():
        return None
    text = normalize_answer_scaffold(text)
    if not text:
        return None
    low = text.lower().strip()
    digits = re.sub(r"\D", "", low)
    if not digits:
        # Try spoken digits: "two one three..."
        digits = _spoken_digit_sequence_to_digits(low)
    
    if len(digits) == 11 and digits.startswith("1"):
        digits = digits[1:]
    if len(digits) == 10:
        return digits
    
    m = re.search(r"(\d{3})\D*(\d{3})\D*(\d{4})", low)
    if m:
        return "".join(m.groups())
    return None

def _spoken_digit_sequence_to_digits(text: str) -> str:
    digit_map = {
        "zero": "0", "oh": "0", "o": "0", "one": "1", "won": "1", "two": "2", "too": "2", "to": "2",
        "three": "3", "four": "4", "for": "4", "five": "5", "six": "6", "seven": "7", "eight": "8",
        "ate": "8", "nine": "9",
    }
    tokens = [re.sub(r"[^a-z]", "", t) for t in text.lower().split()]
    res = []
    for t in tokens:
        if t in digit_map:
            res.append(digit_map[t])
        elif t.isdigit() and len(t) == 1:
            res.append(t)
    return "".join(res)

_SPOKEN_EMAIL_NUM_AUX = frozenset({"hundred", "thousand", "million", "and"})
_SPOKEN_EMAIL_SPELL_DIGIT_WORDS = frozenset(
    {
        "zero", "oh", "o", "one", "won", "two", "too", "to", "three", "four", "for",
        "five", "six", "seven", "eight", "ate", "nine",
    }
)

def _chunk_is_digit_spell_only(tokens: list[str]) -> bool:
    if not tokens:
        return False
    for t in tokens:
        tok = re.sub(r"[^a-z0-9]", "", (t or "").lower())
        if not tok:
            return False
        if tok.isdigit():
            if len(tok) != 1:
                return False
            continue
        if tok not in _SPOKEN_EMAIL_SPELL_DIGIT_WORDS:
            return False
    return True

def _token_is_spoken_email_number_piece(t: str) -> bool:
    tok = re.sub(r"[^a-z0-9]", "", (t or "").lower())
    if not tok:
        return False
    if tok.isdigit():
        return True
    return tok in _WORD_NUMS or tok in _SPOKEN_EMAIL_NUM_AUX

def _expand_spoken_numbers_in_email_segment(segment: str) -> str:
    segment = (segment or "").strip().replace("-", " ")
    if not segment:
        return segment
    tokens = re.findall(r"\S+", segment)
    out: list[str] = []
    i = 0
    n = len(tokens)
    while i < n:
        if not _token_is_spoken_email_number_piece(tokens[i]):
            out.append(tokens[i])
            i += 1
            continue
        max_span = min(14, n - i)
        replaced = False
        for length in range(max_span, 0, -1):
            chunk = tokens[i : i + length]
            if not all(_token_is_spoken_email_number_piece(t) for t in chunk):
                continue
            phrase = " ".join(chunk)
            num_val: Optional[int] = None
            digit_run = ""
            if _chunk_is_digit_spell_only(chunk):
                digit_run = _spoken_digit_sequence_to_digits(phrase)
                if digit_run:
                    out.append(digit_run)
                    i += length
                    replaced = True
                    break
                num_val = _spoken_number_to_int(phrase)
            else:
                num_val = _spoken_number_to_int(phrase)
                if num_val is None:
                    digit_run = _spoken_digit_sequence_to_digits(phrase)
            if num_val is not None:
                out.append(str(num_val))
                i += length
                replaced = True
                break
            if digit_run:
                out.append(digit_run)
                i += length
                replaced = True
                break
        if not replaced:
            out.append(tokens[i])
            i += 1
    return " ".join(out)

def _normalize_spoken_email_domain(rhs: str) -> str:
    s = (rhs or "").strip().lower()
    if not s:
        return s
    changed = True
    while changed:
        changed = False
        ns = re.sub(r"\s+(?:dot|period)\s+", ".", s)
        if ns != s:
            s = ns
            changed = True
    s = re.sub(r"\s+(?:dot|period)\s*$", "", s).strip()
    s = re.sub(r"\s+", "", s)
    if s and "." not in s:
        s = f"{s}.com"
    return s

def _normalize_spoken_email_local(lhs: str) -> str:
    expanded = _expand_spoken_numbers_in_email_segment((lhs or "").strip())
    raw_l = expanded.strip().lower()
    raw_l = (
        raw_l.replace(" dot ", ".")
        .replace(" period ", ".")
        .replace(" underscore ", "_")
        .replace(" dash ", "-")
        .replace(" hyphen ", "-")
    )
    return raw_l.replace(" ", "")

def _normalize_spoken_email_separator_phrases(raw: str) -> str:
    s = re.sub(r"\s+", " ", (raw or "").strip())
    low = s.lower()
    low = re.sub(r"\bat\s+the\s+rate\b", " at ", low)
    low = re.sub(r"\bat\s+sign\b", " at ", low)
    return low

def infer_email_from_transcript(text: str) -> Optional[str]:
    if not text or not str(text).strip():
        return None
    text = normalize_answer_scaffold(text)
    if not text:
        return None
    raw = _normalize_spoken_email_separator_phrases(text.strip().lower())
    if " at " in raw:
        left, right = raw.split(" at ", 1)
        local = _normalize_spoken_email_local(left).rstrip(".")
        normalized = f"{local}@{_normalize_spoken_email_domain(right)}"
    elif "@" in raw:
        lp, rp = raw.split("@", 1)
        local = _normalize_spoken_email_local(lp).rstrip(".")
        normalized = f"{local}@{_normalize_spoken_email_domain(rp)}"
    else:
        return None
    normalized = normalized.rstrip(".,;:!?")
    if STRICT_EMAIL_REGEX.match(normalized):
        return normalized
    return None


def infer_property_address_from_transcript(text: str) -> Optional[str]:
    if not text or not str(text).strip():
        return None
    text = normalize_answer_scaffold(text)
    if not text:
        return None
    low = text.lower().strip()
    low = re.sub(r"[^\w\s,.-]", " ", low)
    low = re.sub(r"\s+", " ", low).strip()
    had_comma = "," in low
    low_number_scan = re.sub(r"[,.-]+", " ", low)
    low_number_scan = re.sub(r"\s+", " ", low_number_scan).strip()
    words_to_digits = re.match(
        r"^((?:(?:zero|oh|one|two|three|four|five|six|seven|eight|nine|ten|"
        r"eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|"
        r"nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|"
        r"hundred)\s+){1,8})([a-z].*)$",
        low_number_scan,
    )
    if words_to_digits:
        g1 = words_to_digits.group(1).strip()
        g2 = (words_to_digits.group(2) or "").strip()
        merged_phrase = g1
        tail = g2
        if g2:
            parts2 = g2.split()
            if parts2:
                first_w = re.sub(r"[^a-z0-9]", "", parts2[0]).lower()
                if first_w in _WORD_NUMS:
                    combined = _spoken_number_to_int(f"{g1} {first_w}".strip())
                    head_only = _spoken_number_to_int(g1)
                    if combined is not None and (
                        head_only is None or combined != head_only
                    ):
                        merged_phrase = f"{g1} {first_w}".strip()
                        tail = " ".join(parts2[1:]).strip()
        num_val = _spoken_number_to_int(merged_phrase.replace("-", " "))
        digit_seq = _spoken_digit_sequence_to_digits(merged_phrase.replace("-", " "))
        if num_val is not None:
            if not digit_seq or num_val >= int(digit_seq):
                number = num_val
            else:
                number = int(digit_seq)
        elif digit_seq:
            number = int(digit_seq)
        else:
            number = None

        if number is not None:
            low = f"{number} {tail}".strip()
    low = re.sub(r"\s+", " ", low).strip()
    low = re.sub(r"\b(uh|um|ah|er|like)\b", " ", low)
    low = re.sub(r"\s+", " ", low).strip()
    if re.search(r"\b(lost|list|students|items|below|where)\b", low):
        return None
    m = re.match(r"^\s*(\d{1,6})[\s,.-]+([a-z0-9\s.-]{3,})", low)
    if not m:
        return None
    num = m.group(1)
    rest = m.group(2)
    rest = re.split(r",|\bin\s+[a-z]+\b|\bcalifornia\b|\bca\b", rest, maxsplit=1)[0].strip()
    if not rest:
        return None
    rest = re.sub(r"\bblvd\b", "boulevard", rest)
    rest = re.sub(r"\bave\b", "avenue", rest)
    rest = re.sub(r"\brd\b", "road", rest)
    rest = re.sub(r"\bdr\b", "drive", rest)
    rest = re.sub(r"\bln\b", "lane", rest)
    rest = re.sub(r"\bct\b", "court", rest)
    rest = re.sub(r"\bpkwy\b", "parkway", rest)
    rest = re.sub(r"\bst\b", "street", rest)
    ordinal_word_to_num = {
        "first": "1st", "second": "2nd", "third": "3rd", "fourth": "4th", "fifth": "5th",
        "sixth": "6th", "seventh": "7th", "eighth": "8th", "ninth": "9th", "tenth": "10th",
        "eleventh": "11th", "twelfth": "12th", "thirteenth": "13th", "fourteenth": "14th",
        "fifteenth": "15th", "sixteenth": "16th", "seventeenth": "17th", "eighteenth": "18th",
        "nineteenth": "19th", "twentieth": "20th",
    }
    for w, n in ordinal_word_to_num.items():
        rest = re.sub(rf"\b{w}\b", n, rest)
    rest = re.sub(r"\s+", " ", rest).strip(" .,-")
    if had_comma:
        street_line = re.match(
            r"^(.+?\b(?:street|avenue|boulevard|road|drive|lane|court|parkway|place)\b)",
            rest,
        )
        if street_line:
            rest = street_line.group(1).strip()
    if re.fullmatch(r"(los angeles|california|ca)", rest):
        return None
    if len(rest) < 3:
        return None
    rest_l = rest.lower()
    if re.fullmatch(
        r"(zero|oh|one|two|three|four|five|six|seven|eight|nine|ten|"
        r"eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|"
        r"twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)",
        rest_l,
    ):
        return None
    formatted_tokens: list[str] = []
    for token in rest.split():
        if re.search(r"\d", token):
            formatted_tokens.append(token.lower())
        else:
            formatted_tokens.append(token.capitalize())
    return f"{num} {' '.join(formatted_tokens)}"

def _state_abbr_from_text(text: str) -> Optional[str]:
    normalized = re.sub(r"[^a-zA-Z\s]", " ", text or "")
    normalized = re.sub(r"\s+", " ", normalized).strip().lower()
    for name, abbr in _STATE_NAME_TO_ABBR.items():
        if re.search(rf"\b{re.escape(name)}\b", normalized):
            return abbr
    m = re.search(r"\b([A-Z]{2})\b\s*$", (text or "").strip().upper())
    if m and m.group(1) in set(_STATE_NAME_TO_ABBR.values()):
        return m.group(1)
    return None

def infer_property_city_state_from_transcript(text: str) -> dict[str, str]:
    if not text or not str(text).strip():
        return {}
    text = normalize_answer_scaffold(text)
    if not text:
        return {}
    out: dict[str, str] = {}
    parts = [p.strip(" .") for p in re.split(r",", str(text)) if p.strip(" .")]
    if len(parts) >= 2:
        city_part = parts[1]
        state_from_city_part = _state_abbr_from_text(city_part)
        city = city_part
        if state_from_city_part:
            out["state"] = state_from_city_part
            city = re.sub(r"\b[A-Z]{2}\b", " ", city, flags=re.IGNORECASE)
            for state_name in _STATE_NAME_TO_ABBR:
                city = re.sub(rf"\b{re.escape(state_name)}\b", " ", city, flags=re.IGNORECASE)
        city = re.sub(r"\b\d{5}\b", " ", city)
        city = re.sub(r"[^a-zA-Z\s.-]", " ", city)
        city = re.sub(r"\s+", " ", city).strip(" .-")
        if city and not _state_abbr_from_text(city):
            out["city"] = city.title()
        if len(parts) >= 3:
            state = _state_abbr_from_text(parts[2])
            if state:
                out["state"] = state
    if "state" not in out:
        state = _state_abbr_from_text(str(text))
        if state:
            out["state"] = state
    return out

def build_address_autocomplete_type_query(
    street_line: Optional[str],
    spoken_transcript_hint: Optional[str],
) -> Optional[str]:
    street = (street_line or "").strip()
    if not street:
        return None
    sh = (spoken_transcript_hint or "").strip()
    if not sh:
        return street
    hints = infer_property_city_state_from_transcript(sh)
    tail = " ".join(
        x.strip()
        for x in (hints.get("city"), hints.get("state"))
        if x and str(x).strip()
    )
    if tail:
        return f"{street} {tail}".strip()
    return street

def _spoken_letters_to_word(text: str) -> Optional[str]:
    if not text or not str(text).strip():
        return None
    letter_words = {
        "a": "a", "ay": "a", "b": "b", "be": "b", "bee": "b", "c": "c", "see": "c", "sea": "c",
        "d": "d", "dee": "d", "e": "e", "ee": "e", "f": "f", "eff": "f", "g": "g", "gee": "g",
        "h": "h", "aitch": "h", "i": "i", "eye": "i", "j": "j", "jay": "j", "k": "k", "kay": "k",
        "l": "l", "el": "l", "m": "m", "em": "m", "n": "n", "en": "n", "o": "o", "oh": "o",
        "p": "p", "pee": "p", "q": "q", "cue": "q", "queue": "q", "r": "r", "are": "r", "s": "s",
        "ess": "s", "t": "t", "tea": "t", "tee": "t", "u": "u", "you": "u", "v": "v", "vee": "v",
        "w": "w", "doubleyou": "w", "x": "x", "ex": "x", "y": "y", "why": "y", "z": "z", "zee": "z", "zed": "z",
    }
    cleaned = re.sub(r"\b(letter|letters|spelled|spell|is)\b", " ", text.lower())
    tokens = re.findall(r"[a-z]+", cleaned)
    letters = [letter_words.get(token) for token in tokens]
    if not letters or any(letter is None for letter in letters):
        return None
    if len(letters) < 2 or len(letters) > 30:
        return None
    return "".join(letters).title()

def infer_name_fields_from_transcript(
    text: str,
    *,
    need_first: bool,
    need_last: bool,
    slot_focus: Optional[str] = None,
) -> dict[str, str]:
    if not text or not str(text).strip():
        return {}
    if transcript_is_status_check(text):
        return {}
    raw_s = text.strip().lower()
    explicit_name = bool(
        re.match(
            r"^\W*(my\s+(?:full\s+)?name\s+is|name\s+is|i\s+am|i'?m|this\s+is)\b",
            raw_s,
        )
    )
    s = normalize_answer_scaffold(text).lower()
    if not s:
        return {}
    spelled = _spoken_letters_to_word(s)
    if spelled:
        if slot_focus == "first" and need_first:
            return {"first_name": spelled}
        if slot_focus == "last" and need_last:
            return {"last_name": spelled}
        if need_first and not need_last:
            return {"first_name": spelled}
        if need_last and not need_first:
            return {"last_name": spelled}
    if re.search(
        r"\b("
        r"whatever|brother|sister|mother|father|"
        r"thank(s| you)?|god|word|"
        r"good|hear|there|hello|hi|"
        r"not\s+sure|sure|repeat"
        r")\b",
        s,
    ):
        return {}
    if (
        re.match(r"^\W*my\b", s)
        and not re.match(r"^\W*(my name is|name is|i am|im)\b", s)
    ):
        return {}
    s = re.sub(r"^(my name is|name is|i am|im|it's|it is|this is)\s+", "", s)
    tokens = re.findall(r"[a-zA-Z][a-zA-Z'-]{0,30}", s)
    if not tokens or len(tokens) > 2 or re.search(r"\d", s):
        return {}
    stop = {
        "yes", "no", "primary", "secondary", "investment", "owner", "sole", "joint",
        "million", "thousand", "income", "address", "hello", "hi", "sad", "but",
        "whatever", "brother", "sister", "mother", "father", "god", "word", "thanks", "thank",
    }
    if any(t.lower() in stop for t in tokens):
        return {}
    if len(tokens) == 2 and tokens[0].lower() == tokens[1].lower():
        t = tokens[0].title()
        if need_first and need_last:
            return {"first_name": t}
        if need_first:
            return {"first_name": t}
        if need_last:
            return {"last_name": t}
        return {}
    if slot_focus == "first" and need_first:
        if len(tokens) in (1, 2):
            return {"first_name": tokens[0].title()}
        return {}
    if slot_focus == "last" and need_last:
        if len(tokens) == 1:
            return {"last_name": tokens[0].title()}
        if len(tokens) == 2:
            return {"last_name": tokens[1].title()}
        return {}
    out: dict[str, str] = {}
    if len(tokens) == 1:
        t = tokens[0].title()
        if need_first:
            out["first_name"] = t
        elif need_last:
            out["last_name"] = t
        return out
    t1, t2 = tokens[0].title(), tokens[1].title()
    if need_first and need_last:
        if not explicit_name:
            out["first_name"] = t1
            return out
        out["first_name"] = t1
        out["last_name"] = t2
    elif need_first:
        out["first_name"] = t1
    elif need_last:
        out["last_name"] = t2
    return out

def infer_yes_no_from_transcript(text: str) -> Optional[bool]:
    if not text or not str(text).strip():
        return None
    # Scan RAW text first — normalize_answer_scaffold strips "yeah"/"yep" as
    # filler, which would hide consent like "Yeah. Go ahead." → "Go ahead."
    raw = re.sub(r"[^\w\s']", " ", str(text).lower())
    raw = re.sub(r"\s+", " ", raw).strip()
    negation_re = re.compile(r"\b(no|nope|nah|don't|do not|not really|negative|disagree)\b")
    affirm_re = re.compile(
        r"\b(yes|yeah|yep|yup|yah|sure|sure thing|"
        r"correct|right|agree|agreed|"
        r"absolutely|definitely|certainly|of course|"
        # Consent / "proceed" phrasings — caller is signalling YES, go.
        r"go ahead|go for it|fire away|shoot|proceed|"
        r"let'?s do (?:it|this)|ready|i'?m ready|sounds good)\b"
    )
    if affirm_re.search(raw) and not negation_re.search(raw):
        return True
    if negation_re.search(raw) and not affirm_re.search(raw):
        return False

    # Fall back to scaffold-stripped check for cases like "I'm telling you yes"
    text = normalize_answer_scaffold(text)
    if not text:
        return None
    t = text.strip().lower().rstrip(".,!?")
    compact = re.sub(r"[^\w\s']", " ", t)
    compact = re.sub(r"\s+", " ", compact).strip()
    repeated_tokens = compact.split()
    if repeated_tokens and len(set(repeated_tokens)) == 1:
        t = repeated_tokens[0]
    if affirm_re.search(compact) and not negation_re.search(compact):
        return True
    if negation_re.search(compact):
        return False
    if t in {
        "yes", "yeah", "yep", "yup", "correct", "right", "i agree", "agree", "sure",
        "of course", "absolutely", "definitely", "certainly",
    }:
        return True
    if t in {"no", "nope", "nah", "not really", "i don't think so", "negative"}:
        return False
    return None

def looks_like_answer_for_current_field(text: str, missing_line: str) -> bool:
    if not text or not str(text).strip():
        return False
    if transcript_is_status_check(text) or _is_obvious_non_answer(text):
        return False
    ask = _ask_from_missing_line(missing_line).lower()
    if not ask:
        return True
    normalized = normalize_answer_scaffold(text)
    low = normalized.lower()
    bare = re.sub(r"[^\w\s']", " ", low)
    bare = re.sub(r"\s+", " ", bare).strip()
    if "on file" in ask or "is that correct" in ask or "is that still correct" in ask:
        return infer_yes_no_from_transcript(normalized) is not None
    if "property address" in ask:
        if infer_property_address_from_transcript(normalized):
            return True
        if any(ch.isdigit() for ch in bare):
            return True
        address_patterns = (
            r"\blos\s+angeles\b", r"\bcalifornia\b", r"\bstreet\b", r"\bst\b", r"\bavenue\b", r"\bave\b",
            r"\bboulevard\b", r"\bblvd\b", r"\broad\b", r"\brd\b", r"\bdrive\b", r"\bdr\b", r"\blane\b",
            r"\bln\b", r"\bcourt\b", r"\bct\b", r"\bparkway\b", r"\bpkwy\b", r"\bplace\b", r"\bpl\b",
        )
        return any(re.search(p, bare) for p in address_patterns)
    if "ownership type" in ask or "sole or joint" in ask:
        if infer_ownership_from_transcript(normalized):
            return True
        return bool(re.search(r"\b(sole|soul|sol|joint|owner|one|two|won|too|loner|lonas|alone)\b", bare))
    if "property type" in ask or "primary, secondary, or investment" in ask or ("primary home" in ask and "second home" in ask):
        return infer_occupancy_from_transcript(normalized) is not None
    if "currently for sale" in ask or "for sale" in ask:
        return infer_for_sale_from_transcript(normalized) is not None
    if "annual pre-tax income" in ask or "annual income" in ask:
        return infer_annual_income_from_transcript(normalized) is not None
    if "other income" in ask:
        return infer_other_income_from_transcript(normalized) is not None
    if "primary income source" in ask or "income source" in ask:
        return infer_income_source_from_transcript(normalized) is not None
    if "first name" in ask:
        return bool(infer_name_fields_from_transcript(normalized, need_first=True, need_last=False, slot_focus="first"))
    if "last name" in ask:
        return bool(infer_name_fields_from_transcript(normalized, need_first=False, need_last=True, slot_focus="last"))
    if "date of birth" in ask:
        return infer_dob_from_transcript(normalized) is not None
    if "mobile phone" in ask or "phone" in ask:
        return infer_phone_from_transcript(normalized) is not None
    if "email" in ask:
        return infer_email_from_transcript(normalized) is not None
    if "privacy policy" in ask or "electronic communications" in ask or "agree" in ask:
        return infer_yes_no_from_transcript(normalized) is not None
    return True


def _stt_confidence_triggers_reprompt(stt_confidence: Optional[float]) -> bool:
    """True when STT reports confidence in the 'brittle repair' range (deterministic scripts)."""
    if stt_confidence is None:
        return False
    # Only trigger for reasonable unit-interval inputs.
    if stt_confidence < 0.0 or stt_confidence > 1.0:
        return False
    # The [0.50, 0.51] range is reserved for our deterministic 'repair' triggers.
    return 0.50 <= stt_confidence <= 0.51

def _name_slot_focus_from_missing(missing_line: str) -> Optional[Literal["first", "last"]]:
    """Determine if the current question is specifically asking for first or last name."""
    ask = _ask_from_missing_line(missing_line).lower()
    if not ask:
        return None
    if "first name" in ask:
        return "first"
    if "last name" in ask:
        return "last"
    return None

def _use_deterministic_address_parsing() -> bool:
    return os.getenv("PATH2_DETERMINISTIC_ADDRESS", "1").strip().lower() not in (
        "0",
        "false",
        "no",
    )

VOICE_SCRIPT_OWNERSHIP_ASK = "Are you the sole owner, or is it jointly owned?"
VOICE_SCRIPT_OCCUPANCY_ASK = (
    "Is this your primary home, a second home, or an investment property?"
)
VOICE_SCRIPT_FOR_SALE_ASK = "Quick one — is the property currently listed for sale?"
OCCUPANCY_LOW_CONF_REPROMPT = (
    "Sorry, I didn't catch that. Say primary home, second home, or investment property."
)
OWNERSHIP_LOW_CONF_REPROMPT = (
    "Sorry, I didn't catch that. Are you the sole owner, or is it jointly owned?"
)
FOR_SALE_LOW_CONF_REPROMPT = (
    "Sorry, I didn't catch that. Is the property listed for sale — yes or no?"
)

CONFIDENCE_THRESHOLD = 0.85
ADDRESS_CONFIDENCE_THRESHOLD = 0.90
ADDRESS_CONFIDENCE_FIELDS = {"address_line_1", "city", "state", "zipcode"}
HIGH_CONFIDENCE_FIELDS = {
    "email_address", "mobile_phone_number", "date_of_birth",
    "agreed_to_privacy_policy", "agreed_to_commerce_communication",
}
DEFAULT_INTAKE_OWNERSHIP = "Sole owner"
DEFAULT_INTAKE_OCCUPANCY = "Primary property"
DEFAULT_INTAKE_FOR_SALE = False
DEFAULT_INTAKE_OTHER_INCOME = 0
DEFAULT_INTAKE_ANNUAL_PRE_TAX = 50000
DEFAULT_INTAKE_PRIMARY_SOURCE = "Employed full-time"
PARTIAL_DOB_REPROMPT = "Sorry, I didn't catch that. Could you say your full date of birth, including the year?"
PARTIAL_MOBILE_PHONE_REPROMPT = "Sorry, I didn't catch that. Could you say your full 10-digit phone number?"

_DOB_PARTIAL_SPOKEN_ORDINALS_ALT_RE = re.compile(
    r"\b(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|"
    r"eleventh|twelfth|thirteenth|fourteenth|fifteen|sixteenth|seventeenth|"
    r"eighteenth|nineteenth|twentieth|twenty-first|twenty-second|twenty-third|"
    r"twenty-fourth|twenty-fifth|twenty-sixth|twenty-seventh|twenty-eighth|"
    r"twenty-ninth|thirtieth|thirty-first)\b",
    re.IGNORECASE,
)
_DOB_PARTIAL_CARDINAL_DAY_WORDS_RE = re.compile(
    r"\b(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    r"thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|"
    r"twenty-one|twenty-two|twenty-three|twenty-four|twenty-five|twenty-six|"
    r"twenty-seven|twenty-eight|twenty-nine|thirty|thirty-one)\b",
    re.IGNORECASE,
)
