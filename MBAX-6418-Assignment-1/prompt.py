"""
Reusable LLM prompt for Amazon review classification (MBAX 6418 Assignment 1).

Step 1 requirement: a prompt that takes a review's TITLE and TEXT and asks the
LLM to classify it, returning a clear, machine-readable answer. It must NOT see
the star rating.

Two prompt modes:
  - BINARY (Steps 1/2): the fixed sentiment prompt — asks for exactly one of
    POSITIVE / NEGATIVE, as a single word. Uses ONLY title + text.
  - THREE-CLASS + EMOTION (Steps 5/6): asks for POSITIVE / NEUTRAL / NEGATIVE
    AND the primary emotion, returned as a strict JSON object (so the same
    prompt covers both Step 5 emotion and Step 6 three-class).
"""
import json
import re

# The eight NRC emotion categories from the assignment.
NRC_EMOTIONS = [
    "anger", "anticipation", "disgust", "fear",
    "joy", "sadness", "surprise", "trust",
]
SENTIMENTS = {
    "binary": ["NEGATIVE", "POSITIVE"],
    "three": ["NEGATIVE", "NEUTRAL", "POSITIVE"],
}


# ---------------------------------------------------------------------------
# BINARY prompt (the fixed, spec'd wording)
# ---------------------------------------------------------------------------
def build_binary_prompt(title: str, text: str) -> str:
    """Build the fixed Step 1/2 sentiment prompt for one review.

    Returns the exact prompt template with the review's title and text filled
    in. The model is asked to reply with a single word only: POSITIVE or
    NEGATIVE.
    """
    title_s = (title or "").strip() or "(no title)"
    text_s = (text or "").strip() or "(empty)"
    return (
        "Classify the sentiment of the Amazon review below as either POSITIVE "
        "or NEGATIVE.\n"
        "\n"
        "Use ONLY the review title and review text to make your decision. Do "
        "not use or infer the sentiment from any star rating or other rating "
        "information.\n"
        "\n"
        "Guidelines:\n"
        "- POSITIVE: The reviewer is generally satisfied, happy, or expresses "
        "a favorable opinion.\n"
        "- NEGATIVE: The reviewer is generally dissatisfied, unhappy, or "
        "expresses an unfavorable opinion.\n"
        "- If the title and review text conflict, prioritize the overall "
        "meaning of the review text.\n"
        "- For very short reviews, classify based on the sentiment that is "
        "explicitly expressed.\n"
        "- Always choose exactly one label.\n"
        "\n"
        "Return ONLY one of these two words:\n"
        "POSITIVE\n"
        "NEGATIVE\n"
        "\n"
        f"Review Title:\n{title_s}\n"
        "\n"
        f"Review Text:\n{text_s}"
    )


# ---------------------------------------------------------------------------
# THREE-CLASS + EMOTION system / user prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are an expert reviewer of Amazon product reviews. You read the "
    "review's TITLE and TEXT that a customer wrote and you determine (1) the "
    "overall sentiment of the review and (2) the single dominant emotion "
    "expressed. Your judgment must be based purely on the language of the "
    "review; you are never given a star rating. Respond with ONLY a single "
    "JSON object, no prose, no markdown, no extra keys."
)


def build_three_prompt(title: str, text: str, emotion_based: str = "") -> str:
    """Build the three-class + emotion user prompt for one review."""
    allowed = SENTIMENTS["three"]
    if emotion_based:
        emo_allowed = emotion_based
    else:
        emo_allowed = ", ".join(NRC_EMOTIONS)

    title_s = (title or "").strip() or "(no title)"
    text_s = (text or "").strip() or "(empty)"

    return (
        f"Review title: {title_s}\n"
        f"Review text: {text_s}\n\n"
        f"Return a JSON object with exactly these two keys:\n"
        f'  - "sentiment": one of {json.dumps(allowed)}\n'
        f'  - "emotion": one primary emotion from this list: {emo_allowed}\n\n'
        f"Do not mention the star rating (you have not been given it). "
        f"Answer with the JSON object only."
    )


def build_user_prompt(title: str, text: str, mode: str = "binary",
                      emotion_based: str = "") -> str:
    """Dispatch to the right prompt for the requested mode."""
    if mode == "binary":
        return build_binary_prompt(title, text)
    return build_three_prompt(title, text, emotion_based)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
def parse_llm_response(content, mode: str, valid_emotions=None):
    """Parse the model's answer into (sentiment, emotion).

    - binary mode: expected answer is a single word POSITIVE/NEGATIVE.
    - three mode: expected answer is a JSON object with "sentiment"+"emotion".

    Tolerates stray prose; normalises to the allowed sets.
    Returns (sentiment, emotion); either may be None if not recoverable.
    """
    sentiment, emotion = None, None
    text = (content or "").strip()

    if mode == "three":
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                data = json.loads(m.group(0))
            except Exception:
                data = {}
            if isinstance(data, dict):
                sentiment = str(data.get("sentiment", "")).upper()
                emotion = str(data.get("emotion", "")).lower()
        # Fall back: search for the label tokens anywhere in the answer.
        for s in SENTIMENTS["three"]:
            if sentiment != s and re.search(rf"\b{s}\b", text.upper()):
                sentiment = s
        if valid_emotions:
            for e in valid_emotions:
                if emotion != e and e in text.lower():
                    emotion = e
        return sentiment or None, emotion or None

    # binary: single-word (or tokenised) answer.
    up = text.upper()
    # First look for an exact-lines / standalone POSITIVE|NEGATIVE token.
    tokens = re.findall(r"\b(POSITIVE|NEGATIVE)\b", up)
    if tokens:
        # If both appear, fall back to JSON-ish preference; otherwise take it.
        if "POSITIVE" in tokens and "NEGATIVE" in tokens:
            for s in ["POSITIVE", "NEGATIVE"]:
                if up.strip() == s:
                    sentiment = s
                    break
            if sentiment is None:
                sentiment = tokens[0]  # ambiuous; pick first
        else:
            sentiment = tokens[0]
    return sentiment or None, emotion or None
