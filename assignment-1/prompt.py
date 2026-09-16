"""
Reusable LLM prompt for Amazon review classification (MBAX 6418 Assignment 1).

Step 1 requirement: a prompt that takes a review's TITLE and TEXT and asks the
LLM to classify it, returning a clear, machine-readable answer. It must NOT see
the star rating.

Design:
  - binary mode (Step 1/2): POSITIVE / NEGATIVE  (>=4 star is the "correct"
    answer used only for scoring AFTERWARDS — never shown to the model)
  - three-class mode (Step 6): POSITIVE / NEUTRAL / NEGATIVE
  - both modes also ask for a primary emotion so the same prompt covers Step 5.
  - Output is strict JSON so it can be parsed programmatically; the model is
    told to answer with the JSON only.
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


SYSTEM_PROMPT = (
    "You are an expert reviewer of Amazon product reviews. You read the "
    "review's TITLE and TEXT that a customer wrote and you determine (1) the "
    "overall sentiment of the review and (2) the single dominant emotion "
    "expressed. Your judgment must be based purely on the language of the "
    "review. Respond with ONLY a single JSON object, no prose, no markdown, "
    "no extra keys."
)


def build_user_prompt(title: str, text: str, mode: str = "binary",
                      emotion_based: str = "") -> str:
    """Build the user message for one review.

    mode: 'binary' -> POSITIVE/NEGATIVE, 'three' -> POSITIVE/NEUTRAL/NEGATIVE.
    emotion_based: optional comma-separated hint list for the emotion choice;
                   empty means the model chooses from the standard NRC set.
    """
    allowed = SENTIMENTS[mode]
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


def parse_llm_response(content, mode: str, valid_emotions=None):
    """Parse the model's free-text answer into (sentiment, emotion).

    Tolerates prose around the JSON and normalises to the allowed sets.
    Returns (sentiment, emotion); either may be None if not recoverable.
    """
    sentiment, emotion = None, None
    text = (content or "").strip()

    # Extract a JSON object if present.
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            data = json.loads(m.group(0))
        except Exception:
            data = {}
        if isinstance(data, dict):
            sentiment = str(data.get("sentiment", "")).upper()
            emotion = str(data.get("emotion", "")).lower()

    # If no JSON, fall back to scanning the raw text for the expected tokens.
    if sentiment not in SENTIMENTS[mode]:
        for s in SENTIMENTS[mode]:
            if s in text.upper():
                sentiment = s
                break
    if valid_emotions and emotion not in valid_emotions:
        for e in valid_emotions:
            if e in text.lower():
                emotion = e
                break

    return sentiment or None, emotion or None
