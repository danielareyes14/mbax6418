"""Live check: new binary prompt against the LLM endpoint (a few examples)."""
import openai

from prompt import build_binary_prompt, parse_llm_response, SYSTEM_PROMPT

client = openai.OpenAI(base_url="http://dobolyi.com:9000/v1", api_key="6418")
model = "DeepSeek-V4-Flash-0731"

cases = [
    ("New mom loves it", "Didn't have to go anywhere got it the next day", "POSITIVE"),
    ("Card never arrives", "I ordered last month and still nothing came.", "NEGATIVE"),
    ("Total scam", "The card was already redeemed when I got it. Do not buy!", "NEGATIVE"),
    ("Great gift", "Having Amazon money is always good.", "POSITIVE"),
    ("Directions confusing", "The wallet setup is confusing and took a long time.", "NEGATIVE"),
]

for title, text, expected in cases:
    q = build_binary_prompt(title, text)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": q}],   # user-only, no system
        temperature=0, max_tokens=64,
    )
    content = (resp.choices[0].message.content or "").strip()
    sent, _ = parse_llm_response(content, "binary")
    ok = "✓" if sent == expected else "✗"
    print(f"{ok} raw={content!r:28} parsed={sent!s:<8} expected={expected} | {title}")
    # Show whether the model answered with extra prose or just the word.
    bare = content.strip().upper() in ("POSITIVE", "NEGATIVE")
    if not bare:
        print(f"   (model added prose; parser recovered {sent})")
