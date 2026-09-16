"""Step 1 spot-check: obvious positive/negative/neutral short reviews."""
import openai

from prompt import SYSTEM_PROMPT, build_user_prompt, parse_llm_response, NRC_EMOTIONS

client = openai.OpenAI(base_url="http://dobolyi.com:9000/v1", api_key="6418")
model = "DeepSeek-V4-Flash-0731"

cases = [
    # (title, text, expected sentiment)
    ("Great gift", "Having Amazon money is always good.", "POSITIVE"),
    ("Card never arrives", "I ordered last month and still nothing came.", "NEGATIVE"),
    ("Total scam", "The card was already redeemed when I got it. Do not buy!", "NEGATIVE"),
    ("Perfect", "Exactly what I wanted, worked instantly.", "POSITIVE"),
    ("Okay", "It's fine, nothing special about it.", "NEUTRAL"),
    ("Directions confusing", "The wallet setup is confusing and took a long time.", "NEGATIVE"),
]

for title, text, expected in cases:
    user_msg = build_user_prompt(title, text, mode="three")
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": SYSTEM_PROMPT},
                  {"role": "user", "content": user_msg}],
        temperature=0, max_tokens=1024,
    )
    content = (resp.choices[0].message.content or "").strip()
    sent, emo = parse_llm_response(content, "three", valid_emotions=NRC_EMOTIONS)
    ok = "✓" if sent == expected else "✗"
    print(f"{ok} expected={expected:<8} llm={str(sent):<8} emotion={str(emo):<6} "
          f"| {title[:30]}")
