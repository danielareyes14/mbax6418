"""Verify the interactive filter shows exactly the right rows + live count."""
import os

from playwright.sync_api import sync_playwright

ROOT = os.path.dirname(os.path.abspath(__file__))
HTML = os.path.join(ROOT, "Sentiment_Emotion_Dashboard.html")

with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page(viewport={"width": 1460, "height": 1200})
    pg.goto("file:///" + HTML.replace("\\", "/"))
    pg.wait_for_timeout(1200)

    # Baseline: figure out how many mismatched rows exist from the data itself.
    total_mismatch = pg.evaluate(
        "D.rows.filter(r => r.correct === false).length")
    total_match = pg.evaluate(
        "D.rows.filter(r => r.correct === true).length")
    total = pg.evaluate("D.rows.length")
    print(f"data-derived: matched={total_match}, mismatched={total_mismatch}, total={total}")
    print(f"check sums: {total_match + total_mismatch} == {total} ? "
          f"{total_match + total_mismatch == total}")

    # Click the MISMATCHED segment button.
    pg.click("#segResult button[data-v='mis']")
    pg.wait_for_timeout(300)
    shown = pg.text_content("#showCount")
    tbody_rows = pg.eval_on_selector_all("#tbody tr", "e => e.length")
    print(f"after MISMATCH filter: showCount={shown}, tbody rows={tbody_rows}")

    # Every displayed row should be a mismatch.
    all_mis = pg.evaluate(
        "[...document.querySelectorAll('#tbody tr')].every(tr => tr.textContent.includes('✗'))")
    print(f"all displayed rows are mismatches: {all_mis}")

    # Search 'gift' on all, then check count shrinks sensibly
    pg.click("#segResult button[data-v='all']")
    pg.fill("#q", "gift")
    pg.wait_for_timeout(300)
    print("gift search showCount:", pg.text_content("#showCount"))

    # Mode filter -> binary only
    pg.fill("#q", "")
    pg.select_option("#fMode", "binary")
    pg.wait_for_timeout(300)
    print("binary-only showCount (expect 100):", pg.text_content("#showCount"))
    b.close()
