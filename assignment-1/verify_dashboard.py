"""Verify the dashboard renders real content (not blank/error) + pixel variance."""
import os

from playwright.sync_api import sync_playwright

ROOT = os.path.dirname(os.path.abspath(__file__))
HTML = os.path.join(ROOT, "Sentiment_Emotion_Dashboard.html")

with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page(viewport={"width": 1460, "height": 2400})
    errors = []
    pg.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
    pg.on("pageerror", lambda e: errors.append(str(e)))
    pg.goto("file:///" + HTML.replace("\\", "/"))
    pg.wait_for_timeout(1500)

    # Confirm the data-loaded + mounted DOM is populated.
    checks = {
        "kpix_count": pg.eval_on_selector_all("#kpix .kpi", "els => els.length"),
        "rows_rendered": pg.eval_on_selector_all("#tbody tr", "els => els.length"),
        "showCount": pg.text_content("#showCount"),
        "totCount": pg.text_content("#totCount"),
        "matrix_cells": pg.eval_on_selector_all("#cmpan .cm td", "els => els.length"),
        "truepred_bars": pg.eval_on_selector_all("#truepred .bar-row", "els => els.length"),
        "emollm_bars": pg.eval_on_selector_all("#emollm .bar-row", "els => els.length"),
        "emonrc_bars": pg.eval_on_selector_all("#emonrc .bar-row", "els => els.length"),
    }
    print("DOM checks:", checks)
    print("console/page errors:", errors[:8] if errors else "none")

    # Pixel variance: sample the page; blank page == near-uniform pixels.
    data = pg.screenshot(full_page=True)
    b.close()

from PIL import Image
import io, statistics
img = Image.open(io.BytesIO(data)).convert("RGB")
px = list(img.resize((60, 40)).getdata())
lums = [round(0.299*r+0.587*g+0.114*b) for r, g, b in px]
print("distinct colors sampled:", len(set(lums)))
print("luminance std-dev:", round(statistics.pstdev(lums), 2))
print("=> page is populated/colorful" if len(set(lums)) > 20 else "=> page looks BLANK/flat")
