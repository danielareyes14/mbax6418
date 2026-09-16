"""Render the self-contained dashboard to a PNG screenshot for the README."""
import os

from playwright.sync_api import sync_playwright

ROOT = os.path.dirname(os.path.abspath(__file__))
HTML = os.path.join(ROOT, "Sentiment_Emotion_Dashboard.html")
OUT = os.path.join(ROOT, "dashboard_screenshot.png")

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 1460, "height": 2400})
    page.goto("file:///" + HTML.replace("\\", "/"))
    page.wait_for_timeout(1200)  # let JS render
    # Capture the full scrollable page.
    page.screenshot(path=OUT, full_page=False, clip=None)
    browser.close()

print("Saved", OUT, os.path.getsize(OUT), "bytes")
