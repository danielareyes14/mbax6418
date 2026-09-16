# MBAX 6418 — Assignment 1
# Sentiment & Emotion Classification of Amazon Reviews

**Daniela Reyes** · University of Colorado

## 📄 Full report

The complete assignment report — data source, approach, dashboard, the four findings with
numbers, and reproduction steps — lives in the project folder:

➡️ **[`MBAX-6418-Assignment-1/README.md`](MBAX-6418-Assignment-1/README.md)**

## 📂 Project structure

```
MBAX-6418-Assignment-1/
├── README.md                              ← the report
├── prompt.py                              ← the reusable structured prompt
├── llm_classify.py                        ← the scoring script (runs the reviews)
├── nrc_emotion.py                         ← the word-list script (adds emotions)
├── spot_check.py                          ← Step-1 sanity check
├── 01_prepare_data.py                     ← load + clean + build ground truth
├── make_dashboard.py                      ← the dashboard generator
├── first_100_results.csv                  ← Step-2 binary scored output
├── balanced_results.csv                   ← Step-6 balanced scored output
├── raw_output/balanced_run_raw_output.jsonl
├── dashboard/Sentiment_Emotion_Dashboard.html
└── screenshots/dashboard_screenshot.png
```

<!-- This root README is a short index. The full assignment report is
MBAX-6418-Assignment-1/README.md and renders when opened. -->
