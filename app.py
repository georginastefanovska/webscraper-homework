import json
import time
from dataclasses import dataclass
from datetime import datetime, date
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st
from transformers import pipeline


# -----------------------------
# Config
# -----------------------------
@dataclass
class AppConfig:
    data_file: str = "webscraper.json"
    year_for_filter: int = 2023
    sentiment_model: str = "distilbert-base-uncased-finetuned-sst-2-english"
    sentiment_batch_size: int = 16


cfg = AppConfig()


# -----------------------------
# Data loading utilities
# -----------------------------
def load_scraped_data(path: str) -> Dict[str, Any]:
    file_path = Path(path)
    if not file_path.exists():
        st.error(f"Couldn't find '{path}'. Run your scraper first so this file exists.")
        st.stop()

    try:
        return json.loads(file_path.read_text(encoding="utf-8"))
    except Exception as ex:
        st.error(f"Failed to read '{path}': {ex}")
        st.stop()


def to_dataframe(items: Any, fallback_cols: Optional[List[str]] = None) -> pd.DataFrame:
    df = pd.DataFrame(items)
    if df.empty and fallback_cols:
        df = pd.DataFrame(columns=fallback_cols)
    return df


def parse_review_date(value: str) -> Optional[date]:
    """Parse date string safely; return None if invalid."""
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None

    # Try common formats first
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(raw, fmt).date()
        except ValueError:
            pass

    # Last-resort: let pandas try
    try:
        parsed = pd.to_datetime(raw, errors="coerce")
        if pd.isna(parsed):
            return None
        return parsed.date()
    except Exception:
        return None


def month_options_for_year(year: int) -> List[date]:
    return [date(year, m, 1) for m in range(1, 13)]


def month_label(d: date) -> str:
    return d.strftime("%b %Y")  # e.g., "Jan 2023"


# -----------------------------
# Sentiment utilities (Hugging Face)
# -----------------------------
@st.cache_resource
def get_sentiment_pipeline(model_name: str):
    # Returns POSITIVE / NEGATIVE
    return pipeline("sentiment-analysis", model=model_name)


def classify_sentiment(texts: List[str], clf, batch_size: int) -> List[Dict[str, Any]]:
    """Run sentiment in batches for speed."""
    outputs: List[Dict[str, Any]] = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        outputs.extend(clf(chunk))
    return outputs


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Scraped Data Viewer", layout="wide")
st.title("Scraped Data Viewer")

data = load_scraped_data(cfg.data_file)

products = data.get("products", [])
testimonials = data.get("testimonials", [])
reviews = data.get("reviews", [])

st.sidebar.header("Navigation")
section = st.sidebar.radio("Choose a section", ["Products", "Testimonials", "Reviews"])


# -----------------------------
# Products
# -----------------------------
if section == "Products":
    st.subheader("Products")
    df_products = to_dataframe(products, fallback_cols=["name", "price"])

    if df_products.empty:
        st.info("No products found in data.json.")
    else:
        st.dataframe(df_products, use_container_width=True)


# -----------------------------
# Testimonials
# -----------------------------
elif section == "Testimonials":
    st.subheader("Testimonials")
    df_testimonials = to_dataframe(testimonials, fallback_cols=["text", "author"])

    if df_testimonials.empty:
        st.info("No testimonials found in data.json.")
    else:
        st.dataframe(df_testimonials, use_container_width=True)


# -----------------------------
# Reviews + month filter + sentiment
# -----------------------------
else:
    st.subheader("Reviews (Filter by Month in 2023 + Sentiment)")

    df_reviews = to_dataframe(reviews, fallback_cols=["text", "date"])
    if df_reviews.empty:
        st.info("No reviews found in data.json.")
        st.stop()

    # Parse dates
    df_reviews["parsed_date"] = df_reviews["date"].apply(parse_review_date)
    df_reviews = df_reviews.dropna(subset=["parsed_date"]).copy()

    if df_reviews.empty:
        st.warning("Reviews exist, but none have a valid date format to filter by.")
        st.stop()

    # Month selector
    months = month_options_for_year(cfg.year_for_filter)
    chosen_month = st.select_slider(
        "Select a month in 2023",
        options=months,
        format_func=month_label,
        value=date(cfg.year_for_filter, 1, 1),
    )

    # Filter range
    start = chosen_month
    if chosen_month.month == 12:
        end = date(cfg.year_for_filter + 1, 1, 1)
    else:
        end = date(cfg.year_for_filter, chosen_month.month + 1, 1)

    filtered = df_reviews[(df_reviews["parsed_date"] >= start) & (df_reviews["parsed_date"] < end)].copy()
    filtered = filtered.sort_values("parsed_date")

    st.caption(f"Showing reviews from **{month_label(start)}**")

    if filtered.empty:
        st.info("No reviews in that month.")
        st.stop()

    st.write("### Sentiment Analysis (Hugging Face Transformers)")

    run_sentiment = st.checkbox("Run sentiment classification for the selected month", value=True)
    if run_sentiment:
        clf = get_sentiment_pipeline(cfg.sentiment_model)
        texts = filtered["text"].fillna("").astype(str).tolist()

        with st.spinner("Classifying sentiment..."):
            preds = classify_sentiment(texts, clf, cfg.sentiment_batch_size)

        # Map model outputs to Positive/Negative
        filtered["sentiment"] = [
            "Positive" if str(p.get("label", "")).upper() == "POSITIVE" else "Negative"
            for p in preds
        ]
        filtered["confidence"] = [float(p.get("score", 0.0)) for p in preds]

        show_df = filtered[["text", "date", "sentiment", "confidence"]].reset_index(drop=True)
        st.dataframe(show_df, use_container_width=True)

        pos = int((filtered["sentiment"] == "Positive").sum())
        neg = int((filtered["sentiment"] == "Negative").sum())

        c1, c2, c3 = st.columns(3)
        c1.metric("Total", len(filtered))
        c2.metric("Positive", pos)
        c3.metric("Negative", neg)

        st.write("### Sentiment Count (Bar Chart)")

        sentiment_counts = pd.DataFrame(
            {
                "Sentiment": ["Positive", "Negative"],
                "Count": [pos, neg],
            }
        )

        st.bar_chart(sentiment_counts.set_index("Sentiment"))

        with st.expander("Notes"):
            st.write(
                "- Model: `distilbert-base-uncased-finetuned-sst-2-english`\n"
                "- Output labels are mapped to **Positive**/**Negative**.\n"
                "- Confidence is the model score for the predicted label."
            )
    else:
        show_df = filtered[["text", "date"]].reset_index(drop=True)
        st.dataframe(show_df, use_container_width=True)

    # Optional: let user download filtered results
    st.write("")
    csv_bytes = filtered.drop(columns=["parsed_date"], errors="ignore").to_csv(index=False).encode("utf-8")
    st.download_button("Download filtered reviews as CSV", data=csv_bytes, file_name="filtered_reviews.csv", mime="text/csv")
