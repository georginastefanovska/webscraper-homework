import json
from dataclasses import dataclass
from datetime import datetime, date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
    # Lightweight model (Render free / low RAM friendly)
    sentiment_model: str = "philschmid/tiny-bert-sst2-distilled"
    sentiment_batch_size: int = 8
    # Safety limit so you don't overload tiny instances
    max_reviews_to_analyze: int = 120


cfg = AppConfig()


# -----------------------------
# Helpers
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


def parse_review_date(value: Any) -> Optional[date]:
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None

    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(raw, fmt).date()
        except ValueError:
            pass

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
    return d.strftime("%b %Y")


# -----------------------------
# Hugging Face Sentiment
# -----------------------------
@st.cache_resource
def get_sentiment_pipeline(model_name: str):
    return pipeline("sentiment-analysis", model=model_name)


def classify_sentiment(texts: List[str], clf, batch_size: int) -> List[Dict[str, Any]]:
    outputs: List[Dict[str, Any]] = []
    for i in range(0, len(texts), batch_size):
        outputs.extend(clf(texts[i:i + batch_size]))
    return outputs


@st.cache_data(show_spinner=False)
def cached_sentiment(texts: Tuple[str, ...], model_name: str, batch_size: int) -> List[Dict[str, Any]]:
    clf = get_sentiment_pipeline(model_name)
    return classify_sentiment(list(texts), clf, batch_size)


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Web Scraper + Sentiment App", layout="wide")
st.title("Web Scraper + Sentiment Analysis App")

data = load_scraped_data(cfg.data_file)

products = data.get("products", [])
testimonials = data.get("testimonials", [])
reviews = data.get("reviews", [])

st.sidebar.header("Navigation")
section = st.sidebar.radio("Choose a section", ["Products", "Testimonials", "Reviews"])

st.sidebar.divider()
st.sidebar.caption("Data source file:")
st.sidebar.code(cfg.data_file)


# -----------------------------
# Products
# -----------------------------
if section == "Products":
    st.subheader("Products")
    st.caption("This section displays all scraped products (name + price).")

    df_products = to_dataframe(products, fallback_cols=["name", "price"])
    if df_products.empty:
        st.info("No products found in the data file.")
    else:
        st.dataframe(df_products, use_container_width=True)


# -----------------------------
# Testimonials
# -----------------------------
elif section == "Testimonials":
    st.subheader("Testimonials")
    st.caption("This section displays all scraped testimonials (text + author).")

    df_testimonials = to_dataframe(testimonials, fallback_cols=["text", "author"])
    if df_testimonials.empty:
        st.info("No testimonials found in the data file.")
    else:
        st.dataframe(df_testimonials, use_container_width=True)


# -----------------------------
# Reviews + Month filter + Sentiment + Bar chart
# -----------------------------
else:
    st.subheader("Reviews")
    st.caption(
        "Select a month in 2023 to filter reviews. Then run sentiment analysis (Positive/Negative) "
        "using a lightweight Hugging Face Transformer model."
    )

    df_reviews = to_dataframe(reviews, fallback_cols=["text", "date"])
    if df_reviews.empty:
        st.info("No reviews found in the data file.")
        st.stop()

    # Parse dates and keep valid ones
    df_reviews["parsed_date"] = df_reviews["date"].apply(parse_review_date)
    df_reviews = df_reviews.dropna(subset=["parsed_date"]).copy()

    if df_reviews.empty:
        st.warning("Reviews exist, but none have a valid date format to filter by.")
        st.stop()

    # Month selector
    months = month_options_for_year(cfg.year_for_filter)
    chosen_month = st.select_slider(
        "Select a month",
        options=months,
        format_func=month_label,
        value=date(cfg.year_for_filter, 1, 1),
    )

    # Date range for month
    start = chosen_month
    if chosen_month.month == 12:
        end = date(cfg.year_for_filter + 1, 1, 1)
    else:
        end = date(cfg.year_for_filter, chosen_month.month + 1, 1)

    # Filter
    filtered = df_reviews[(df_reviews["parsed_date"] >= start) & (df_reviews["parsed_date"] < end)].copy()
    filtered = filtered.sort_values("parsed_date")

    if filtered.empty:
        st.info("No reviews in that month.")
        st.stop()

    # Show filtered reviews first
    st.write("### Filtered Reviews")
    st.write(f"Found **{len(filtered)}** reviews for **{month_label(start)}**.")
    st.dataframe(filtered[["text", "date"]].reset_index(drop=True), use_container_width=True)

    st.divider()
    st.write("### Sentiment Analysis")

    # Limit analyzed reviews (Render safety)
    max_n = min(cfg.max_reviews_to_analyze, len(filtered))
    analyze_n = st.slider("How many reviews to analyze?", 1, max_n, min(50, max_n))

    colA, colB = st.columns([1, 2])
    with colA:
        run = st.button("Run Sentiment", type="primary")
    with colB:
        st.caption(f"Model: `{cfg.sentiment_model}` (lightweight for low-memory deployment)")

    if run:
        filtered_for_ai = filtered.head(analyze_n).copy()
        texts_tuple = tuple(filtered_for_ai["text"].fillna("").astype(str).tolist())

        with st.spinner("Running sentiment model..."):
            preds = cached_sentiment(texts_tuple, cfg.sentiment_model, cfg.sentiment_batch_size)

        filtered_for_ai["sentiment"] = [
            "Positive" if str(p.get("label", "")).upper() == "POSITIVE" else "Negative"
            for p in preds
        ]
        filtered_for_ai["confidence"] = [float(p.get("score", 0.0)) for p in preds]

        pos = int((filtered_for_ai["sentiment"] == "Positive").sum())
        neg = int((filtered_for_ai["sentiment"] == "Negative").sum())

        c1, c2, c3 = st.columns(3)
        c1.metric("Analyzed", len(filtered_for_ai))
        c2.metric("Positive", pos)
        c3.metric("Negative", neg)

        st.write("### Sentiment Count (Bar Chart)")
        sentiment_counts = pd.DataFrame({"Sentiment": ["Positive", "Negative"], "Count": [pos, neg]})
        st.bar_chart(sentiment_counts.set_index("Sentiment"))

        st.write("### Labeled Reviews")
        st.dataframe(
            filtered_for_ai[["text", "date", "sentiment", "confidence"]].reset_index(drop=True),
            use_container_width=True,
        )

        st.download_button(
            "Download analyzed reviews as CSV",
            data=filtered_for_ai.drop(columns=["parsed_date"], errors="ignore").to_csv(index=False).encode("utf-8"),
            file_name=f"reviews_{month_label(start).replace(' ', '_')}.csv",
            mime="text/csv",
        )

