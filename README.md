# webscraper-homework
# Web Scraper and Sentiment Analysis App

An integrated Python solution for automated data extraction and interactive sentiment analysis. This project uses a modular architecture to scrape e-commerce data including products, testimonials, and reviews, then visualizes it through a Streamlit dashboard powered by Hugging Face Transformers.

---

## Features

* **Multi-Source Scraper (main.py)**:
    * **HTML Scraping**: Extracts product names and prices using BeautifulSoup.
    * **REST API Extraction**: Fetches testimonial data from internal API endpoints.
    * **GraphQL Integration**: Pulls user reviews via structured GraphQL queries.
* **Interactive Dashboard (app.py)**:
    * **Sentiment Analysis**: Uses the tiny-bert-sst2-distilled model for efficient text classification.
    * **Time-Based Filtering**: Allows users to filter reviews by specific months within a target year.
    * **Data Visualization**: Generates metrics and bar charts to compare positive versus negative sentiment.
    * **Export Capabilities**: Analyzed data can be downloaded directly as a CSV file.

---

## Architecture

1.  **Extraction Layer**: main.py handles the connection logic and session management, saving results into a webscraper.json file.
2.  **Analysis Layer**: app.py loads the JSON data, processes date formats, and runs the sentiment pipeline using Hugging Face.

---

## Installation

1.  **Clone the repository**:
    ```bash
    git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
    cd your-repo-name
    ```

2.  **Install the required dependencies**:
    ```bash
    pip install requests beautifulsoup4 pandas streamlit transformers torch
    ```

---

## Usage

### 1. Run the Scraper
First, execute the scraper to collect the data from the target website. This will generate the webscraper.json file.
```bash
python main.py
