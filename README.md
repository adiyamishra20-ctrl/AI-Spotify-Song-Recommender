# 🎧 AI Spotify Song Picker

A content-based music recommendation project combining **BeautifulSoup metadata extraction, PyTorch representation learning, Scikit-learn similarity search, and optional Llama-generated explanations** in an interactive Streamlit app.

## Pipeline

```text
Music metadata
    ↓
BeautifulSoup extraction / CSV ingestion
    ↓
Feature engineering
    ↓
Standardized audio features
    ↓
PyTorch autoencoder
    ↓
Compact song embeddings
    ↓
Cosine similarity
    ↓
Mood/context recommendations
    ↓
Optional Llama explanations
    ↓
Streamlit interface
```

## Skills demonstrated

- Web/data extraction with **BeautifulSoup**
- Data preparation with **Pandas**
- Representation learning with a **PyTorch autoencoder**
- Content-based recommendation with **cosine similarity**
- Interactive deployment with **Streamlit**
- Optional natural-language explanations with **Llama 3.1**

## Dataset

The included `data/sample_tracks.csv` is a **synthetic 200-row demo dataset** so the repository can run without external downloads.

For the full experiment, replace it with a public Spotify tracks dataset containing audio features. One suitable option is:

https://www.kaggle.com/datasets/saichaitanyareddyai/spotify-tracks-dataset-audio-features

That dataset contains 114,000 Spotify tracks with metadata and audio features including danceability, energy, acousticness, instrumentalness, speechiness, valence and tempo.

## Setup

### 1. Create a virtual environment

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

### 2. Install packages

```bash
pip install -r requirements.txt
```

### 3. Train the autoencoder

```bash
python scripts/train_model.py
```

### 4. Benchmark the learned embeddings

```bash
python scripts/benchmark.py
```

The benchmark calculates the actual mood-retrieval improvement of the autoencoder over raw standardized features. **No performance number is hard-coded.**

### 5. Launch the application

```bash
streamlit run app/streamlit_app.py
```

## Optional Llama explanations

The app supports Hugging Face Inference API. Set your Hugging Face token before running Streamlit:

PowerShell:

```powershell
$env:HF_TOKEN="your_token_here"
$env:HF_MODEL="meta-llama/Llama-3.1-8B-Instruct"
```

If no token is supplied, the app uses a deterministic explanation based on the similarity score.

## BeautifulSoup component

`scripts/scrape_metadata.py` parses static HTML/exported music pages containing `data-track-name`, `data-artists`, `data-genre` and `data-mood` attributes.

Spotify's current developer platform provides an official Web API for retrieving Spotify content metadata. This project therefore avoids relying on a fragile direct scrape of Spotify's JavaScript-rendered pages.

## Project structure

```text
ai-spotify-song-recommender/
├── app/
│   └── streamlit_app.py
├── data/
│   ├── README.md
│   └── sample_tracks.csv
├── models/
│   └── README.md
├── notebooks/
│   └── 01_eda_and_recommendation.ipynb
├── outputs/
├── scripts/
│   ├── benchmark.py
│   ├── scrape_metadata.py
│   └── train_model.py
├── src/
│   ├── __init__.py
│   ├── features.py
│   └── model.py
├── .gitignore
├── requirements.txt
└── README.md
```

## Important for your CV

The repository deliberately does **not** fabricate a 2,000+ track dataset or a 15% improvement. Once you replace the demo data with the real 2,000+ track dataset and run the benchmark, use the measured result in your CV.

## License

Code: MIT. Follow the license/terms of whichever external dataset you use.
