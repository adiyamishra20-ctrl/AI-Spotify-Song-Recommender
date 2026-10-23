🎵 Live Demo
Launch Song Compass →

Discover songs by mood, explore reference-song recommendations, and view annotation-based explanations.

# Song Compass

A mood-first music recommendation prototype with optional reference-song
ranking and local Llama explanations.

## Features

- BeautifulSoup pipeline collecting 3,500 chart entries from 1990–2024
- 3,211 provisional song candidates after cleaning and deduplication
- Historical Last.fm dataset and API metadata enrichment
- 1,800 candidates with tags
- 1,085 songs with recognized genre and mood tags in the app's model catalog
- Personal mood reviews: Supported, Unsupported, and Uncertain
- Scikit-learn cosine-similarity recommendations
- PyTorch autoencoder: 61 input features compressed into 16 dimensions
- Local Llama 3.2 explanations through Ollama
- Streamlit interface and Spotify search links

## Local setup

From the project folder, with the virtual environment activated:

    python -m pip install -r requirements.txt

Install Ollama separately, then download the explanation model:

    ollama pull llama3.2:3b

Keep Ollama running and launch the app:

    python -m streamlit run app.py

The app requires the generated data files; installing packages alone
does not recreate the catalog.

## Main generated inputs

- data/processed/baseline_songs.csv
- data/processed/baseline_features.csv
- data/processed/song_embeddings.csv
- reports/reference_mood_review.csv

The saved PyTorch checkpoint is models/autoencoder.pt.
The app reads precomputed embeddings rather than running the encoder live.

## Recommendation behavior

Mood is required. Genre and reference song are optional.

Personal Unsupported and Uncertain judgments exclude a song for that mood.
Reviewed-only mode includes only Supported song–mood pairs.
Dataset-only matches are explicitly labeled as unverified.

Without a reference song, eligible songs are ordered by review status,
then alphabetically. No learned mood-only ranking is claimed.

With a reference song, raw-feature similarity is the default.
The app also offers the original uncentered autoencoder embeddings
as an experimental option.

Llama explanations are available for reference-song matches.
Without a reference song, the app displays direct mood evidence.
A template fallback is used if local generation fails.

## Evaluation

Task: next-song ranking within fixed dataset-tagged mood pools,
using the Cornell radio-playlist dataset.

Metric: NDCG@10 averaged within each reference song, then across references.
Each event treats the observed next song as its relevant item.
Other candidates are unobserved, not proven irrelevant.

Development results:
- Raw features: 0.038410
- Original autoencoder embeddings: 0.036699
- Centered autoencoder embeddings: 0.041052 (+6.88%)
- Best tested mixed hybrid, 25% embeddings: 0.040540 (+5.55%)

Held-out playlist results:
- Raw features: 0.038104
- Centered autoencoder embeddings: 0.032220
- Relative change: -15.44%

The development gains did not generalize to the held-out playlists.
No positive held-out recommendation improvement is claimed.

The test evaluated 8,941 eligible events across 551 reference songs,
within a 975-song candidate universe. It excluded next-song events
that did not satisfy the fixed query mood.

This measures conditional radio-sequence ranking, not independently
verified mood accuracy or user satisfaction. Personal mood overrides
were not applied in this playlist evaluation.

## Model results

Validation reconstruction loss:
- Autoencoder: 0.1042
- Constant-feature predictor: 0.3311

Lower reconstruction loss is not evidence of better recommendation relevance.

## Limitations

- Genre and mood tags are incomplete and noisy.
- Matching by titles and artist credits may not identify recording versions.
- Only a small subset of mood annotations has personal review.
- Existing embeddings retain the original dataset mood features.
- Spotify links perform searches; availability and recording identity
  are not verified.
- Llama can produce unsupported statements despite grounding instructions.
- This is a local deployment. Public hosting requires a reachable
  Llama service; cloud hosting cannot use a laptop's localhost.

## Credentials

Store API keys in .env and keep that file out of version control.
Local Ollama generation does not require an API key.
The app does not need metadata API keys to serve the generated catalog.

## Data attribution

Chart tables: Wikipedia pages for Billboard Year-End Hot 100 singles.
Tag enrichment: Last.fm API and Cornell Playlist Dataset, yes_complete.

Cornell dataset documentation:
https://www.cs.cornell.edu/~shuochen/lme/data_page.html

References:
- Chen, Xu, and Joachims. Multi-space Probabilistic Sequence Modeling, 2013.
- Moore, Chen, Turnbull, and Joachims. Learning to Embed Songs and Tags
  for Playlists Prediction. ISMIR, 2012.
- Chen, Xu, and Joachims. Multi-space Probabilistic Sequence Modeling.
  IJCAI, 2013.

Consult source terms before redistributing data or using it commercially.
The supplied Cornell README lists the third reference as ISMIR 2013;
verify the bibliographic venue before using that citation formally.