# Reddit Moderation Dashboard

Multi-dimensional risk assessment system for content moderation in mental health and support communities. Combines toxicity detection, polarity analysis, and controversy scoring to help moderators prioritize review queues without automating removal decisions.

## Quick Start

```bash
git clone https://github.com/YOUR_USERNAME/reddit-moderation-dashboard.git
cd reddit-moderation-dashboard
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```


### Run

```bash
streamlit run app2.py
```

Dashboard opens at `http://localhost:8501`


## Project Structure

```
reddit-moderation-dashboard/
├── app2.py                 # Main Streamlit application
├── fetcher.py              # Reddit API wrapper (PRAW)
├── toxicity_detector.py    # toxic-bert integration
├── polarity.py             # Vote + sentiment polarity
├── controversy.py          # Multi-component controversy detection
├── requirements.txt        # Dependencies
└── README.md              # This file
```

## Tech Stack

**Core**: Python 3.8+, Streamlit, PRAW  
**ML Models**: HuggingFace Transformers (toxic-bert, DistilBERT-SST2)  
**Visualization**: Plotly, Pandas  
**Processing**: PyTorch (GPU optional)

## Features

- **Multi-signal risk scoring**: Toxicity + Polarity + Controversy  
- **Configurable thresholds**: Adjust sensitivity per community  
- **Moderator workflow**: Track actions (Pending, Needs Action, Solved)  
- **Real-time configuration**: Change weights without re-fetching  
- **CSV exports**: All data, flagged posts, action items  
- **Privacy-preserving**: Local processing, no external APIs

## Usage

1. Enter subreddit name (no `r/` prefix)
2. Set posts to fetch (1-50)
3. Click "Fetch & Analyze Posts"
4. Review flagged posts in expandable cards
5. Mark actions: Solved / Needs Action
6. Export CSVs as needed

**Processing time**: ~5 seconds/post (CPU), ~1 second/post (GPU)

## Scoring System

**Toxicity** (0-1): Explicit harm (threats, hate, harassment)  
**Polarity** (0-1): Emotional complexity + vote disagreement  
**Controversy** (0-1): Community debate + polarizing language  
**Aggregate Risk** = 0.6×T + 0.2×P + 0.2×C (customizable)

## Performance

**Validation** (150 posts, threshold 0.4):  
- Precision: 0.79  
- Recall: 0.84  
- F1: 0.81

**Multi-signal outperforms single metrics**:  
- Toxicity only: F1 0.66  
- Polarity only: F1 0.70  
- Controversy only: F1 0.69

## Troubleshooting

**Invalid credentials**: Check client ID/secret, verify "script" app type  
**Rate limit**: Reduce fetch limit to ≤20 posts  
**Slow processing**: Normal on CPU (4-5s/post), use GPU for real-time  
**Models not downloading**: Check internet, models cache in `~/.cache/huggingface/`

## Research Context

Developed for dissertation research on cognitive strain detection in online support communities (Colorado School of Mines, HCI).

## Ethics

⚠️ **Human-in-the-loop only**: Do not automate content removal or banning  
✅ **Use for**: Moderator triage, community health analysis, research

## License

MIT License - see LICENSE file
