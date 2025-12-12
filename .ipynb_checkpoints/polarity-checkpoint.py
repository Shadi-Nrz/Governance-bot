from typing import Dict
from transformers import pipeline
import torch

# Optional sentiment model (DistilBERT-SST2)
sentiment_model = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    tokenizer="distilbert-base-uncased-finetuned-sst-2-english",
    device=0 if torch.cuda.is_available() else -1
)


def compute_vote_ratio(upvotes: int, downvotes: int) -> float:
    """
    Computes normalized vote ratio between 0 and 1.
    """
    total = upvotes + downvotes
    if total == 0:
        return 0.5  # neutral if no votes
    return upvotes / total


def polarity_from_votes(ratio: float) -> float:
    """
    Converts vote ratio into polarity score P1 in [0,1].
    P1 is highest when votes are evenly split (0.5)
    """
    return 1 - abs(0.5 - ratio) * 2


def get_sentiment(text: str) -> Dict[str, float]:
    """
    Runs sentiment analysis on the text.
    Returns dictionary: {pos, neg, neutral}
    Neutral is optional for SST2, set to 0.
    """
    result = sentiment_model(text, truncation=True)
    pos, neg = 0.0, 0.0
    for r in result:
        if r["label"].lower() == "positive":
            pos += r["score"]
        elif r["label"].lower() == "negative":
            neg += r["score"]
    return {"pos": pos, "neg": neg, "neutral": 0.0}


def compute_post_sentiment_polarity(text: str) -> float:
    """
    Computes sentiment-based polarity P2 from post text.
    """
    sentiment = get_sentiment(text)
    # Use min(pos, neg) as measure of mixed sentiment
    return min(sentiment["pos"], sentiment["neg"])


def compute_polarity_score(post: Dict, alpha: float = 0.7, use_sentiment: bool = True) -> float:
    """
    Computes final polarity score P for a post.
    post: dictionary with keys 'upvotes', 'downvotes', 'text'
    alpha: weight for vote-based polarity (default 0.7)
    use_sentiment: whether to include sentiment polarity
    """
    upvotes = post.get("upvotes", 0)
    downvotes = post.get("downvotes", 0)
    text = post.get("text", "")

    # Vote-based polarity
    ratio = compute_vote_ratio(upvotes, downvotes)
    P1 = polarity_from_votes(ratio)

    # Optional sentiment polarity
    P2 = compute_post_sentiment_polarity(text) if use_sentiment else 0.0

    # Final weighted polarity
    P = alpha * P1 + (1 - alpha) * P2
    return P
