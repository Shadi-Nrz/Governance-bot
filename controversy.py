"""
Controversy detection module for Reddit posts.

This module analyzes posts for controversy signals using multiple dimensions:
1. Vote-based controversy (balanced upvote/downvote distribution)
2. Engagement intensity (high comment-to-vote ratio)
3. Linguistic controversy markers (debate language, polarizing terms)
4. Sentiment variance in comments (optional)

Integrates with toxicity_detector and polarity modules for comprehensive analysis.
"""

from typing import Dict, List, Any, Optional
import math
import re
from transformers import pipeline
import torch


def compute_controversy_score(
    post_data: Dict[str, Any],
    use_comments: bool = False,
    weights: Optional[Dict[str, float]] = None,
    sentiment_model: Optional[Any] = None
) -> float:
    """
    Compute a controversy score for a Reddit post based on multiple signals.
    
    Args:
        post_data: Dictionary containing post information with keys:
            - 'upvotes': int - Number of upvotes
            - 'downvotes': int - Number of downvotes
            - 'num_comments': int - Total number of comments
            - 'text': str - Post title + body text
            - 'upvote_ratio': float - Reddit's upvote ratio (optional)
            - 'comments': List[Dict] - List of comment dicts (optional)
        use_comments: Whether to analyze comment sentiment variance.
                     Requires 'comments' in post_data.
        weights: Custom weights for controversy components. Defaults to:
                {'vote': 0.4, 'engagement': 0.3, 'language': 0.2, 'comment_variance': 0.1}
        sentiment_model: Pre-loaded sentiment analysis pipeline from polarity.py
                        to avoid reloading the model. If None and use_comments=True,
                        will load model internally.
    
    Returns:
        float: Controversy score between 0.0 and 1.0
               - 0.0-0.3: Low controversy
               - 0.3-0.5: Moderate controversy
               - 0.5-0.7: High controversy
               - 0.7-1.0: Extreme controversy
    
    Example:
        >>> post = {
        ...     "upvotes": 150,
        ...     "downvotes": 140,
        ...     "num_comments": 200,
        ...     "text": "This is a controversial opinion...",
        ...     "upvote_ratio": 0.517
        ... }
        >>> score = compute_controversy_score(post)
        >>> print(f"Controversy: {score:.3f}")
    """
    # Default weights
    if weights is None:
        if use_comments:
            weights = {
                'vote': 0.35,
                'engagement': 0.25,
                'language': 0.25,
                'comment_variance': 0.15
            }
        else:
            weights = {
                'vote': 0.45,
                'engagement': 0.35,
                'language': 0.20,
                'comment_variance': 0.0
            }
    
    # Extract post data
    upvotes = post_data.get("upvotes", 0)
    downvotes = post_data.get("downvotes", 0)
    num_comments = post_data.get("num_comments", 0)
    text = post_data.get("text", "")
    comments = post_data.get("comments", [])
    
    # Component 1: Vote-based controversy
    vote_controversy = calculate_vote_controversy(upvotes, downvotes)
    
    # Component 2: Engagement intensity
    engagement_score = calculate_engagement_intensity(upvotes, downvotes, num_comments)
    
    # Component 3: Linguistic controversy markers
    language_controversy = detect_controversial_language(text)
    
    # Component 4: Comment sentiment variance (optional)
    comment_variance = 0.0
    if use_comments and comments:
        comment_variance = calculate_comment_variance(comments, sentiment_model=sentiment_model)
    
    # Weighted combination
    controversy_score = (
        weights['vote'] * vote_controversy +
        weights['engagement'] * engagement_score +
        weights['language'] * language_controversy +
        weights['comment_variance'] * comment_variance
    )
    
    # Ensure score is in [0, 1]
    return min(max(controversy_score, 0.0), 1.0)


def calculate_vote_controversy(upvotes: int, downvotes: int) -> float:
    """
    Calculate controversy based on vote distribution using Reddit's algorithm.
    
    Controversy is highest when:
    - There are many votes on both sides (high magnitude)
    - Votes are evenly split (high balance)
    
    This approximates Reddit's internal controversy score formula.
    
    Args:
        upvotes: Number of upvotes
        downvotes: Number of downvotes
    
    Returns:
        float: Vote controversy score in [0, 1]
    """
    # Handle edge cases
    if upvotes + downvotes == 0:
        return 0.0
    
    if upvotes == 0 or downvotes == 0:
        return 0.0  # One-sided posts are not controversial
    
    total_votes = upvotes + downvotes
    
    # Balance score: How evenly split are the votes?
    # Peaks at 1.0 when upvotes == downvotes
    balance = min(upvotes, downvotes) / max(upvotes, downvotes)
    
    # Magnitude score: More votes = potentially more controversial
    # Use logarithmic scaling to prevent domination by viral posts
    # Scale: ~0.5 at 100 votes, ~0.7 at 1000 votes, ~0.9 at 10000 votes
    magnitude = math.log10(total_votes + 1) / 4.5
    magnitude = min(magnitude, 1.0)
    
    # Combine balance and magnitude
    # Both factors must be present for high controversy
    controversy = balance * (0.6 + 0.4 * magnitude)
    
    return min(controversy, 1.0)


def calculate_engagement_intensity(upvotes: int, downvotes: int, num_comments: int) -> float:
    """
    Calculate engagement-based controversy using comment-to-vote ratio.
    
    Controversial posts often generate disproportionately many comments
    relative to their vote count, as people engage in debates.
    
    Args:
        upvotes: Number of upvotes
        downvotes: Number of downvotes
        num_comments: Total number of comments
    
    Returns:
        float: Engagement intensity score in [0, 1]
    """
    total_votes = upvotes + downvotes
    
    # Avoid division by zero
    if total_votes == 0:
        # High comments with no votes is unusual - moderate controversy
        return min(num_comments / 50.0, 0.5)
    
    # Calculate comment-to-vote ratio
    comment_ratio = num_comments / total_votes
    
    # Typical non-controversial posts: ratio ~ 0.05-0.15
    # Controversial posts: ratio > 0.3
    # Highly controversial: ratio > 0.6
    
    # Normalize using sigmoid-like scaling
    if comment_ratio < 0.1:
        score = 0.0
    elif comment_ratio < 0.3:
        score = (comment_ratio - 0.1) / 0.2 * 0.4  # 0.1-0.3 → 0-0.4
    elif comment_ratio < 0.6:
        score = 0.4 + (comment_ratio - 0.3) / 0.3 * 0.35  # 0.3-0.6 → 0.4-0.75
    else:
        score = 0.75 + min((comment_ratio - 0.6) / 0.4, 0.25)  # 0.6+ → 0.75-1.0
    
    return min(score, 1.0)


def detect_controversial_language(text: str) -> float:
    """
    Detect linguistic markers that indicate controversial content.
    
    Analyzes text for:
    - Controversial topic keywords
    - Debate-inviting phrases
    - Strong emotional language
    - Polarizing statements
    - Question patterns that invite disagreement
    
    Args:
        text: Post title and body text
    
    Returns:
        float: Language controversy score in [0, 1]
    """
    if not text or len(text.strip()) < 10:
        return 0.0
    
    text_lower = text.lower()
    controversy_score = 0.0
    
    # 1. Controversial topic markers
    controversial_keywords = [
        'controversial', 'unpopular opinion', 'hot take', 'change my mind',
        'cmv', 'debate', 'disagree', 'wrong about', 'prove me wrong',
        'am i wrong', 'am i the asshole', 'aita', 'politically incorrect',
        'trigger warning', 'sensitive topic', 'divisive', 'polarizing'
    ]
    keyword_matches = sum(1 for kw in controversial_keywords if kw in text_lower)
    controversy_score += min(keyword_matches * 0.2, 0.35)
    
    # 2. Debate-inviting phrases
    debate_patterns = [
        r'why (do|does|is|are|don\'t|doesn\'t) (people|everyone|anyone|someone)',
        r'am i the only one (who|that)',
        r'does anyone else',
        r'is it just me',
        r'why (can\'t|won\'t|don\'t) (we|people|they)',
        r'prove me wrong',
        r'change my (mind|view)',
        r'convince me'
    ]
    debate_matches = sum(1 for pattern in debate_patterns if re.search(pattern, text_lower))
    controversy_score += min(debate_matches * 0.15, 0.25)
    
    # 3. Strong polarizing language
    polarizing_words = [
        'always', 'never', 'everyone', 'nobody', 'all', 'none',
        'completely', 'totally', 'absolutely', 'definitely',
        'obviously', 'clearly', 'undeniably'
    ]
    polarizing_count = sum(1 for word in polarizing_words if f' {word} ' in f' {text_lower} ')
    controversy_score += min(polarizing_count * 0.05, 0.15)
    
    # 4. Intense emotional language
    intense_emotion_words = [
        'hate', 'love', 'terrible', 'amazing', 'worst', 'best',
        'awful', 'horrible', 'disgusting', 'ridiculous', 'stupid',
        'brilliant', 'perfect', 'disaster', 'outrageous', 'absurd',
        'insane', 'crazy', 'unbelievable'
    ]
    emotion_count = sum(1 for word in intense_emotion_words if word in text_lower)
    controversy_score += min(emotion_count * 0.08, 0.2)
    
    # 5. Excessive punctuation (!!!, ???)
    exclamation_count = text.count('!')
    question_count = text.count('?')
    
    exclamation_ratio = exclamation_count / max(len(text), 1)
    question_ratio = question_count / max(len(text), 1)
    
    punctuation_score = min((exclamation_ratio + question_ratio) * 100, 0.15)
    controversy_score += punctuation_score
    
    # 6. ALL CAPS detection
    words = text.split()
    if words:
        caps_words = sum(1 for word in words if word.isupper() and len(word) > 2)
        caps_ratio = caps_words / len(words)
        controversy_score += min(caps_ratio * 0.5, 0.1)
    
    return min(controversy_score, 1.0)


def calculate_comment_variance(comments: List[Dict[str, Any]], sentiment_model=None) -> float:
    """
    Calculate sentiment-based controversy from comments using sentiment analysis.
    
    Uses the formula: C = 1 - abs(pos/(pos+neg) - 0.5) * 2
    
    High controversy when:
    - pos ≈ neg → C close to 1.0 (mixed sentiment)
    - pos >> neg or neg >> pos → C close to 0.0 (one-sided)
    
    Args:
        comments: List of comment dictionaries with 'body' key
        sentiment_model: Pre-loaded sentiment analysis pipeline (optional)
                        If None, will import and use DistilBERT-SST2
    
    Returns:
        float: Comment sentiment controversy score in [0, 1]
    
    Example:
        >>> comments = [
        ...     {"body": "This is great!"},
        ...     {"body": "This is terrible!"}
        ... ]
        >>> score = calculate_comment_variance(comments)
        >>> # Returns ~1.0 (high controversy - mixed sentiment)
    """
    if not comments or len(comments) < 3:
        return 0.0
    
    # Import sentiment model if not provided
    if sentiment_model is None:
        try:
            sentiment_model = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                device=0 if torch.cuda.is_available() else -1
            )
        except ImportError:
            # Fallback: use comment scores if transformers not available
            return _calculate_comment_variance_fallback(comments)
    
    # Extract comment text
    comment_texts = [c.get('body', '') for c in comments if c.get('body', '').strip()]
    
    if not comment_texts:
        return 0.0
    
    # Analyze sentiment for each comment
    pos_count = 0
    neg_count = 0
    
    for text in comment_texts:
        if len(text.strip()) < 5:  # Skip very short comments
            continue
            
        try:
            # Get sentiment prediction
            result = sentiment_model(text[:512], truncation=True)  # Limit length
            
            if isinstance(result, list) and len(result) > 0:
                label = result[0].get('label', '').lower()
                
                if 'positive' in label or label == 'pos':
                    pos_count += 1
                elif 'negative' in label or label == 'neg':
                    neg_count += 1
        except Exception:
            continue  # Skip problematic comments
    
    # Calculate controversy using the formula: C = 1 - abs(pos/(pos+neg) - 0.5) * 2
    total = pos_count + neg_count
    
    if total < 3:  # Need minimum comments for valid analysis
        return 0.0
    
    # Apply the controversy formula
    ratio = pos_count / total
    controversy = 1.0 - abs(ratio - 0.5) * 2.0
    
    # Scale by confidence: more comments = more reliable
    confidence = min(total / 20.0, 1.0)  # Full confidence at 20+ comments
    
    return controversy * confidence


def _calculate_comment_variance_fallback(comments: List[Dict[str, Any]]) -> float:
    """
    Fallback method using comment scores when sentiment analysis is unavailable.
    
    Args:
        comments: List of comment dictionaries with 'score' keys
    
    Returns:
        float: Approximate controversy based on vote patterns
    """
    scores = [c.get('score', 0) for c in comments if 'score' in c]
    
    if not scores or len(scores) < 3:
        return 0.0
    
    # Count positive and negative scored comments
    pos_count = sum(1 for s in scores if s > 2)
    neg_count = sum(1 for s in scores if s < -1)
    total = pos_count + neg_count
    
    if total < 3:
        return 0.0
    
    # Apply same formula to comment scores
    ratio = pos_count / total
    controversy = 1.0 - abs(ratio - 0.5) * 2.0
    
    return controversy * 0.8  # Slightly lower weight for fallback method


def get_controversy_label(score: float) -> str:
    """
    Convert controversy score to human-readable label with emoji.
    
    Args:
        score: Controversy score in [0, 1]
    
    Returns:
        str: Descriptive label with emoji indicator
    """
    if score >= 0.7:
        return "🔥 Extreme Controversy"
    elif score >= 0.5:
        return "⚠️ High Controversy"
    elif score >= 0.3:
        return "📊 Moderate Discussion"
    else:
        return "✅ Low Controversy"


def get_detailed_controversy_analysis(post_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Perform detailed controversy analysis with component breakdowns.
    
    Args:
        post_data: Post dictionary with all required fields
    
    Returns:
        dict: Detailed analysis including:
            - overall_score: Final controversy score
            - label: Human-readable label
            - components: Breakdown of individual scores
            - risk_factors: List of detected risk factors
    """
    upvotes = post_data.get("upvotes", 0)
    downvotes = post_data.get("downvotes", 0)
    num_comments = post_data.get("num_comments", 0)
    text = post_data.get("text", "")
    
    # Calculate all components
    vote_score = calculate_vote_controversy(upvotes, downvotes)
    engagement_score = calculate_engagement_intensity(upvotes, downvotes, num_comments)
    language_score = detect_controversial_language(text)
    
    # Overall score
    overall = compute_controversy_score(post_data)
    
    # Identify risk factors
    risk_factors = []
    if vote_score > 0.5:
        risk_factors.append("Highly polarized voting pattern")
    if engagement_score > 0.6:
        risk_factors.append("Unusually high comment engagement")
    if language_score > 0.4:
        risk_factors.append("Contains controversial language markers")
    
    total_votes = upvotes + downvotes
    if total_votes > 0:
        comment_ratio = num_comments / total_votes
        if comment_ratio > 0.5:
            risk_factors.append(f"High comment-to-vote ratio ({comment_ratio:.2f})")
    
    return {
        "overall_score": round(overall, 4),
        "label": get_controversy_label(overall),
        "components": {
            "vote_controversy": round(vote_score, 4),
            "engagement_intensity": round(engagement_score, 4),
            "language_markers": round(language_score, 4)
        },
        "risk_factors": risk_factors,
        "total_votes": total_votes,
        "vote_balance": round(min(upvotes, downvotes) / max(upvotes, downvotes), 3) if max(upvotes, downvotes) > 0 else 0
    }