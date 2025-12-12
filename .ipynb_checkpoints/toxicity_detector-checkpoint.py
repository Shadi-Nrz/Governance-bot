"""
Toxicity detection module using the unitary/toxic-bert pre-trained model.

This module provides functionality to analyze text for various toxicity categories
including toxicity, severe_toxicity, obscene, threat, insult, and identity_hate.
"""

from typing import List, Dict, Any
from transformers import pipeline
import torch


# Load the pre-trained toxicity classification model
# Uses GPU if available, otherwise falls back to CPU
device = 0 if torch.cuda.is_available() else -1
toxicity_classifier = pipeline(
    "text-classification",
    model="unitary/toxic-bert",
    device=device,
    top_k=None  # Return scores for all labels
)


def chunk_text(text: str, max_tokens: int = 200) -> List[str]:
    """
    Split text into chunks of approximately max_tokens words.

    The BERT model has token limits, so long texts must be chunked before
    processing. This function splits text by word boundaries to respect
    the approximate token limit while maintaining readability.

    Args:
        text: The input text to chunk.
        max_tokens: Maximum number of words per chunk (approximate).
                   Defaults to 200 words.

    Returns:
        List[str]: A list of text chunks, each containing approximately
                  max_tokens words. The final chunk may be shorter.
    """
    # Split text into words
    words = text.split()

    # Return empty list if text is empty
    if not words:
        return []

    # Return original text if it's smaller than chunk size
    if len(words) <= max_tokens:
        return [text]

    chunks = []
    current_chunk = []

    # Build chunks by accumulating words up to max_tokens
    for word in words:
        current_chunk.append(word)

        # Add chunk when it reaches max_tokens and continue with next chunk
        if len(current_chunk) >= max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = []

    # Add any remaining words as the final chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def get_toxicity_score(text: str, chunk_size: int = 200) -> Dict[str, Any]:
    """
    Analyze text for toxicity and return aggregated toxicity scores.

    Processes text through the toxicity model, handling long texts by chunking.
    Aggregates results across all chunks and toxicity categories.

    Args:
        text: The input text to analyze for toxicity.
        chunk_size: Maximum number of words per chunk (approximate).
                   Defaults to 200 words.

    Returns:
        Dict[str, Any]: A dictionary containing:
            - label_scores: Dict with per-label aggregated scores
                - Each label includes:
                    - max: Highest score for that label across chunks
                    - avg: Average score for that label across chunks
            - overall_max_toxicity: Maximum toxicity score across all labels
            - num_chunks: Number of chunks the text was split into
            - labels_present: List of all toxicity labels detected

    Example:
        >>> result = get_toxicity_score("Some text to analyze")
        >>> print(result['overall_max_toxicity'])
        0.95
    """
    # Handle empty text
    if not text or not text.strip():
        return {
            "label_scores": {},
            "overall_max_toxicity": 0.0,
            "num_chunks": 0,
            "labels_present": []
        }

    # Split text into manageable chunks
    chunks = chunk_text(text, max_tokens=chunk_size)

    # Initialize storage for aggregated scores
    label_scores: Dict[str, List[float]] = {}
    all_labels = set()

    # Process each chunk through the toxicity model
    for chunk in chunks:
        # Get predictions for this chunk (returns list of label dicts)
        # predictions = toxicity_classifier(chunk)

        # fix the error:
        # Process each chunk through the toxicity model
        # Get predictions for this chunk (returns list of label dicts)
        predictions = toxicity_classifier(chunk, truncation=True, max_length=512)  # ← ADD THIS!


        # predictions is a list of lists: [[{label, score}, ...]]
        for result_list in predictions:
            for label_dict in result_list:
                label = label_dict["label"]
                score = label_dict["score"]

                # Track all labels seen
                all_labels.add(label)

                # Store score for aggregation
                if label not in label_scores:
                    label_scores[label] = []
                label_scores[label].append(score)

    # Aggregate scores across chunks
    aggregated_scores: Dict[str, Dict[str, float]] = {}
    overall_max = 0.0

    for label, scores in label_scores.items():
        max_score = max(scores)
        avg_score = sum(scores) / len(scores)

        aggregated_scores[label] = {
            "max": round(max_score, 4),
            "avg": round(avg_score, 4)
        }

        # Track overall maximum toxicity across all labels
        if max_score > overall_max:
            overall_max = max_score

    return {
        "label_scores": aggregated_scores,
        "overall_max_toxicity": round(overall_max, 4),
        "num_chunks": len(chunks),
        "labels_present": sorted(list(all_labels))
    }