"""
Reddit post fetcher module using PRAW (Python Reddit API Wrapper).

This module provides functionality to authenticate with Reddit and fetch posts
from subreddits along with their metadata and top-level comments.
"""

from typing import Optional, Dict, List, Any
import praw


def create_reddit_client(
    client_id: str,
    client_secret: str,
    user_agent: str
) -> praw.Reddit:
    """
    Create and return an authenticated Reddit client instance.

    Args:
        client_id: OAuth client ID from Reddit app registration.
        client_secret: OAuth client secret from Reddit app registration.
        user_agent: User agent string identifying your application.
                   Format: platform:app_name:version (by /u/username)

    Returns:
        praw.Reddit: An authenticated Reddit instance ready for API calls.

    Raises:
        praw.exceptions.InvalidClient: If authentication credentials are invalid.
    """
    reddit = praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent
    )
    return reddit


def fetch_posts(
    reddit: praw.Reddit,
    subreddit_name: str,
    limit: int,
    fetch_comments: bool = True,
    comment_limit: int = 50
) -> List[Dict[str, Any]]:
    """
    Fetch the latest posts from a specified subreddit with optional comments.

    Args:
        reddit: An authenticated praw.Reddit instance.
        subreddit_name: Name of the subreddit (without 'r/' prefix).
        limit: Maximum number of posts to fetch.
        fetch_comments: Whether to fetch top-level comments for each post.
                       Defaults to True.
        comment_limit: Maximum number of top-level comments to fetch per post.
                      Defaults to 50. Only used if fetch_comments is True.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries containing post data.
            Each post dictionary includes:
                - id: Post ID
                - title: Post title
                - text: Post text/body content
                - score: Post upvote score
                - num_comments: Total number of comments on the post
                - url: Direct link to the post
                - comments: List of comment dictionaries (if fetch_comments=True)
                            Each comment dict contains:
                                - body: Comment text
                                - score: Comment upvote score

    Raises:
        praw.exceptions.InvalidSubreddit: If the subreddit does not exist.
    """
    posts_data = []

    # Access the subreddit
    subreddit = reddit.subreddit(subreddit_name)

    # Iterate through new posts up to the specified limit
    for post in subreddit.new(limit=limit):
        post_dict: Dict[str, Any] = {
            "id": post.id,
            "title": post.title,
            "text": post.selftext,
            "score": post.score,               # net score (upvotes - downvotes)
            "num_comments": post.num_comments,
            "url": post.url,
            "upvote_ratio": post.upvote_ratio,
        }

        # Optionally fetch top-level comments
        if fetch_comments:
            comments = []
            # Replace MoreComments objects with actual Comment objects
            post.comments.replace_more(limit=0)

            # Iterate through top-level comments
            for comment in post.comments[:comment_limit]:
                comment_dict = {
                    "body": comment.body,
                    "score": comment.score,
                }
                comments.append(comment_dict)

            post_dict["comments"] = comments

        posts_data.append(post_dict)

    return posts_data