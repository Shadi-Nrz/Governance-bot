import streamlit as st
import pandas as pd
from fetcher import create_reddit_client, fetch_posts
from toxicity_detector import get_toxicity_score
# from polarity import compute_polarity_score
from polarity import compute_polarity_score, sentiment_model  # Add sentiment_model
from controversy import compute_controversy_score
# ---------------------------
# 1️⃣ Setup Reddit Client
# ---------------------------
reddit = create_reddit_client(
    client_id="9yiR3z9k9otq8zWsgnxsDg",
    client_secret="Lvw2p4nWi_D9jVAYspn6fL4TUG5J_g",
    user_agent="PhDResearchBot:v1.0 (by /u/JesanOvi)"
)

# ---------------------------
# 2️⃣ Streamlit UI Setup
# ---------------------------
st.set_page_config(page_title="Moderator Dashboard", layout="wide")
st.title("🧠 Reddit Post Moderation Dashboard")

# Sidebar: subreddit and number of posts
subreddit_name = st.sidebar.text_input("Enter subreddit:", value="learnprogramming")
post_limit = st.sidebar.slider("Number of posts to fetch", 1, 20, 5)
flag_threshold = st.sidebar.slider("Flag Threshold", 0.0, 1.0, 0.4)

# Fetch posts button
if st.sidebar.button("Fetch Posts"):
    with st.spinner(f"Fetching latest {post_limit} posts from r/{subreddit_name}..."):
        posts = fetch_posts(reddit, subreddit_name, limit=post_limit)

        # Compute toxicity, polarity, controversy, aggregate risk
        for post in posts:
            
            
            #post["polarity"] = 0
           
            
            # post["controversy"] = 0


            text = post.get("title", "") + " " + post.get("text", "")
            
            # Truncate text if too long (BERT limit is ~512 tokens, roughly 2000 chars)
            if len(text) > 2000:
                text = text[:2000]  # Truncate to first 2000 characters
            toxicity_result = get_toxicity_score(text)

            if isinstance(toxicity_result, dict):
                toxicity = toxicity_result.get("overall_max_toxicity", 0.0)
            else:
                toxicity = float(toxicity_result)

            post["toxicity"] = toxicity

            if post["score"] >= 0:
                upvotes = int(post["score"] * post["upvote_ratio"] / (2 * post["upvote_ratio"] - 1)) if post["upvote_ratio"] != 0.5 else post["score"]
                downvotes = upvotes - post["score"]
            else:
                # If score negative, we invert the calculation
                downvotes = int(-post["score"] * (1 - post["upvote_ratio"]) / (2 * (1 - post["upvote_ratio"]) - 1)) if post["upvote_ratio"] != 0.5 else post["score"]
                upvotes = downvotes + post["score"]

            sample_post = {
                "upvotes": upvotes,
                "downvotes": downvotes,
                "text": text
            }

            polarity = compute_polarity_score(sample_post)
            post["polarity"] = polarity

            sample_post_controversy = {
                "upvotes": upvotes,
                "downvotes": downvotes,
                "num_comments": post.get("num_comments", 0),
                "text": text,
                "upvote_ratio": post.get("upvote_ratio", 0.5),
                "comments": post.get("comments", [])
            }
            controversy = compute_controversy_score(
                sample_post_controversy,
                use_comments=False,
                sentiment_model=sentiment_model
            )
            post["controversy"] = controversy
            post["aggregate risk"] = 0.6 * toxicity + 0.2 * post["polarity"] + 0.2 * post["controversy"]

        # Save the DataFrame to session_state
        st.session_state.df = pd.DataFrame(posts)

# Only show DataFrame if it exists
if "df" in st.session_state:
    df = st.session_state.df

    st.subheader(f"Posts from r/{subreddit_name}")
    st.dataframe(df[["id", "title", "text", "toxicity", "polarity", "controversy", "aggregate risk"]])

    flagged = df[df["aggregate risk"] >= flag_threshold]
    st.subheader(f"🚩 Flagged Posts (Aggregated Risk ≥ {flag_threshold})")
    st.dataframe(flagged[["id", "title", "text", "toxicity", "polarity", "controversy", "aggregate risk"]])

    #avg_tox = df["toxicity"].mean()
    #max_tox = df["toxicity"].max()
    #st.metric("Average Toxicity", f"{avg_tox:.4f}")
    #st.metric("Max Toxicity", f"{max_tox:.4f}")
