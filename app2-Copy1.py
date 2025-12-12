import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from fetcher import create_reddit_client, fetch_posts
from toxicity_detector import get_toxicity_score
from polarity import compute_polarity_score, sentiment_model
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
# 2️⃣ Page Configuration
# ---------------------------
st.set_page_config(
    page_title="Reddit Moderation Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1F77B4;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .high-risk {
        border-left: 5px solid #ff4b4b;
    }
    .medium-risk {
        border-left: 5px solid #ffa500;
    }
    .low-risk {
        border-left: 5px solid #00cc00;
    }
    .stAlert {
        padding: 1rem;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------
# 3️⃣ Header
# ---------------------------
st.markdown('<p class="main-header">🛡️ Reddit Moderation Dashboard</p>', unsafe_allow_html=True)
st.markdown("---")

# ---------------------------
# 4️⃣ Sidebar Controls
# ---------------------------
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Subreddit selection
    subreddit_name = st.text_input(
        "📍 Subreddit",
        value="learnprogramming",
        help="Enter the name without 'r/'"
    )
    
    # Number of posts
    post_limit = st.slider(
        "📊 Posts to Fetch",
        min_value=1,
        max_value=50,
        value=10,
        help="Number of recent posts to analyze"
    )
    
    st.markdown("---")
    
    # Risk thresholds
    st.subheader("🎯 Risk Thresholds")
    flag_threshold = st.slider(
        "Flagging Threshold",
        0.0, 1.0, 0.4, 0.05,
        help="Posts above this aggregate risk are flagged"
    )
    
    high_risk_threshold = st.slider(
        "High Risk Threshold",
        0.0, 1.0, 0.7, 0.05,
        help="Posts above this are considered high priority"
    )
    
    st.markdown("---")
    
    # Weight configuration
    with st.expander("⚖️ Risk Weights"):
        st.caption("Adjust how each metric contributes to aggregate risk")
        weight_toxicity = st.slider("Toxicity", 0.0, 1.0, 0.6, 0.05)
        weight_polarity = st.slider("Polarity", 0.0, 1.0, 0.2, 0.05)
        weight_controversy = st.slider("Controversy", 0.0, 1.0, 0.2, 0.05)
        
        # Normalize weights
        total = weight_toxicity + weight_polarity + weight_controversy
        if total > 0:
            weight_toxicity /= total
            weight_polarity /= total
            weight_controversy /= total
    
    st.markdown("---")
    
    # Fetch button
    fetch_button = st.button("🔄 Fetch & Analyze Posts", type="primary", use_container_width=True)

# ---------------------------
# 5️⃣ Main Content Area
# ---------------------------

if fetch_button:
    with st.spinner(f"🔍 Analyzing r/{subreddit_name}..."):
        posts = fetch_posts(reddit, subreddit_name, limit=post_limit)
        
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Process posts
        for i, post in enumerate(posts):
            status_text.text(f"Analyzing post {i+1}/{len(posts)}...")
            progress_bar.progress((i + 1) / len(posts))
            
            # Get text
            text = post.get("title", "") + " " + post.get("text", "")
            if len(text) > 2000:
                text = text[:2000]
            
            # Toxicity
            toxicity_result = get_toxicity_score(text)
            toxicity = toxicity_result.get("overall_max_toxicity", 0.0) if isinstance(toxicity_result, dict) else float(toxicity_result)
            post["toxicity"] = toxicity
            
            # Calculate upvotes/downvotes
            if post["score"] >= 0:
                upvotes = int(post["score"] * post["upvote_ratio"] / (2 * post["upvote_ratio"] - 1)) if post["upvote_ratio"] != 0.5 else post["score"]
                downvotes = upvotes - post["score"]
            else:
                downvotes = int(-post["score"] * (1 - post["upvote_ratio"]) / (2 * (1 - post["upvote_ratio"]) - 1)) if post["upvote_ratio"] != 0.5 else post["score"]
                upvotes = downvotes + post["score"]
            
            # Polarity
            polarity = compute_polarity_score({
                "upvotes": upvotes,
                "downvotes": downvotes,
                "text": text
            })
            post["polarity"] = polarity
            
            # Controversy
            controversy = compute_controversy_score(
                {
                    "upvotes": upvotes,
                    "downvotes": downvotes,
                    "num_comments": post.get("num_comments", 0),
                    "text": text,
                    "upvote_ratio": post.get("upvote_ratio", 0.5),
                    "comments": post.get("comments", [])
                },
                use_comments=False,
                sentiment_model=sentiment_model
            )
            post["controversy"] = controversy
            
            # Aggregate risk
            post["aggregate_risk"] = (
                weight_toxicity * toxicity +
                weight_polarity * polarity +
                weight_controversy * controversy
            )
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        # Save to session state
        st.session_state.df = pd.DataFrame(posts)
        st.session_state.subreddit_name = subreddit_name
        st.success(f"✅ Successfully analyzed {len(posts)} posts from r/{subreddit_name}")

# ---------------------------
# 6️⃣ Display Results
# ---------------------------

if "df" in st.session_state:
    df = st.session_state.df
    subreddit_name = st.session_state.get("subreddit_name", subreddit_name)
    
    # ---------------------------
    # Summary Statistics
    # ---------------------------
    st.header("📊 Overview Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_posts = len(df)
        st.metric(
            "Total Posts",
            total_posts,
            help="Total number of posts analyzed"
        )
    
    with col2:
        flagged_count = len(df[df["aggregate_risk"] >= flag_threshold])
        flagged_pct = (flagged_count / total_posts * 100) if total_posts > 0 else 0
        st.metric(
            "🚩 Flagged Posts",
            flagged_count,
            f"{flagged_pct:.1f}%",
            delta_color="inverse"
        )
    
    with col3:
        high_risk_count = len(df[df["aggregate_risk"] >= high_risk_threshold])
        st.metric(
            "🔴 High Risk",
            high_risk_count,
            help="Posts requiring immediate attention"
        )
    
    with col4:
        avg_risk = df["aggregate_risk"].mean()
        st.metric(
            "Average Risk",
            f"{avg_risk:.3f}",
            help="Mean aggregate risk across all posts"
        )
    
    st.markdown("---")
    
    # ---------------------------
    # Visualizations
    # ---------------------------
    st.header("📈 Risk Analysis")
    
    viz_col1, viz_col2 = st.columns(2)
    
    with viz_col1:
        # Risk distribution histogram
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=df["aggregate_risk"],
            nbinsx=20,
            marker_color='#1F77B4',
            name='Risk Distribution'
        ))
        fig_hist.add_vline(
            x=flag_threshold,
            line_dash="dash",
            line_color="orange",
            annotation_text="Flag Threshold"
        )
        fig_hist.add_vline(
            x=high_risk_threshold,
            line_dash="dash",
            line_color="red",
            annotation_text="High Risk"
        )
        fig_hist.update_layout(
            title="Aggregate Risk Distribution",
            xaxis_title="Risk Score",
            yaxis_title="Number of Posts",
            showlegend=False,
            height=400
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    
    with viz_col2:
        # Metric comparison
        metrics_df = pd.DataFrame({
            'Metric': ['Toxicity', 'Polarity', 'Controversy'],
            'Average Score': [
                df['toxicity'].mean(),
                df['polarity'].mean(),
                df['controversy'].mean()
            ]
        })
        
        fig_bar = px.bar(
            metrics_df,
            x='Metric',
            y='Average Score',
            color='Metric',
            title='Average Metric Scores',
            color_discrete_map={
                'Toxicity': '#ff4b4b',
                'Polarity': '#ffa500',
                'Controversy': '#1F77B4'
            }
        )
        fig_bar.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig_bar, use_container_width=True)
    
    st.markdown("---")
    
    # ---------------------------
    # Flagged Posts Section
    # ---------------------------
    flagged_df = df[df["aggregate_risk"] >= flag_threshold].sort_values("aggregate_risk", ascending=False)
    
    if len(flagged_df) > 0:
        st.header(f"🚨 Flagged Posts ({len(flagged_df)})")
        
        for idx, post in flagged_df.iterrows():
            # Determine risk level
            risk = post["aggregate_risk"]
            if risk >= high_risk_threshold:
                risk_badge = "🔴 HIGH PRIORITY"
                risk_class = "high-risk"
                risk_color = "#ff4b4b"
            elif risk >= flag_threshold:
                risk_badge = "🟡 MODERATE"
                risk_class = "medium-risk"
                risk_color = "#ffa500"
            else:
                risk_badge = "🟢 LOW"
                risk_class = "low-risk"
                risk_color = "#00cc00"
            
            with st.expander(f"{risk_badge} | {post['title'][:80]}... | Risk: {risk:.3f}", expanded=(risk >= high_risk_threshold)):
                
                # Risk metrics row
                metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                
                with metric_col1:
                    st.metric("Aggregate Risk", f"{post['aggregate_risk']:.3f}")
                with metric_col2:
                    st.metric("Toxicity", f"{post['toxicity']:.3f}")
                with metric_col3:
                    st.metric("Polarity", f"{post['polarity']:.3f}")
                with metric_col4:
                    st.metric("Controversy", f"{post['controversy']:.3f}")
                
                # Post details
                st.markdown("**Post Content:**")
                content = post.get('text', '')
                preview = content[:300] + "..." if len(content) > 300 else content
                st.text_area("", preview, height=100, key=f"content_{idx}", label_visibility="collapsed")
                
                # Metadata
                info_col1, info_col2, info_col3 = st.columns(3)
                with info_col1:
                    st.write(f"**Score:** {post['score']}")
                with info_col2:
                    st.write(f"**Comments:** {post['num_comments']}")
                with info_col3:
                    st.write(f"**Upvote Ratio:** {post['upvote_ratio']:.2%}")
                
                # Link to post
                st.markdown(f"🔗 [**View on Reddit**]({post['url']})")
    
    else:
        st.success("✅ No posts flagged! All posts are below the flagging threshold.")
    
    st.markdown("---")
    
    # ---------------------------
    # All Posts Table
    # ---------------------------
    st.header("📋 All Posts")
    
    # Add risk category
    def categorize_risk(risk):
        if risk >= high_risk_threshold:
            return "🔴 High"
        elif risk >= flag_threshold:
            return "🟡 Moderate"
        else:
            return "🟢 Low"
    
    display_df = df.copy()
    display_df['Risk Level'] = display_df['aggregate_risk'].apply(categorize_risk)
    
    # Format scores
    display_df['Toxicity'] = display_df['toxicity'].apply(lambda x: f"{x:.3f}")
    display_df['Polarity'] = display_df['polarity'].apply(lambda x: f"{x:.3f}")
    display_df['Controversy'] = display_df['controversy'].apply(lambda x: f"{x:.3f}")
    display_df['Aggregate Risk'] = display_df['aggregate_risk'].apply(lambda x: f"{x:.3f}")
    
    # Select columns to display
    table_df = display_df[[
        'Risk Level', 'title', 'Toxicity', 'Polarity', 
        'Controversy', 'Aggregate Risk', 'score', 'num_comments'
    ]].rename(columns={
        'title': 'Title',
        'score': 'Score',
        'num_comments': 'Comments'
    })
    
    st.dataframe(
        table_df,
        use_container_width=True,
        height=400
    )
    
    # ---------------------------
    # Export Options
    # ---------------------------
    st.markdown("---")
    st.header("💾 Export Data")
    
    export_col1, export_col2 = st.columns(2)
    
    with export_col1:
        # CSV export
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download Full Data (CSV)",
            data=csv,
            file_name=f"reddit_moderation_{subreddit_name}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with export_col2:
        # Flagged posts only
        if len(flagged_df) > 0:
            flagged_csv = flagged_df.to_csv(index=False)
            st.download_button(
                label="🚩 Download Flagged Posts Only (CSV)",
                data=flagged_csv,
                file_name=f"reddit_flagged_{subreddit_name}.csv",
                mime="text/csv",
                use_container_width=True
            )

else:
    # Initial state - no data yet
    st.info("👈 Configure settings in the sidebar and click 'Fetch & Analyze Posts' to begin")
    
    # Show feature highlights
    st.markdown("---")
    st.header("✨ Dashboard Features")
    
    feature_col1, feature_col2, feature_col3 = st.columns(3)
    
    with feature_col1:
        st.markdown("""
        ### 📊 Analytics
        - Risk distribution charts
        - Metric comparisons
        - Real-time statistics
        - Trend visualization
        """)
    
    with feature_col2:
        st.markdown("""
        ### 🎯 Smart Detection
        - Toxicity analysis
        - Polarity measurement
        - Controversy detection
        - Aggregate risk scoring
        """)
    
    with feature_col3:
        st.markdown("""
        ### 🛠️ Tools
        - Customizable thresholds
        - Adjustable weights
        - Data export (CSV)
        - Direct Reddit links
        """)