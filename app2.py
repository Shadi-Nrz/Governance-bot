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

# Initialize session state for moderator actions
if 'moderator_actions' not in st.session_state:
    st.session_state.moderator_actions = {}

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
    
    subreddit_name = st.text_input(
        "📍 Subreddit",
        value="learnprogramming",
        help="Enter the name without 'r/'"
    )
    
    post_limit = st.slider(
        "📊 Posts to Fetch",
        min_value=1,
        max_value=50,
        value=10,
        help="Number of recent posts to analyze"
    )
    
    st.markdown("---")
    
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
    
    fetch_button = st.button("🔄 Fetch & Analyze Posts", type="primary", use_container_width=True)

# ---------------------------
# 5️⃣ Helper: Moderator Action Logic
# ---------------------------

def get_post_id(row):
    return row.get('id', row.name)

def get_default_moderator_action(row, flag_threshold: float) -> str:
    """
    Default moderation logic:
    - If already decided in session_state → return that.
    - If risk < flag_threshold → 'No Need Action'.
    - Else → 'Pending'.
    Also writes the default into session_state if not present.
    """
    post_id = get_post_id(row)
    if post_id in st.session_state.moderator_actions:
        return st.session_state.moderator_actions[post_id]
    
    if row['aggregate_risk'] < flag_threshold:
        action = "No Need Action"
    else:
        action = "Pending"
    
    st.session_state.moderator_actions[post_id] = action
    return action

# ---------------------------
# 6️⃣ Main Content Area
# ---------------------------

if fetch_button:
    with st.spinner(f"🔍 Analyzing r/{subreddit_name}..."):
        posts = fetch_posts(reddit, subreddit_name, limit=post_limit)
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, post in enumerate(posts):
            status_text.text(f"Analyzing post {i+1}/{len(posts)}...")
            progress_bar.progress((i + 1) / len(posts))
            
            text = post.get("title", "") + " " + post.get("text", "")
            if len(text) > 2000:
                text = text[:2000]
            
            toxicity_result = get_toxicity_score(text)
            toxicity = toxicity_result.get("overall_max_toxicity", 0.0) if isinstance(toxicity_result, dict) else float(toxicity_result)
            post["toxicity"] = toxicity
            
            if post["score"] >= 0:
                upvotes = int(post["score"] * post["upvote_ratio"] / (2 * post["upvote_ratio"] - 1)) if post["upvote_ratio"] != 0.5 else post["score"]
                downvotes = upvotes - post["score"]
            else:
                downvotes = int(-post["score"] * (1 - post["upvote_ratio"]) / (2 * (1 - post["upvote_ratio"]) - 1)) if post["upvote_ratio"] != 0.5 else post["score"]
                upvotes = downvotes + post["score"]
            
            polarity = compute_polarity_score({
                "upvotes": upvotes,
                "downvotes": downvotes,
                "text": text
            })
            post["polarity"] = polarity
            
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
            
            post["aggregate_risk"] = (
                weight_toxicity * toxicity +
                weight_polarity * polarity +
                weight_controversy * controversy
            )
            
            # Initialize moderator action based on risk (but don't overwrite existing)
            post_id = post.get("id", i)
            if post_id not in st.session_state.moderator_actions:
                if post["aggregate_risk"] < flag_threshold:
                    st.session_state.moderator_actions[post_id] = "No Need Action"
                else:
                    st.session_state.moderator_actions[post_id] = "Pending"
        
        progress_bar.empty()
        status_text.empty()
        
        st.session_state.df = pd.DataFrame(posts)
        st.session_state.subreddit_name = subreddit_name
        st.success(f"✅ Successfully analyzed {len(posts)} posts from r/{subreddit_name}")

# ---------------------------
# 7️⃣ Display Results
# ---------------------------

if "df" in st.session_state:
    df = st.session_state.df
    subreddit_name = st.session_state.get("subreddit_name", subreddit_name)
    
    # ---------------------------
    # Overview
    # ---------------------------
    st.header("📊 Overview Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_posts = len(df)
        st.metric("Total Posts", total_posts)
    
    with col2:
        flagged_count = len(df[df["aggregate_risk"] >= flag_threshold])
        flagged_pct = (flagged_count / total_posts * 100) if total_posts > 0 else 0
        st.metric("🚩 Flagged Posts", flagged_count, f"{flagged_pct:.1f}%")
    
    with col3:
        high_risk_count = len(df[df["aggregate_risk"] >= high_risk_threshold])
        st.metric("🔴 High Risk", high_risk_count)
    
    with col4:
        avg_risk = df["aggregate_risk"].mean()
        st.metric("Average Risk", f"{avg_risk:.3f}")
    
    st.markdown("---")
    
    # ---------------------------
    # Visualizations
    # ---------------------------
    st.header("📈 Risk Analysis")
    
    viz_col1, viz_col2 = st.columns(2)
    
    with viz_col1:
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
    # Flagged Posts (only Pending)
    # ---------------------------
    flagged_df = df[df["aggregate_risk"] >= flag_threshold].copy()
    flagged_df["moderator_action"] = flagged_df.apply(
        lambda row: get_default_moderator_action(row, flag_threshold),
        axis=1
    )
    flagged_df = flagged_df[flagged_df["moderator_action"] == "Pending"].sort_values(
        "aggregate_risk", ascending=False
    )
    
    if len(flagged_df) > 0:
        st.header(f"🚨 Flagged Posts ({len(flagged_df)})")
        
        for idx, post in flagged_df.iterrows():
            risk = post["aggregate_risk"]
            post_id = get_post_id(post)
            
            if risk >= high_risk_threshold:
                risk_badge = "🔴 HIGH PRIORITY"
            else:
                risk_badge = "🟡 MODERATE"
            
            action_status = st.session_state.moderator_actions.get(post_id, "Pending")
            if action_status in ["Solved", "No Need Action"]:
                status_emoji = "✅"
            elif action_status == "Needs Action":
                status_emoji = "⚠️"
            else:
                status_emoji = "⏳"
            
            with st.expander(
                f"{risk_badge} | {status_emoji} {action_status} | {post['title'][:80]}... | Risk: {risk:.3f}",
                expanded=(risk >= high_risk_threshold)
            ):
                metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                
                with metric_col1:
                    st.metric("Aggregate Risk", f"{post['aggregate_risk']:.3f}")
                with metric_col2:
                    st.metric("Toxicity", f"{post['toxicity']:.3f}")
                with metric_col3:
                    st.metric("Polarity", f"{post['polarity']:.3f}")
                with metric_col4:
                    st.metric("Controversy", f"{post['controversy']:.3f}")
                
                st.markdown("**Post Content:**")
                content = post.get('text', '')
                preview = content[:300] + "..." if len(content) > 300 else content
                st.text_area("", preview, height=100, key=f"content_{idx}", label_visibility="collapsed")
                
                info_col1, info_col2, info_col3 = st.columns(3)
                with info_col1:
                    st.write(f"**Score:** {post['score']}")
                with info_col2:
                    st.write(f"**Comments:** {post['num_comments']}")
                with info_col3:
                    st.write(f"**Upvote Ratio:** {post['upvote_ratio']:.2%}")
                
                # 🔗 Link + Actions (NO Reset)
                link_col, action_col1, action_col2 = st.columns([2, 1, 1])
                
                with link_col:
                    st.markdown(f"🔗 [**View on Reddit**]({post['url']})")
                
                with action_col1:
                    if st.button("✅ Mark as Solved", key=f"solved_{post_id}", use_container_width=True):
                        st.session_state.moderator_actions[post_id] = "Solved"
                        st.rerun()
                
                with action_col2:
                    if st.button("⚠️ Needs Action", key=f"needs_action_{post_id}", use_container_width=True):
                        st.session_state.moderator_actions[post_id] = "Needs Action"
                        st.rerun()
    
    else:
        st.success("✅ No posts flagged! All posts are below the threshold or already handled.")
    
    st.markdown("---")
    
    # ---------------------------
    # All Posts Table (Editable Moderator Action)
    # ---------------------------
    st.header("📋 All Posts")
    
    def categorize_risk(risk):
        if risk >= high_risk_threshold:
            return "🔴 High"
        elif risk >= flag_threshold:
            return "🟡 Moderate"
        else:
            return "🟢 Low"
    
    display_df = df.copy()
    display_df["Post ID"] = display_df.apply(get_post_id, axis=1)
    display_df['Risk Level'] = display_df['aggregate_risk'].apply(categorize_risk)
    
    display_df['Moderator Action'] = display_df.apply(
        lambda row: get_default_moderator_action(row, flag_threshold),
        axis=1
    )
    
    display_df['Toxicity'] = display_df['toxicity'].apply(lambda x: f"{x:.3f}")
    display_df['Polarity'] = display_df['polarity'].apply(lambda x: f"{x:.3f}")
    display_df['Controversy'] = display_df['controversy'].apply(lambda x: f"{x:.3f}")
    display_df['Aggregate Risk'] = display_df['aggregate_risk'].apply(lambda x: f"{x:.3f}")
    
    display_df['Reddit Link'] = display_df['url']  # store raw URL
    
    table_df = display_df[[
        'Post ID', 'Risk Level', 'Moderator Action', 'title', 'Toxicity', 'Polarity',
        'Controversy', 'Aggregate Risk', 'score', 'num_comments', 'Reddit Link'
    ]].rename(columns={
        'title': 'Title',
        'score': 'Score',
        'num_comments': 'Comments'
    })
    
    # 🔥 Editable table with dropdown for Moderator Action
    edited_table_df = st.data_editor(
        table_df,
        key="posts_table_editor",
        use_container_width=True,
        height=400,
        column_config={
            "Post ID": st.column_config.Column("Post ID", disabled=True),
            "Reddit Link": st.column_config.LinkColumn("Reddit Link", display_text="View Post"),
            "Moderator Action": st.column_config.SelectboxColumn(
                "Moderator Action",
                options=["Pending", "No Need Action", "Needs Action", "Solved"],
                required=True
            ),
        }
    )
    
    # Sync edited moderator actions back to session_state
    for _, row in edited_table_df.iterrows():
        post_id = row["Post ID"]
        new_action = row["Moderator Action"]
        st.session_state.moderator_actions[post_id] = new_action
    
    # ---------------------------
    # Export Options
    # ---------------------------
    st.markdown("---")
    st.header("💾 Export Data")
    
    export_col1, export_col2, export_col3 = st.columns(3)
    
    with export_col1:
        csv_df = df.copy()
        csv_df['moderator_action'] = csv_df.apply(
            lambda row: get_default_moderator_action(row, flag_threshold),
            axis=1
        )
        csv = csv_df.to_csv(index=False)
        st.download_button(
            label="📥 Download All Data (CSV)",
            data=csv,
            file_name=f"reddit_moderation_{subreddit_name}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with export_col2:
        flagged_all_df = df[df["aggregate_risk"] >= flag_threshold].copy()
        if len(flagged_all_df) > 0:
            flagged_all_df['moderator_action'] = flagged_all_df.apply(
                lambda row: get_default_moderator_action(row, flag_threshold),
                axis=1
            )
            flagged_csv = flagged_all_df.to_csv(index=False)
            st.download_button(
                label="🚩 Download Flagged Posts (CSV)",
                data=flagged_csv,
                file_name=f"reddit_flagged_{subreddit_name}.csv",
                mime="text/csv",
                use_container_width=True
            )
    
    with export_col3:
        needs_action_df = csv_df[csv_df['moderator_action'] == 'Needs Action']
        if len(needs_action_df) > 0:
            action_csv = needs_action_df.to_csv(index=False)
            st.download_button(
                label="⚠️ Download Action Items (CSV)",
                data=action_csv,
                file_name=f"reddit_action_items_{subreddit_name}.csv",
                mime="text/csv",
                use_container_width=True
            )
        else:
            st.info("No posts marked as 'Needs Action'")

else:
    st.info("👈 Configure settings in the sidebar and click 'Fetch & Analyze Posts' to begin")
    
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
        ### 🛠️ Moderation Tools
        - Direct Reddit links
        - Mark posts as Solved/Needs Action
        - Export action items
        - Track decisions
        """)
