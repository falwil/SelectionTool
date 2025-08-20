import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

# Try to import upsetplot, handle gracefully if not available
try:
    import upsetplot
    UPSETPLOT_AVAILABLE = True
except ImportError:
    UPSETPLOT_AVAILABLE = False
    st.sidebar.warning("⚠️ upsetplot library not installed. UpSet plots will be disabled. Install with: pip install upsetplot")

# -----------------------------
# Simulated data
# -----------------------------
np.random.seed(42)
num_customers = 100
num_articles = 20

# Simulate customer purchase history matrix
customer_article_matrix = pd.DataFrame(
    np.random.randint(0, 2, size=(num_customers, num_articles)),
    columns=[f"Article_{i}" for i in range(num_articles)]
)

# Simulate article features
article_features = pd.DataFrame({
    "Article": [f"Article_{i}" for i in range(num_articles)],
    "Gold": np.random.randint(0, 2, size=num_articles),
    "Silver": np.random.randint(0, 2, size=num_articles),
    "Heritage": np.random.randint(0, 2, size=num_articles)
})

# Simulate customer contact details
customer_data = pd.DataFrame({
    "CustomerID": [f"CUST_{i}" for i in range(num_customers)],
    "Email": [f"cust{i}@example.com" for i in range(num_customers)],
    "Phone": [f"555-010{i:02d}" for i in range(num_customers)],
    "Address": [f"Address {i}" for i in range(num_customers)]
})

# -----------------------------
# Helpers
# -----------------------------

def minmax_scale(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    mn, mx = np.min(x), np.max(x)
    rng = mx - mn
    if rng <= 1e-12:
        return np.zeros_like(x)
    return (x - mn) / (rng + 1e-12)


def create_customer_features(selected_customers_df):
    """
    Create customer features based on their purchase patterns for UpSet plot
    Works with filtered customer data (excluding those who already purchased the article)
    """
    if len(selected_customers_df) == 0:
        return pd.DataFrame()
    
    # Get customer indices from CustomerID
    customer_indices = []
    for cust_id in selected_customers_df['CustomerID']:
        idx = int(cust_id.split('_')[1])
        customer_indices.append(idx)
    
    # Get purchase data for selected customers
    selected_purchases = customer_article_matrix.iloc[customer_indices]
    
    # Create feature categories based on article features and purchase patterns
    customer_features = pd.DataFrame(index=selected_customers_df.index)
    
    # Feature 1: High Activity (purchased more than median number of articles)
    total_purchases = selected_purchases.sum(axis=1)
    median_purchases = total_purchases.median()
    if pd.isna(median_purchases):
        median_purchases = 0
    customer_features['High Activity'] = (total_purchases > median_purchases).fillna(False)
    
    # Feature 2: Gold Buyer (purchased at least one gold article)
    gold_articles = article_features[article_features['Gold'] == 1]['Article'].tolist()
    gold_columns = [col for col in selected_purchases.columns if col in gold_articles]
    if gold_columns:
        customer_features['Gold Buyer'] = (selected_purchases[gold_columns].sum(axis=1) > 0).fillna(False)
    else:
        customer_features['Gold Buyer'] = False
    
    # Feature 3: Silver Buyer (purchased at least one silver article)  
    silver_articles = article_features[article_features['Silver'] == 1]['Article'].tolist()
    silver_columns = [col for col in selected_purchases.columns if col in silver_articles]
    if silver_columns:
        customer_features['Silver Buyer'] = (selected_purchases[silver_columns].sum(axis=1) > 0).fillna(False)
    else:
        customer_features['Silver Buyer'] = False
    
    # Feature 4: Heritage Buyer (purchased at least one heritage article)
    heritage_articles = article_features[article_features['Heritage'] == 1]['Article'].tolist()
    heritage_columns = [col for col in selected_purchases.columns if col in heritage_articles]
    if heritage_columns:
        customer_features['Heritage Buyer'] = (selected_purchases[heritage_columns].sum(axis=1) > 0).fillna(False)
    else:
        customer_features['Heritage Buyer'] = False
    
    # Feature 5: High Propensity (score above 75th percentile)
    score_75th = selected_customers_df['Score'].quantile(0.75)
    if pd.isna(score_75th):
        score_75th = selected_customers_df['Score'].median()
    customer_features['High Propensity'] = (selected_customers_df['Score'] > score_75th).fillna(False)
    
    # Feature 6: Diverse Buyer (purchased from multiple categories)
    category_counts = pd.DataFrame(index=customer_features.index)
    if gold_columns:
        category_counts['Gold'] = selected_purchases[gold_columns].sum(axis=1) > 0
    else:
        category_counts['Gold'] = False
    if silver_columns:
        category_counts['Silver'] = selected_purchases[silver_columns].sum(axis=1) > 0  
    else:
        category_counts['Silver'] = False
    if heritage_columns:
        category_counts['Heritage'] = selected_purchases[heritage_columns].sum(axis=1) > 0
    else:
        category_counts['Heritage'] = False
        
    customer_features['Diverse Buyer'] = (category_counts.sum(axis=1) >= 2).fillna(False)
    
    # Ensure all columns are boolean and no NaN values
    for col in customer_features.columns:
        customer_features[col] = customer_features[col].astype(bool).fillna(False)
    
    return customer_features


def score_and_sort(article_number: str):
    """
    Compute ML + item-based CF scores and return a sorted dataframe by Score (desc).
    Excludes customers who have already purchased the selected article.
    This function is called when the user clicks 'Score Customers'.
    """
    X = customer_article_matrix.values  # (n_customers, n_articles)
    article_index = int(article_number.split("_")[1])

    # --- Exclude customers who already purchased the selected article ---
    already_purchased = customer_article_matrix.iloc[:, article_index] == 1
    eligible_customers = ~already_purchased
    
    # Store exclusion info in session state for display
    st.session_state.total_customers = len(customer_data)
    st.session_state.excluded_customers = already_purchased.sum()
    st.session_state.eligible_customers = eligible_customers.sum()
    
    # Filter data to only eligible customers
    X_eligible = X[eligible_customers]
    customer_data_eligible = customer_data[eligible_customers].reset_index(drop=True)
    
    if len(X_eligible) == 0:
        # All customers have already purchased this article
        return pd.DataFrame()

    # --- ML score (simulated) ---
    y = np.random.choice([0, 1], size=X_eligible.shape[0])
    model = RandomForestClassifier(random_state=42)
    model.fit(X_eligible, y)
    ml_scores = model.predict_proba(X_eligible)[:, 1] if len(
        model.classes_) > 1 else np.zeros(X_eligible.shape[0])

    # --- Item-based CF ---
    item_sim = cosine_similarity(X.T)         # Use full matrix for similarity
    sim_to_selected = item_sim[article_index]  # (n_articles,)
    cf_raw = X_eligible @ sim_to_selected     # Only for eligible customers
    cf_scores = minmax_scale(cf_raw)

    combined_scores = (ml_scores + cf_scores) / 2.0

    scored = customer_data_eligible.copy()
    scored["Score"] = combined_scores
    df_sorted = scored.sort_values(
        by="Score", ascending=False).reset_index(drop=True)

    return df_sorted


# -----------------------------
# UI
# -----------------------------
st.title("Customer Selection Tool for Marketing Campaigns")

article_number = st.selectbox(
    "Select Article Number", article_features["Article"])
channel = st.selectbox("Select Promotion Channel", [
                       "Telesales", "Email", "Direct Mail"])
cost_per_contact = st.number_input(
    "Cost per Contact (CAD)", min_value=0.0, value=1.0)
gross_margin = st.number_input(
    "Gross Margin of Product (EPS)", min_value=0.0, value=7.0)

# --- Initialize session state containers ---
if "df_sorted" not in st.session_state:
    st.session_state.df_sorted = None
if "last_scored_article" not in st.session_state:
    st.session_state.last_scored_article = None
if "selection_size" not in st.session_state:
    st.session_state.selection_size = 20
if "total_customers" not in st.session_state:
    st.session_state.total_customers = 0
if "excluded_customers" not in st.session_state:
    st.session_state.excluded_customers = 0
if "eligible_customers" not in st.session_state:
    st.session_state.eligible_customers = 0

# --- Trigger scoring and persist results ---
if st.button("Score Customers"):
    with st.spinner("Scoring customers..."):
        st.session_state.df_sorted = score_and_sort(article_number)
        st.session_state.last_scored_article = article_number
        
        if len(st.session_state.df_sorted) > 0:
            # Compute optimal k for current cost/margin and prefill the slider once
            cum_conv = st.session_state.df_sorted["Score"].cumsum().values
            ks = np.arange(1, len(st.session_state.df_sorted) + 1)
            total_costs = ks * cost_per_contact
            expected_revenues = cum_conv * gross_margin
            rois = np.where(total_costs > 0, (expected_revenues -
                            total_costs) / total_costs, 0.0)
            optimal_k = int(ks[np.argmax(rois)])
            st.session_state.selection_size = int(np.clip(optimal_k, 1, len(st.session_state.df_sorted)))

# --- Render results if we have them ---
if st.session_state.df_sorted is not None:
    # Warn user if article changed after scoring (and don't show outdated filtering info)
    if article_number != st.session_state.last_scored_article:
        st.info(
            "Article changed. Click **Score Customers** to refresh rankings for the new article.")
    else:
        # Only show exclusion information if we're viewing results for the current article
        if st.session_state.total_customers > 0:
            st.info(
                f"**Customer Filtering:** {st.session_state.excluded_customers} customers excluded "
                f"(already purchased {st.session_state.last_scored_article}). "
                f"{st.session_state.eligible_customers} eligible customers available for targeting."
            )
    
    # Check if no eligible customers (only for current article)
    if article_number == st.session_state.last_scored_article and len(st.session_state.df_sorted) == 0:
        st.warning(f"⚠️ All customers have already purchased {st.session_state.last_scored_article}. No customers available for targeting!")
        st.stop()

    df_sorted = st.session_state.df_sorted

    # Consistent ROI calculation for both optimal k and live updates
    def calculate_roi_metrics(cost_per_contact, gross_margin):
        """Calculate ROI metrics consistently across the app"""
        cum_conv = df_sorted["Score"].cumsum().values
        ks = np.arange(1, len(df_sorted) + 1)
        total_costs = ks * cost_per_contact
        expected_revenues = cum_conv * gross_margin
        rois = np.where(total_costs > 0, (expected_revenues - total_costs) / total_costs, 0.0)
        return ks, total_costs, expected_revenues, rois, cum_conv
    
    # ROI Threshold Control
    roi_threshold = st.slider("ROI Threshold for Customer Selection", min_value=0.0, max_value=5.0, value=3.0, step=0.1,
                             help="Automatically selects customers up to the point where ROI falls below this threshold")
    
    # Calculate current ROI metrics
    ks, total_costs, expected_revenues, rois, cum_conv = calculate_roi_metrics(cost_per_contact, gross_margin)
    optimal_k = int(ks[np.argmax(rois)])

    st.subheader("Selection Size")

    # Determine selection size based on ROI threshold
    roi_crosses_threshold = np.any(rois >= roi_threshold)
    
    if roi_crosses_threshold:
        threshold_indices = np.where(rois >= roi_threshold)[0]
        auto_selection_size = threshold_indices[-1] + 1
        threshold_info = f"Auto-selected {auto_selection_size} customers (last k where ROI ≥ {roi_threshold:.1f})"
    else:
        auto_selection_size = optimal_k
        max_roi = np.max(rois)
        threshold_info = f"ROI never reaches {roi_threshold:.1f}. Using optimal k={optimal_k} (max ROI={max_roi:.2f})"
    
    st.session_state.selection_size = auto_selection_size
    st.info(threshold_info)
    
    selection_size = int(st.session_state.selection_size)
    selected_customers = df_sorted.head(selection_size)

    # KPIs for chosen k
    total_cost = selection_size * cost_per_contact
    expected_conversion_total = cum_conv[selection_size - 1]
    expected_revenue = expected_conversion_total * gross_margin
    roi = (expected_revenue - total_cost) / total_cost if total_cost > 0 else 0.0

    # Dashboard
    st.subheader("Dashboard")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Channel", channel)
    col2.metric("Eligible Pool", f"{st.session_state.eligible_customers}")
    col3.metric("Total Cost (CAD)", f"{total_cost:.2f}")
    col4.metric("Expected Revenue (CAD)", f"{expected_revenue:.2f}")
    col5.metric("ROI", f"{roi:.2f}")

    # Score distribution with selection visualization
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    
    all_scores = df_sorted["Score"].values
    selected_scores = selected_customers["Score"].values
    
    bins = 20
    hist_range = (all_scores.min(), all_scores.max())
    
    ax1.hist(all_scores, bins=bins, range=hist_range, color='lightgray', 
             edgecolor='black', alpha=0.7, label='All Eligible Customers')
    
    ax1.hist(selected_scores, bins=bins, range=hist_range, color='skyblue', 
             edgecolor='navy', alpha=0.8, label=f'Selected Customers (n={len(selected_scores)})')
    
    ax1.set_title("Customer Propensity Scores - Selection Overview")
    ax1.set_xlabel("Propensity Score")
    ax1.set_ylabel("Number of Customers")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    if len(selected_scores) > 0:
        min_selected_score = selected_scores.min()
        ax1.axvline(min_selected_score, color='red', linestyle='--', alpha=0.7,
                   label=f'Selection Threshold (Score ≥ {min_selected_score:.3f})')
        ax1.legend()
    
    st.pyplot(fig1)

    # ROI curve vs k
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.plot(ks, rois, color='purple', linewidth=2)
    ax2.set_title("ROI vs. Selection Size (k)")
    ax2.set_xlabel("k (Top-k Customers by Score)")
    ax2.set_ylabel("ROI")
    ax2.grid(True, alpha=0.3)
    
    current_roi = rois[selection_size - 1]
    ax2.scatter([selection_size], [current_roi], color='blue', s=100, zorder=5, 
               label=f'Auto Selection (k={selection_size}, ROI={current_roi:.2f})')
    
    ax2.axhline(roi_threshold, color='red', linestyle='--', alpha=0.7,
                label=f'ROI Threshold = {roi_threshold:.1f}')
    
    if roi_crosses_threshold:
        threshold_indices = np.where(rois >= roi_threshold)[0]
        first_k = ks[threshold_indices[0]]
        last_k = ks[threshold_indices[-1]] if len(threshold_indices) > 1 else first_k
        
        ax2.scatter([first_k], [roi_threshold], color='red', s=80, zorder=5, 
                   marker='o', label=f'First crossing at k={first_k}')
        
        if last_k != first_k:
            ax2.scatter([last_k], [roi_threshold], color='darkred', s=80, zorder=5, 
                       marker='s', label=f'Last crossing at k={last_k}')
            
        valid_region = ks[threshold_indices]
        valid_rois = rois[threshold_indices]
        ax2.fill_between(valid_region, roi_threshold, valid_rois, alpha=0.2, color='green',
                        label='Selected region (ROI ≥ threshold)')

    ax2.legend()
    st.pyplot(fig2)

    # UpSet Plot - Customer Feature Analysis
    st.subheader("Customer Feature Analysis")
    
    if UPSETPLOT_AVAILABLE and len(selected_customers) > 0:
        try:
            customer_features = create_customer_features(selected_customers)
            
            if len(customer_features) > 0:
                feature_counts = customer_features.sum()
                valid_features = feature_counts[feature_counts > 0]
                
                if len(valid_features) == 0:
                    st.warning("No customers have any of the analyzed features. Try selecting more customers or adjusting the ROI threshold.")
                else:
                    customer_features_filtered = customer_features[valid_features.index]
                    
                    memberships = []
                    for i in range(len(customer_features_filtered)):
                        customer_row = customer_features_filtered.iloc[i]
                        member_features = customer_row[customer_row == True].index.tolist()
                        memberships.append(member_features)
                    
                    if not all(len(m) == 0 for m in memberships):
                        upset_data = upsetplot.from_memberships(
                            memberships,
                            data=selected_customers['Score'].values
                        )
                        
                        if len(upset_data) > 0:
                            fig3 = plt.figure(figsize=(12, 8))
                            upset_plot = upsetplot.UpSet(upset_data, subset_size='count', show_counts=True)
                            upset_plot.plot(fig=fig3)
                            plt.suptitle(f"Feature Intersections Among Selected Customers (n={len(selected_customers)})", 
                                        fontsize=14, y=0.98)
                            st.pyplot(fig3)
                            
                            # Feature summary
                            st.subheader("Feature Summary")
                            feature_summary = pd.DataFrame({
                                'Feature': customer_features_filtered.columns,
                                'Count': customer_features_filtered.sum(),
                                'Percentage': (customer_features_filtered.sum() / len(customer_features_filtered) * 100).round(1)
                            }).sort_values('Count', ascending=False)
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.dataframe(feature_summary, hide_index=True)
                            
                            with col2:
                                st.write("**Top Intersections:**")
                                intersection_sizes = upset_data.groupby(level=list(range(upset_data.index.nlevels))).size()
                                top_intersections = intersection_sizes.sort_values(ascending=False).head(5)
                                
                                for idx, size in top_intersections.items():
                                    features = [customer_features_filtered.columns[i] for i, val in enumerate(idx) if val]
                                    if len(features) > 0:
                                        st.write(f"• {' ∩ '.join(features)}: {size} customers")
                                    else:
                                        st.write(f"• No features: {size} customers")
                        else:
                            st.warning("Unable to create UpSet plot: no valid intersections found.")
                    else:
                        st.warning("All selected customers have no features.")
            else:
                st.warning("Unable to create customer features.")
                
        except Exception as e:
            st.error(f"Error creating UpSet plot: {str(e)}")
            st.info("Install upsetplot library with: `pip install upsetplot`")
    
    elif not UPSETPLOT_AVAILABLE:
        st.warning("UpSet plot requires the `upsetplot` library. Install with: `pip install upsetplot`")
        
        # Show basic feature summary instead
        try:
            customer_features = create_customer_features(selected_customers)
            if len(customer_features) > 0:
                st.subheader("Basic Feature Summary")
                feature_summary = pd.DataFrame({
                    'Feature': customer_features.columns,
                    'Count': customer_features.sum(),
                    'Percentage': (customer_features.sum() / len(customer_features) * 100).round(1)
                }).sort_values('Count', ascending=False)
                st.dataframe(feature_summary, hide_index=True)
        except Exception as e:
            st.warning(f"Could not create feature summary: {str(e)}")
    
    else:
        st.info("No customers selected for feature analysis.")

    # Export
    st.subheader("Export Selected Customers")
    
    if len(selected_customers) > 0:
        try:
            customer_features = create_customer_features(selected_customers)
            if len(customer_features) > 0:
                export_data = selected_customers.copy()
                for col in customer_features.columns:
                    export_data[col] = customer_features[col].values
                st.dataframe(export_data)
                csv = export_data.to_csv(index=False).encode('utf-8')
                st.download_button("Download CSV (with features)", data=csv,
                                 file_name="selected_customers_with_features.csv", mime="text/csv")
            else:
                st.dataframe(selected_customers)
                csv = selected_customers.to_csv(index=False).encode('utf-8')
                st.download_button("Download CSV", data=csv,
                                 file_name="selected_customers.csv", mime="text/csv")
        except Exception as e:
            st.dataframe(selected_customers)
            csv = selected_customers.to_csv(index=False).encode('utf-8')
            st.download_button("Download CSV", data=csv,
                             file_name="selected_customers.csv", mime="text/csv")
    else:
        st.info("No data to export.")

else:
    st.caption("Press **Score Customers** to compute scores and show dashboards.")
