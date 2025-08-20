import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import upsetplot

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
    """
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
    
    # Ensure all columns are boolean and no NaN values
    for col in customer_features.columns:
        customer_features[col] = customer_features[col].astype(bool).fillna(False)
    
    return customer_features


def score_and_sort(article_number: str):
    """
    Compute ML + item-based CF scores and return a sorted dataframe by Score (desc).
    This function is called when the user clicks 'Score Customers'.
    """
    X = customer_article_matrix.values  # (n_customers, n_articles)

    # --- ML score (simulated) ---
    y = np.random.choice([0, 1], size=X.shape[0])
    model = RandomForestClassifier()
    model.fit(X, y)
    ml_scores = model.predict_proba(X)[:, 1] if len(
        model.classes_) > 1 else np.zeros(X.shape[0])

    # --- Item-based CF ---
    article_index = int(article_number.split("_")[1])
    item_sim = cosine_similarity(X.T)         # (n_articles x n_articles)
    sim_to_selected = item_sim[article_index]  # (n_articles,)
    cf_raw = X @ sim_to_selected              # (n_customers,)
    cf_scores = minmax_scale(cf_raw)

    combined_scores = (ml_scores + cf_scores) / 2.0

    scored = customer_data.copy()
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
    st.session_state.selection_size = 20  # default before first scoring

# --- Trigger scoring and persist results ---
if st.button("Score Customers"):
    st.session_state.df_sorted = score_and_sort(article_number)
    st.session_state.last_scored_article = article_number
    # Compute optimal k for current cost/margin and prefill the slider once
    cum_conv = st.session_state.df_sorted["Score"].cumsum().values
    ks = np.arange(1, len(st.session_state.df_sorted) + 1)
    total_costs = ks * cost_per_contact
    expected_revenues = cum_conv * gross_margin
    rois = np.where(total_costs > 0, (expected_revenues -
                    total_costs) / total_costs, 0.0)
    optimal_k = int(ks[np.argmax(rois)])
    st.session_state.selection_size = int(np.clip(optimal_k, 1, num_customers))

# --- Render results if we have them ---
if st.session_state.df_sorted is not None:
    # Warn user if article changed after scoring
    if article_number != st.session_state.last_scored_article:
        st.info(
            "Article changed. Click **Score Customers** to refresh rankings for the new article.")

    df_sorted = st.session_state.df_sorted

    # Consistent ROI calculation for both optimal k and live updates
    def calculate_roi_metrics(cost_per_contact, gross_margin):
        """Calculate ROI metrics consistently across the app"""
        cum_conv = df_sorted["Score"].cumsum().values  # Cumulative conversion probability
        ks = np.arange(1, len(df_sorted) + 1)
        total_costs = ks * cost_per_contact
        expected_revenues = cum_conv * gross_margin  # No artificial inflation
        rois = np.where(total_costs > 0, (expected_revenues - total_costs) / total_costs, 0.0)
        return ks, total_costs, expected_revenues, rois, cum_conv
    
    # ROI Threshold Control (moved here before calculations)
    roi_threshold = st.slider("ROI Threshold for Customer Selection", min_value=0.0, max_value=5.0, value=3.0, step=0.1,
                             help="Automatically selects customers up to the point where ROI falls below this threshold")
    
    # Calculate current ROI metrics
    ks, total_costs, expected_revenues, rois, cum_conv = calculate_roi_metrics(cost_per_contact, gross_margin)
    optimal_k = int(ks[np.argmax(rois)])

    st.subheader("Selection Size")

    # Determine selection size based on ROI threshold
    roi_crosses_threshold = np.any(rois >= roi_threshold)
    
    if roi_crosses_threshold:
        # Find all k values where ROI >= threshold
        threshold_indices = np.where(rois >= roi_threshold)[0]
        auto_selection_size = threshold_indices[-1] + 1  # Last k where ROI >= threshold
        threshold_info = f"Auto-selected {auto_selection_size} customers (last k where ROI ≥ {roi_threshold:.1f})"
    else:
        # If threshold never reached, use optimal k
        auto_selection_size = optimal_k
        max_roi = np.max(rois)
        threshold_info = f"ROI never reaches {roi_threshold:.1f}. Using optimal k={optimal_k} (max ROI={max_roi:.2f})"
    
    # Update session state with auto-selected size
    st.session_state.selection_size = auto_selection_size
    
    st.info(threshold_info)
    
    selection_size = int(st.session_state.selection_size)
    selected_customers = df_sorted.head(selection_size)

    # KPIs for chosen k (consistent with ROI curve calculation)
    total_cost = selection_size * cost_per_contact
    expected_conversion_total = cum_conv[selection_size - 1]  # Cumulative conversion up to k
    expected_revenue = expected_conversion_total * gross_margin
    roi = (expected_revenue - total_cost) / total_cost if total_cost > 0 else 0.0

    # Dashboard
    st.subheader("Dashboard")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Channel", channel)
    col2.metric("Total Cost (CAD)", f"{total_cost:.2f}")
    col3.metric("Expected Revenue (CAD)", f"{expected_revenue:.2f}")
    col4.metric("ROI", f"{roi:.2f}")

    # Score distribution with selection visualization
    fig1, ax1 = plt.subplots()
    
    # Get all scores and selected scores
    all_scores = df_sorted["Score"].values
    selected_scores = selected_customers["Score"].values
    
    # Create histogram bins
    bins = 20
    hist_range = (all_scores.min(), all_scores.max())
    
    # Plot all customers (background)
    ax1.hist(all_scores, bins=bins, range=hist_range, color='lightgray', 
             edgecolor='black', alpha=0.7, label='All Customers')
    
    # Overlay selected customers
    ax1.hist(selected_scores, bins=bins, range=hist_range, color='skyblue', 
             edgecolor='navy', alpha=0.8, label=f'Selected Customers (n={len(selected_scores)})')
    
    ax1.set_title("Customer Propensity Scores - Selection Overview")
    ax1.set_xlabel("Propensity Score")
    ax1.set_ylabel("Number of Customers")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add selection threshold line (minimum score of selected customers)
    if len(selected_scores) > 0:
        min_selected_score = selected_scores.min()
        ax1.axvline(min_selected_score, color='red', linestyle='--', alpha=0.7,
                   label=f'Selection Threshold (Score ≥ {min_selected_score:.3f})')
        ax1.legend()
    
    st.pyplot(fig1)

    # ROI curve vs k
    fig2, ax2 = plt.subplots()
    ax2.plot(ks, rois, color='purple', linewidth=2)
    ax2.set_title("ROI vs. Selection Size (k)")
    ax2.set_xlabel("k (Top-k Customers by Score)")
    ax2.set_ylabel("ROI")
    ax2.grid(True, alpha=0.3)
    
    # Mark current auto-selection on the curve
    current_roi = rois[selection_size - 1]
    ax2.scatter([selection_size], [current_roi], color='blue', s=100, zorder=5, 
               label=f'Auto Selection (k={selection_size}, ROI={current_roi:.2f})')
    
    # Add horizontal ROI threshold line
    ax2.axhline(roi_threshold, color='red', linestyle='--', alpha=0.7,
                label=f'ROI Threshold = {roi_threshold:.1f}')
    
    # Find and mark intersection points
    if roi_crosses_threshold:
        # Find all k values where ROI >= threshold
        threshold_indices = np.where(rois >= roi_threshold)[0]
        first_k = ks[threshold_indices[0]]
        last_k = ks[threshold_indices[-1]] if len(threshold_indices) > 1 else first_k
        
        # Mark intersection points
        ax2.scatter([first_k], [roi_threshold], color='red', s=80, zorder=5, 
                   marker='o', label=f'First crossing at k={first_k}')
        
        if last_k != first_k:
            ax2.scatter([last_k], [roi_threshold], color='darkred', s=80, zorder=5, 
                       marker='s', label=f'Last crossing at k={last_k}')
            
        # Highlight the selected region
        valid_region = ks[threshold_indices]
        valid_rois = rois[threshold_indices]
        ax2.fill_between(valid_region, roi_threshold, valid_rois, alpha=0.2, color='green',
                        label='Selected region (ROI ≥ threshold)')
            
    else:
        # Show max ROI if threshold never reached
        max_roi_index = np.argmax(rois)
        max_k = ks[max_roi_index]
        max_roi = rois[max_roi_index]

    ax2.legend()
    st.pyplot(fig2)

    # UpSet Plot - Customer Feature Analysis
    st.subheader("Customer Feature Analysis - UpSet Plot")
    
    if len(selected_customers) > 0:
        try:
            # Create customer features
            customer_features = create_customer_features(selected_customers)
            
            # Ensure we have at least one feature with True values
            feature_counts = customer_features.sum()
            valid_features = feature_counts[feature_counts > 0]
            
            if len(valid_features) == 0:
                st.warning("No customers have any of the analyzed features. Try selecting more customers or adjusting the ROI threshold.")
            else:
                # Filter to only include features that have at least one True value
                customer_features_filtered = customer_features[valid_features.index]
                
                # Convert to the format needed for upsetplot
                # Create membership lists for each customer
                memberships = []
                for i in range(len(customer_features_filtered)):
                    customer_row = customer_features_filtered.iloc[i]
                    member_features = customer_row[customer_row == True].index.tolist()
                    memberships.append(member_features)
                
                # Handle case where some customers have no features
                if all(len(m) == 0 for m in memberships):
                    st.warning("All selected customers have no features. This might indicate an issue with the feature calculation.")
                else:
                    # Create upset data with proper error handling
                    upset_data = upsetplot.from_memberships(
                        memberships,
                        data=selected_customers['Score'].values
                    )
                    
                    if len(upset_data) > 0:
                        # Create the UpSet plot
                        fig3 = plt.figure(figsize=(12, 8))
                        
                        # Plot the UpSet plot with subset sizes and aggregated scores
                        upset_plot = upsetplot.UpSet(upset_data, subset_size='count', show_counts=True)
                        upset_plot.plot(fig=fig3)
                        
                        plt.suptitle(f"Feature Intersections Among Selected Customers (n={len(selected_customers)})", 
                                    fontsize=14, y=0.98)
                        
                        st.pyplot(fig3)
                        
                        # Show feature summary statistics
                        st.subheader("Feature Summary")
                        feature_summary = pd.DataFrame({
                            'Feature': customer_features_filtered.columns,
                            'Count': customer_features_filtered.sum(),
                            'Percentage': (customer_features_filtered.sum() / len(customer_features_filtered) * 100).round(1)
                        })
                        feature_summary = feature_summary.sort_values('Count', ascending=False)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.dataframe(feature_summary, hide_index=True)
                        
                        with col2:
                            # Show intersection details
                            st.write("**Top Intersections:**")
                            intersection_sizes = upset_data.groupby(level=list(range(upset_data.index.nlevels))).size()
                            top_intersections = intersection_sizes.sort_values(ascending=False).head(5)
                            
                            intersection_count = 0
                            for idx, size in top_intersections.items():
                                if intersection_count >= 5:
                                    break
                                features = [customer_features_filtered.columns[i] for i, val in enumerate(idx) if val]
                                if len(features) > 0:
                                    st.write(f"• {' ∩ '.join(features)}: {size} customers")
                                else:
                                    st.write(f"• No features: {size} customers")
                                intersection_count += 1
                    else:
                        st.warning("Unable to create UpSet plot: no valid intersections found.")
            
        except Exception as e:
            st.error(f"Error creating UpSet plot: {str(e)}")
            st.info("This might occur if there are too few customers selected or if the upsetplot library is not installed. To install: `pip install upsetplot`")
            
            # Show debug information
            with st.expander("Debug Information"):
                st.write("Selected customers shape:", selected_customers.shape)
                try:
                    debug_features = create_customer_features(selected_customers)
                    st.write("Customer features shape:", debug_features.shape)
                    st.write("Customer features summary:")
                    st.dataframe(debug_features.describe())
                    st.write("Feature value counts:")
                    for col in debug_features.columns:
                        st.write(f"{col}: {debug_features[col].sum()} True, {(~debug_features[col]).sum()} False")
                except Exception as debug_e:
                    st.write("Error in feature creation:", str(debug_e))
    
    else:
        st.info("No customers selected for UpSet plot analysis.")

    # Export
    st.subheader("Export Selected Customers")
    
    # Add feature information to export data
    if len(selected_customers) > 0:
        try:
            customer_features = create_customer_features(selected_customers)
            export_data = selected_customers.copy()
            for col in customer_features.columns:
                export_data[col] = customer_features[col].values
            st.dataframe(export_data)
            csv = export_data.to_csv(index=False).encode('utf-8')
            st.download_button("Download CSV (with features)", data=csv,
                             file_name="selected_customers_with_features.csv", mime="text/csv")
        except:
            st.dataframe(selected_customers)
            csv = selected_customers.to_csv(index=False).encode('utf-8')
            st.download_button("Download CSV", data=csv,
                             file_name="selected_customers.csv", mime="text/csv")
    else:
        st.info("No data to export.")

else:
    st.caption("Press **Score Customers** to compute scores and show dashboards.")
