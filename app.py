import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

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
    ml_scores = model.predict_proba(X)[:, 1] if len(model.classes_) > 1 else np.zeros(X.shape[0])

    # --- Item-based CF ---
    article_index = int(article_number.split("_")[1])
    item_sim = cosine_similarity(X.T)         # (n_articles x n_articles)
    sim_to_selected = item_sim[article_index] # (n_articles,)
    cf_raw = X @ sim_to_selected              # (n_customers,)
    cf_scores = minmax_scale(cf_raw)

    combined_scores = (ml_scores + cf_scores) / 2.0

    scored = customer_data.copy()
    scored["Score"] = combined_scores
    df_sorted = scored.sort_values(by="Score", ascending=False).reset_index(drop=True)

    return df_sorted

# -----------------------------
# UI
# -----------------------------
st.title("Customer Selection Tool for Marketing Campaigns")

article_number = st.selectbox("Select Article Number", article_features["Article"])
channel = st.selectbox("Select Promotion Channel", ["Telesales", "Email", "Direct Mail"])
cost_per_contact = st.number_input("Cost per Contact (CAD)", min_value=0.0, value=1.0)
gross_margin = st.number_input("Gross Margin of Product (CAD)", min_value=0.0, value=10.0)

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
    rois = np.where(total_costs > 0, (expected_revenues - total_costs) / total_costs, 0.0)
    optimal_k = int(ks[np.argmax(rois)])
    st.session_state.selection_size = int(np.clip(optimal_k, 1, num_customers))

# --- Render results if we have them ---
if st.session_state.df_sorted is not None:
    # Warn user if article changed after scoring
    if article_number != st.session_state.last_scored_article:
        st.info("Article changed. Click **Score Customers** to refresh rankings for the new article.")

    df_sorted = st.session_state.df_sorted

    # Recompute ROI curve live with current cost/margin (scores stay the same)
    cum_conv = df_sorted["Score"].cumsum().values
    ks = np.arange(1, len(df_sorted) + 1)
    total_costs = ks * cost_per_contact
    expected_revenues = cum_conv * gross_margin
    rois = np.where(total_costs > 0, (expected_revenues - total_costs) / total_costs, 0.0)
    optimal_k = int(ks[np.argmax(rois)])

    st.subheader("Selection Size")
    st.caption("The suggested k updates when Cost/Margin change. Move the slider to explore trade-offs.")
    # NOTE: no 'value=' here. The current value lives in st.session_state.selection_size.
    st.slider(
        "Select Number of Customers",
        min_value=1,
        max_value=num_customers,
        key="selection_size"
    )

    selection_size = int(st.session_state.selection_size)
    selected_customers = df_sorted.head(selection_size)

    # KPIs for chosen k (use current inputs)
    total_cost = selection_size * cost_per_contact
    expected_conversion = selected_customers["Score"].mean()
    expected_revenue = expected_conversion * selection_size * gross_margin
    roi = (expected_revenue - total_cost) / total_cost if total_cost > 0 else 0.0

    # Dashboard
    st.subheader("Dashboard")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Channel", channel)
    col2.metric("Total Cost (CAD)", f"{total_cost:.2f}")
    col3.metric("Expected Revenue (CAD)", f"{expected_revenue:.2f}")
    col4.metric("ROI", f"{roi:.2f}")

    # Score distribution
    fig1, ax1 = plt.subplots()
    ax1.hist(df_sorted["Score"], bins=20, color='skyblue', edgecolor='black')
    ax1.set_title("Customer Propensity Scores")
    ax1.set_xlabel("Score")
    ax1.set_ylabel("Number of Customers")
    st.pyplot(fig1)

    # ROI curve vs k
    fig2, ax2 = plt.subplots()
    ax2.plot(ks, rois, color='purple')
    ax2.axvline(optimal_k, color='green', linestyle='--', label=f"Optimal k = {optimal_k}")
    ax2.set_title("ROI vs. Selection Size (k)")
    ax2.set_xlabel("k (Top-k Customers by Score)")
    ax2.set_ylabel("ROI")
    ax2.legend()
    st.pyplot(fig2)

    # Export
    st.subheader("Export Selected Customers")
    st.dataframe(selected_customers)
    csv = selected_customers.to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV", data=csv, file_name="selected_customers.csv", mime="text/csv")

else:
    st.caption("Press **Score Customers** to compute scores and show dashboards.")
