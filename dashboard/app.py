import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from core.config import settings
from core.model_loader import (
    load_default_model, load_labeled_customers, load_segment_profiles, load_raw_rfm,
)
from core.clustering import cluster_with_k

st.set_page_config(page_title="Customer Segmentation", page_icon="🧩", layout="wide")

st.title("🧩 Customer Segmentation Dashboard")
st.caption("RFM-based clustering — group customers by Recency, Frequency, and Monetary value.")

customers = load_labeled_customers()
profiles = load_segment_profiles()

tab_overview, tab_explore, tab_lookup = st.tabs(
    ["📊 Segment Overview", "🔍 Explore Different k", "🧑 Customer Lookup"]
)


def plot_clusters(df: pd.DataFrame, profiles_df: pd.DataFrame, title: str):
    fig, ax = plt.subplots(figsize=(7, 5.5))
    cmap = plt.get_cmap("tab10")

    name_by_cluster = dict(zip(profiles_df["Cluster"], profiles_df["segment_name"]))

    for i, cluster_id in enumerate(sorted(df["Cluster"].unique())):
        subset = df[df["Cluster"] == cluster_id]
        label = name_by_cluster.get(cluster_id, f"Cluster {cluster_id}")
        ax.scatter(subset["pca_x"], subset["pca_y"], s=18, alpha=0.7,
                   color=cmap(i % 10), label=label)

    ax.set_xlabel("PCA Component 1")
    ax.set_ylabel("PCA Component 2")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=9, frameon=True)
    ax.grid(alpha=0.2)
    return fig


# ---------------------------------------------------------------------------
# Tab 1: Segment Overview (the saved/default model)
# ---------------------------------------------------------------------------
with tab_overview:
    st.subheader("Segments from the trained model")

    col_chart, col_profiles = st.columns([1.1, 1])

    with col_chart:
        fig = plot_clusters(customers, profiles, "Customer segments (PCA-reduced)")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with col_profiles:
        st.markdown("**Segment profiles**")
        for _, row in profiles.sort_values("composite_score", ascending=False).iterrows():
            with st.container(border=True):
                st.markdown(f"**{row['segment_name']}** — {int(row['size'])} customers")
                st.caption(
                    f"Avg Recency: {row['avg_recency']:.0f} days · "
                    f"Avg Frequency: {row['avg_frequency']:.1f} orders · "
                    f"Avg Monetary: ${row['avg_monetary']:,.0f}"
                )
                st.write(row["suggested_action"])

    st.markdown("---")
    st.markdown("**Full segment table**")
    st.dataframe(profiles, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Tab 2: Explore Different k
# ---------------------------------------------------------------------------
with tab_explore:
    st.subheader("See how the segmentation changes with a different k")
    st.caption(
        "Re-clusters live using the raw RFM table — doesn't touch the saved "
        "model above, so you can experiment freely."
    )

    k = st.slider("Number of clusters (k)", settings.MIN_K, settings.MAX_K, 4)

    if st.button("Run Clustering", type="primary"):
        rfm = load_raw_rfm()
        with st.spinner(f"Clustering with k={k}..."):
            result, live_profiles, sil_score = cluster_with_k(rfm, k)

        st.session_state.explore_result = result
        st.session_state.explore_profiles = live_profiles
        st.session_state.explore_sil = sil_score
        st.session_state.explore_k = k

    if "explore_result" in st.session_state:
        st.metric(f"Silhouette score (k={st.session_state.explore_k})",
                   f"{st.session_state.explore_sil:.4f}")

        col1, col2 = st.columns([1.1, 1])
        with col1:
            fig2 = plot_clusters(
                st.session_state.explore_result, st.session_state.explore_profiles,
                f"k={st.session_state.explore_k} segments",
            )
            st.pyplot(fig2, use_container_width=True)
            plt.close(fig2)
        with col2:
            st.dataframe(
                st.session_state.explore_profiles, use_container_width=True, hide_index=True,
            )

        st.caption(
            "Higher silhouette score (closer to 1) means better-separated, more "
            "cohesive clusters. Compare this to the saved model's k to decide "
            "if a different k would segment customers more cleanly."
        )


# ---------------------------------------------------------------------------
# Tab 3: Customer Lookup
# ---------------------------------------------------------------------------
with tab_lookup:
    st.subheader("Look up a specific customer's segment")

    customer_id = st.text_input("Customer ID")

    if customer_id:
        match = customers[customers["CustomerID"].astype(str) == customer_id.strip()]
        if match.empty:
            st.warning("No customer found with that ID in the trained model's data.")
        else:
            row = match.iloc[0]
            profile_row = profiles[profiles["Cluster"] == row["Cluster"]].iloc[0]

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Segment", profile_row["segment_name"])
            col2.metric("Recency", f"{row['Recency']:.0f} days")
            col3.metric("Frequency", f"{row['Frequency']:.0f} orders")
            col4.metric("Monetary", f"${row['Monetary']:,.0f}")

            st.info(f"**Suggested action:** {profile_row['suggested_action']}")
