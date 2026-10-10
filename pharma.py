import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

st.markdown("""
    <style>
    .stApp {
        background-color: #F5FFFA;
    }
    </style>
""", unsafe_allow_html=True)



st.set_page_config(
    page_title="Getz Pharma Project",
    page_icon="🥼😷",
    layout="wide"
)


st.title("🥼😷 Getz Pharma Project")

st.subheader("Unsupervised Project K-Means")

st.caption("")

st.info("")

# load data

DATA_FILE = "getz_pharma_customer_segmentation_1000.csv"

df = pd.read_csv(DATA_FILE)

# DATASET OVERVIEW

m1, m2, m3, m4 = st.columns(4)
with m1:
    st.metric("Total Customers", len(df), "👥")
with m2:
    st.metric("Top City", df['City'].mode()[0])
with m3:
    st.metric("Max Monthly Spend", f"PKR {df['Monthly_Spend_PKR'].max():,.0f}")
with m4:
    st.metric("Avg Orders", f"{df['Monthly_Orders'].mean():.1f}")


# creating tabs

tab1, tab2, tab3, tab4 = st.tabs(
    
[
    "📊 Data Overview",
    "📈 Elbow Method",
    "👥 Segments Result",
    "💡 Business Insights"
]
)


with tab1:
    st.header("Step1 : Understand the Data")

    st.write(
        "Before using Machine Learing, we first understand what every"
        "column means. A good Data Scientist should never train a model"
        "without understanding the data"
    )

    st.dataframe(df.head(20), width ="stretch")

    st.subheader("Missing Values Check")

    missing_values = df.isna().sum()

    st.dataframe(
        missing_values.reset_index().rename(
            columns={"index": "Column", 0: "Missing_Values"}
        ),
        width = "stretch",
        hide_index=True
    )


# features for K-Means

features = [
    "Monthly_Orders",
    "Avg_Order_Value_PKR",
    "Products_Purchased",
    "Payment_Delay_Days",
    "Days_Since_Last_Order"
]

X = df[features]

# scaling features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


with tab2:
    st.header("Step 2: How many groups should we make?")

    st.subheader("Elbow Method")

    st.write(
        "We can train K-Mean with different values of K ad compare"
        "Inertia: Inertia tells us how tightly customers sit around"
        "their cluster centres. Lower is better, but we do not keep adding "
        "clustering forever. We look for a bend, or 'elbow', where the"
        "improvement starts slowing down."
    )


    # will test k from 1 to 7
    k_values = list(range(1,8))
    inertia_values = []


    # Train a small K-Means model for every K value.
    # we only save the inertia so that we can draw the elbo chart

    for k in k_values:
        elbow_model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )
        elbow_model.fit(X_scaled)
        inertia_values.append(elbow_model.inertia_)

    elbow_df = pd.DataFrame(
        {
            "K" : k_values,
            "Inertia" : inertia_values
        }
    )

    st.line_chart(
        elbow_df,
        x="K",
        y="Inertia",
        x_label="Number of Clusters (K)",
        y_label="Inertia"
    )

    st.success(
        "For this syntetic classroom dataset, the chart has a clear bend"
        "aound K=3. That makes three clusters a sensible teaching choice."
    )

    # clusters

    selected_k = st.slider(
        "Choose the number of clusters (K)",
        min_value = 2,
        max_value = 8,
        value = 5,
        step=1
    )



# n_clusters

model = KMeans(
    n_clusters=selected_k,
    random_state=101,
    n_init=10
)
df["Cluster_ID"] = model.fit_predict(X_scaled)
cluster_names = {
    0: "Premium",
    1: "Regular Buyers", 
    2: "At-Risk Clients",
    3: "New Potentials",
    4: "High-Value Champ"
}
df["Segment"] = df["Cluster_ID"].map(cluster_names)


# add cluster result to a copy of the data


df_clustered = df.copy()
df_clustered["Cluster"] =df_clustered["Segment"]


# creating a cluster summary

cluster_summary =(
    df_clustered.groupby("Cluster") 
    .agg(
        Total_Customers=("Customer_ID", "count"),
        avg_Monthly_Orders=("Monthly_Orders","mean"),
        Avg_Order_Value_PKR=("Avg_Order_Value_PKR","mean"),
        Product_Purchased = ("Products_Purchased", "mean"),
        Payment_Delay_Days=("Days_Since_Last_Order","mean"),
        Monthly_Spend_PKR= ("Monthly_Spend_PKR","mean"),
    )
    .round(1)
)


## Lowest spend -> Low Engagement
## Middle Spend -> Regular Customer
## Highest Spend - > High Value Loyal


if selected_k == 3:
    ordered_clusters = (
        cluster_summary["Monthly_Spend_PKR"]
        .sort_values()
        .index
        .tolist()
    )
    segment_names = {
        ordered_clusters[0] : "Low Engagement",
        ordered_clusters[1] : "Regular Customers",
        ordered_clusters[2] : "High Value loyal",

    }

    df_clustered["Segment"] = df_clustered["Cluster"].map(segment_names)

else:
    df_clustered["Segment"] = (
        "Cluster" + df_clustered["Cluster"].astype(str)
    )

    ## client segment

with tab3:
    st.header("Step 3: Explore the Groups K-Means Discovered")
    st.subheader("How many Clients are in each group?")

    segment_counts = (
        df_clustered["Segment"]
        .value_counts()
        .rename_axis("Segment")
        .reset_index(name="Total_Clients")
    )

    st.bar_chart(
        segment_counts,
        x="Segment",
        y="Total_Clients",
        x_label="Business Groups",
        y_label="Total_Clients",
        color="#2E86AB"
    )

    st.subheader("Visualize the Clusters")

    st.scatter_chart(
        df_clustered,
        x="Monthly_Orders",
        y="Avg_Order_Value_PKR",
        color="Segment",
        size="Products_Purchased",
        x_label="Monthly Orders",
        y_label="Average Order Value (PKR)"
    )

    st.subheader("Cluster Summary")


    segment_summary = (
        df_clustered.groupby("Segment")
        .agg(
            Customers=("Customer_ID","count"),
            Avg_Monthly_Orders=("Monthly_Orders","mean"),
            Avg_Order_Value_PKR=("Avg_Order_Value_PKR","mean"),
            Avg_Products=("Products_Purchased","mean"),
            Avg_Payment_Delay = ("Payment_Delay_Days","mean"),
            Avg_Days_Since_Last_Order=("Days_Since_Last_Order","mean"),
            Avg_Monthly_Spend_PKR=("Monthly_Spend_PKR", "mean"),
        )
        .round(1)
        .reset_index()
    )

    st.dataframe(segment_summary, width="stretch", hide_index=True)

    if selected_k == 3:
        st.success(
            "Great! The model has discovered three useful customer"
            "behaviour groups. Remember: we need them only AFTER studying their behaviour"
        )

        c1, c2, c3 = st.columns(3)

        with c1:
            st.subheader("😜 High Value Loyal")
            st.write(
                "Usually place more orders, buy more products and spend more"
            )
        with c2:
            st.subheader(" 😊 Regular Customers")

        with c3:
            st.subheader("🤦🏻‍♂️ Low Engagement")

    st.subheader("Explore Individual Customers")

    segment_filter = st.selectbox(
        "Choose a Segment",
        options=sorted(df_clustered["Segment"].unique()),
    )

    filtered = df_clustered[
        df_clustered["Segment"] == segment_filter
    ]

    st.dataframe(filtered, width="stretch",hide_index=True)

## tab 4 Business Insights
with tab4:
    "💡 Business Insights"
    st.subheader("Business Insights & Recommendations")

    if selected_k == 3:
        st.markdown("""
        **1. Low (Cluster 0) - Low Engagement:**
        - Ye customer kam order karte hain, inka spend kam hai.
        - **Action:** Inko discount offer / reminder SMS bhejo.

        **2. Regular (Cluster 1) - Regular Buyer:**
        - Ye aapke normal customer hain.
        - **Action:** Inko loyalty points do taake ye Premium ban jayen.

        **3. Loyal (Cluster 2) / VIP - High Value:**
        - Ye sab se zyada kharch karte hain.
        - **Action:** Inko priority delivery aur special service do. Inko khona mat.

        **Overall Strategy for Getz Pharma:**
        - Karachi main sab se zyada customer hain, is liye wahan marketing zyada rakho.
        - At-Risk walon ko wapas lane ke liye call campaign karo.
        """)
    else:
        st.markdown("""
        **Premium:** High spend, loyal - Retain karo
        **Regular:** Average spend - Upsell karo
        **At-Risk:** Purane customer jo ab kam le rahe - Discount do
        **New:** Naye customer - Welcome offer do
        **VIP:** Sab se high-value - Special care do
        """)

    st.dataframe(cluster_summary)
    st.bar_chart(cluster_summary["Monthly_Spend_PKR"])