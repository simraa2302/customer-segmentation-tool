import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# App title
st.title("🛍️ Customer Segmentation using KMeans Clustering")

# Initialize session state to store entered data
if 'df_manual' not in st.session_state:
    st.session_state.df_manual = pd.DataFrame(columns=["CustomerID", "Age", "Annual Income (k$)", "Spending Score (1-100)"])

# Choose input mode
mode = st.radio("Select Input Mode", ["Upload CSV", "Manual Entry"])

# Upload CSV Mode
if mode == "Upload CSV":
    uploaded_file = st.file_uploader("Upload your customer CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.subheader("📄 Uploaded Data")
        st.write(df)
    else:
        df = pd.DataFrame()

# Manual Entry Mode
elif mode == "Manual Entry":
    st.subheader("✍️ Enter Customer Details Using Sliders")

    customer_id = st.text_input("Customer ID")
    age = st.slider("Select Age", 15, 80, 30)
    income = st.slider("Select Annual Income (k$)", 10, 150, 60)
    score = st.slider("Select Spending Score (1-100)", 1, 100, 50)

    if st.button("Add Customer"):
        new_entry = pd.DataFrame([[customer_id, age, income, score]], 
                                 columns=["CustomerID", "Age", "Annual Income (k$)", "Spending Score (1-100)"])
        st.session_state.df_manual = pd.concat([st.session_state.df_manual, new_entry], ignore_index=True)

    df = st.session_state.df_manual

    if not df.empty:
        st.subheader("🧾 Entered Data")
        st.write(df)

# Show clustering only if data is available
if not df.empty:
    k = st.slider("Select number of clusters (k)", min_value=2, max_value=10, value=3)

    if st.button("Run Clustering"):
        try:
            X = df.iloc[:, 1:].values  # Exclude CustomerID

            # Apply KMeans
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            df['Cluster'] = kmeans.fit_predict(X)

            st.subheader("✅ Clustering Results")
            st.write(df)

            # Visualize clusters
            plt.figure(figsize=(8, 5))
            sns.scatterplot(x=df["Annual Income (k$)"], y=df["Spending Score (1-100)"], 
                            hue=df['Cluster'], palette="Set2", s=100)
            plt.xlabel("Annual Income (k$)")
            plt.ylabel("Spending Score (1-100)")
            plt.title("Customer Segments")
            st.pyplot(plt)

            # Cluster descriptions
            st.subheader("📌 Cluster Descriptions")
            for i in range(k):
                cluster_data = df[df['Cluster'] == i]
                st.markdown(f"**Cluster {i}:**")
                st.write(f"Number of Customers: {len(cluster_data)}")
                st.write(f"Average Age: {cluster_data['Age'].mean():.1f}")
                st.write(f"Average Income: {cluster_data['Annual Income (k$)'].mean():.1f} k$")
                st.write(f"Average Spending Score: {cluster_data['Spending Score (1-100)'].mean():.1f}")
                st.markdown("---")

        except Exception as e:
            st.error(f"❌ Error: {e}")
else:
    st.info("📌 Please upload a CSV file or enter customer data manually.")
