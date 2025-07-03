import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.cluster import KMeans

from mlxtend.frequent_patterns import apriori, association_rules

import io

st.set_page_config(page_title="Used Car Inventory Procurement Tool", layout="wide")

# ---- DATA LOADING ----
@st.cache_data
def load_data():
    df = pd.read_excel("cleaned_data.xlsx", sheet_name="cleaned data")
    return df

df = load_data()
all_columns = df.columns.tolist()

# ---- SIDEBAR NAVIGATION ----
st.sidebar.title("Modules")
page = st.sidebar.radio(
    "Select Module",
    ["Data Visualization", "Classification", "Clustering", "Association Rules", "Regression Insights"]
)

# ---- COMMON UTILS ----
def add_description(text):
    st.markdown(f"> **Insight:** {text}")

# ---- 1. DATA VISUALIZATION ----
if page == "Data Visualization":
    st.header("🔍 Data Visualization & Insights")
    st.write("Explore business trends and actionable insights with filters below.")
    # Example filters
    locations = st.multiselect("Select Locations", options=sorted(df['Location'].dropna().unique()), default=list(df['Location'].dropna().unique()))
    car_types = st.multiselect("Select Car Types", options=sorted(df['CarType'].dropna().unique()), default=list(df['CarType'].dropna().unique()))
    min_year, max_year = int(df['ManufacturedYear'].min()), int(df['ManufacturedYear'].max())
    year_range = st.slider("Manufactured Year Range", min_year, max_year, (min_year, max_year), 1)

    filtered = df[
        (df['Location'].isin(locations)) &
        (df['CarType'].isin(car_types)) &
        (df['ManufacturedYear'].between(year_range[0], year_range[1]))
    ]
    st.write(f"Total filtered records: {len(filtered)}")

    c1, c2 = st.columns(2)
    with c1:
        fig1 = px.bar(
            filtered.groupby("Location")["ProfitLoss"].sum().sort_values().reset_index(),
            x="Location", y="ProfitLoss", title="Total Profit/Loss by Location"
        )
        st.plotly_chart(fig1, use_container_width=True)
        add_description("This bar chart shows which cities generate the most/least profit overall for the dealership.")

    with c2:
        fig2 = px.bar(
            filtered.groupby("CarType")["ProfitMargin"].mean().sort_values().reset_index(),
            x="CarType", y="ProfitMargin", title="Avg. Profit Margin by Car Type"
        )
        st.plotly_chart(fig2, use_container_width=True)
        add_description("Dealerships can identify which car types yield highest average margins.")

    c3, c4 = st.columns(2)
    with c3:
        fig3 = px.bar(
            filtered.groupby("CarName")["SaleVolume"].sum().sort_values(ascending=False).head(10).reset_index(),
            x="CarName", y="SaleVolume", title="Top 10 Fastest Selling Models"
        )
        st.plotly_chart(fig3, use_container_width=True)
        add_description("Highlights the top-selling car models for strategic stocking decisions.")

    with c4:
        fig4 = px.scatter(
            filtered, x="Margin", y="DaystoSell", color="CarType",
            hover_data=["CarName"], title="Margin vs. Days to Sell"
        )
        st.plotly_chart(fig4, use_container_width=True)
        add_description("Shows if higher margin cars take longer to sell, helping balance inventory between quick and profitable stock.")

    # More Insights (add more as needed for 10 insights)
    # Example: Profit vs. Engine Power, Sale Price trends over years, Manufacturer profitability, etc.

    # Data Download
    csv = filtered.to_csv(index=False).encode('utf-8')
    st.download_button("Download Filtered Data", csv, "filtered_inventory.csv", "text/csv")

elif page == "Classification":
    st.header("🤖 Sales Success Classification")
    st.write("Train classifiers to predict car sale status based on features. Upload new data to predict outcomes.")

    # Target: Sold or Not (1 = Sold, 0 = Not Sold)
    df_cls = df.dropna(subset=["CarSaleStatus"])
    # Label encode
    y = df_cls["CarSaleStatus"].astype("category").cat.codes

    if len(np.unique(y)) < 2:
        st.error("Not enough data for classification! Both classes must be present in the data. Please check your data.")
    else:
        features = [
            "CarType", "Location", "ManufacturerName", "Color", "Gearbox",
            "ManufacturedYear", "MileageKM", "EnginePowerHP", "Price",
            "PurchasedPrice", "ProfitMargin", "SaleVolume"
        ]
        X = pd.get_dummies(df_cls[features], drop_first=True)

        # Do a stratified split so both train and test have both classes (if possible)
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
        except ValueError:
            st.error("Train/test split failed: one set ended up with only a single class. Try with more data or adjust your filters.")
            st.stop()

        # Now check classes again for train and test
        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            st.error("After splitting, one of the sets has only one class. Try with more data or check your filters.")
            st.stop()

        # --- scaling
        scaler = StandardScaler().fit(X_train)
        X_train_sc, X_test_sc = scaler.transform(X_train), scaler.transform(X_test)

        models = {
            "KNN": KNeighborsClassifier(n_neighbors=5),
            "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "Gradient Boosting": GradientBoostingClassifier(random_state=42)
        }

        results, probs = {}, {}
        for name, mdl in models.items():
            if name == "KNN":
                mdl.fit(X_train_sc, y_train)
                preds = mdl.predict(X_test_sc)
                proba = mdl.predict_proba(X_test_sc)
            else:
                mdl.fit(X_train, y_train)
                preds = mdl.predict(X_test)
                proba = mdl.predict_proba(X_test)

            # Pick probability of class 1 (sold), handle proba shape
            if proba.shape[1] == 2:
                probs[name] = proba[:, 1]
            else:
                probs[name] = None

            results[name] = [
                accuracy_score(y_test, preds),
                precision_score(y_test, preds, zero_division=0),
                recall_score(y_test, preds, zero_division=0),
                f1_score(y_test, preds, zero_division=0)
            ]

        res_df = pd.DataFrame(results, index=["Accuracy", "Precision", "Recall", "F1"]).T
        st.dataframe(res_df.style.background_gradient(axis=0, cmap="Greens"))

        sel_model = st.selectbox("Show Confusion Matrix for:", list(models.keys()))
        mdl = models[sel_model]
        y_pred = mdl.predict(X_test_sc if sel_model == "KNN" else X_test)
        cm = confusion_matrix(y_test, y_pred)
        fig_cm, ax_cm = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm)
        ax_cm.set_xlabel("Predicted"); ax_cm.set_ylabel("Actual")
        st.pyplot(fig_cm)
        st.markdown("> **Insight:** The confusion matrix shows the number of correct and incorrect predictions for car sales status.")

        st.subheader("ROC Curves")
        fig_roc, ax_roc = plt.subplots()
        for name, pr in probs.items():
            if pr is not None:
                fpr, tpr, _ = roc_curve(y_test, pr)
                ax_roc.plot(fpr, tpr, label=f"{name} (AUC={auc(fpr, tpr):.2f})")
        ax_roc.plot([0, 1], [0, 1], linestyle="--", color="gray")
        ax_roc.legend()
        st.pyplot(fig_roc)
        st.markdown("> **Insight:** ROC curve compares true vs. false positive rate for all classifiers.")

        st.markdown("---")
        st.subheader("Upload New Data to Predict Sale Status")
        new_file = st.file_uploader("Upload Excel/CSV (no 'CarSaleStatus' column required)", type=["csv", "xlsx"])
        if new_file:
            if new_file.name.endswith("csv"):
                new_data = pd.read_csv(new_file)
            else:
                new_data = pd.read_excel(new_file)
            new_X = pd.get_dummies(new_data[features], drop_first=True)
            new_X = new_X.reindex(columns=X.columns, fill_value=0)
            new_X_sc = scaler.transform(new_X)
            pred_label = models["Random Forest"].predict(new_X_sc)
            new_data["PredictedSaleStatus"] = pred_label
            st.dataframe(new_data)
            csv_out = new_data.to_csv(index=False).encode('utf-8')
            st.download_button("Download Results", csv_out, "predicted_sales.csv", "text/csv")

# ---- 3. CLUSTERING ----
elif page == "Clustering":
    st.header("🎯 Customer/Car Segmentation (K-Means Clustering)")
    cluster_features = ["CarType", "Location", "ManufacturerName", "ManufacturedYear",
                        "MileageKM", "EnginePowerHP", "Price", "ProfitMargin", "SaleVolume"]
    cluster_data = df[cluster_features]
    for col in cluster_data.select_dtypes(include=['object']):
        cluster_data[col] = pd.factorize(cluster_data[col])[0]
    st.write("Select number of clusters (k):")
    k = st.slider("Number of clusters", 2, 10, 4)
    kmeans = KMeans(n_clusters=k, random_state=42).fit(cluster_data)
    df['Cluster'] = kmeans.labels_
    # Elbow Chart
    costs = []
    for ki in range(2, 11):
        km = KMeans(n_clusters=ki, random_state=42).fit(cluster_data)
        costs.append(km.inertia_)
    fig_cost, ax_cost = plt.subplots()
    ax_cost.plot(range(2, 11), costs, marker="o")
    ax_cost.set_xlabel("k"); ax_cost.set_ylabel("Inertia"); ax_cost.set_title("Elbow Method")
    st.pyplot(fig_cost)
    add_description("The elbow chart helps you choose the optimal number of clusters based on inertia drop.")

    # Persona Table
    persona = df.groupby("Cluster")[["Price", "ProfitMargin", "SaleVolume", "ManufacturedYear"]].mean().round(1)
    st.dataframe(persona)
    add_description("Cluster persona table gives quick insight into the typical characteristics of each segment.")

    # Download Data
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Data with Clusters", csv, "clustered_data.csv", "text/csv")

# ---- 4. ASSOCIATION RULES ----
elif page == "Association Rules":
    st.header("🛒 Association Rule Mining (Apriori)")
    # Use two columns (example: CarType and Location)
    use_cols = st.multiselect("Choose columns for Apriori", ["CarType", "Location", "ManufacturerName", "Color"], default=["CarType", "Location"])
    min_sup = st.slider("Min Support", 0.01, 0.5, 0.05, 0.01)
    min_conf = st.slider("Min Confidence", 0.1, 0.9, 0.5, 0.05)
    min_lift = st.slider("Min Lift", 1.0, 3.0, 1.1, 0.1)
    if st.button("Run Apriori"):
        basket = pd.get_dummies(df[use_cols])
        freq = apriori(basket, min_support=min_sup, use_colnames=True)
        if freq.empty:
            st.warning("No frequent itemsets found.")
        else:
            rules = association_rules(freq, metric="confidence", min_threshold=min_conf)
            rules = rules[rules["lift"] >= min_lift]
            if rules.empty:
                st.warning("No rules at these thresholds.")
            else:
                show = rules.sort_values("lift", ascending=False).head(10)
                st.dataframe(show[["antecedents", "consequents", "support", "confidence", "lift"]])
                add_description("Top-10 rules help identify strong co-occurrence patterns (e.g., which car types and locations often occur together in sales).")

# ---- 5. REGRESSION ----
elif page == "Regression Insights":
    st.header("📈 Regression Insights")
    # Predict Profit Margin as an example
    reg_targets = ["ProfitMargin", "Margin", "SoldPrice", "SaleVolume"]
    reg_target = st.selectbox("Target variable", reg_targets, index=0)
    reg_features = ["ManufacturedYear", "MileageKM", "EnginePowerHP", "Price", "PurchasedPrice", "SaleVolume"]
    Xr = df[reg_features].dropna()
    yr = df[reg_target].loc[Xr.index]
    Xr_train, Xr_test, yr_train, yr_test = train_test_split(Xr, yr, test_size=0.2, random_state=42)
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(),
        "Lasso": Lasso(),
        "Decision Tree": DecisionTreeRegressor(max_depth=5, random_state=42)
    }
    out = []
    for name, mdl in models.items():
        mdl.fit(Xr_train, yr_train)
        preds = mdl.predict(Xr_test)
        out.append({
            "Model": name,
            "R2": round(mdl.score(Xr_test, yr_test), 3),
            "RMSE": round(np.sqrt(((yr_test - preds) ** 2).mean()), 2)
        })
    st.dataframe(pd.DataFrame(out).set_index("Model"))
    add_description("Models compared on how well they predict the target (e.g., ProfitMargin).")

    # Show scatter of predicted vs actual for best model
    best_model = max(out, key=lambda x: x["R2"])
    st.write(f"Best Model: {best_model['Model']}")
