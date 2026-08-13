import streamlit as st
import pandas as pd
import joblib

# Page configuration
st.set_page_config(
    page_title="Restaurant Profit Optimization",
    page_icon="🍽️",
    layout="wide"
)

# Load data and model
df = pd.read_csv(
    "data/SkyCity Auckland Restaurants & Bars - Copy.csv"
)

model = joblib.load("restaurant_profit_model.pkl")
features = joblib.load("model_features.pkl")

# Title
st.title("🍽️ Restaurant Profit Optimization Dashboard")

st.write(
    "Predictive modeling and what-if analysis "
    "for multi-channel restaurant operations."
)

# Sidebar
st.sidebar.header("Scenario Inputs")

restaurant_name = st.sidebar.selectbox(
    "Select Restaurant",
    df["RestaurantName"].unique()
)

restaurant = df[
    df["RestaurantName"] == restaurant_name
].iloc[0]

commission = st.sidebar.slider(
    "Commission Rate",
    min_value=0.05,
    max_value=0.40,
    value=float(restaurant["CommissionRate"]),
    step=0.01
)

delivery_cost = st.sidebar.slider(
    "Delivery Cost Per Order",
    min_value=0.50,
    max_value=10.00,
    value=float(restaurant["DeliveryCostPerOrder"]),
    step=0.10
)

monthly_orders = st.sidebar.slider(
    "Monthly Orders",
    min_value=100,
    max_value=5000,
    value=int(restaurant["MonthlyOrders"]),
    step=50
)

# Prediction input
input_data = pd.DataFrame(
    [restaurant[features].values],
    columns=features
)

input_data["CommissionRate"] = commission
input_data["DeliveryCostPerOrder"] = delivery_cost
input_data["MonthlyOrders"] = monthly_orders

# Prediction
prediction = model.predict(input_data)[0]

# KPI section
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Predicted Monthly Profit",
        f"${prediction:,.2f}"
    )

with col2:
    st.metric(
        "Commission Rate",
        f"{commission * 100:.1f}%"
    )

with col3:
    st.metric(
        "Monthly Orders",
        f"{monthly_orders:,}"
    )

# Restaurant information
st.subheader("Restaurant Information")

info = pd.DataFrame({
    "Metric": [
        "Cuisine Type",
        "Segment",
        "Subregion",
        "Average Order Value"
    ],
    "Value": [
        restaurant["CuisineType"],
        restaurant["Segment"],
        restaurant["Subregion"],
        f"${restaurant['AOV']:.2f}"
    ]
})

st.table(info)

# What-if scenario
st.subheader("What-If Scenario")

st.write(
    "Adjust the sliders on the left to see how "
    "commission, delivery cost, and order volume "
    "affect predicted monthly profit."
)

# Machine learning model
st.subheader("Machine Learning Model")

st.write(
    "Model used: Random Forest Regression"
)

st.write(
    "The model predicts Total Monthly Net Profit "
    "using restaurant, cost, demand, and channel features."
)

# Channel Profit Analysis
st.subheader("📊 Channel Profit Analysis")

channel_profit = {
    "In-Store": restaurant["InStoreNetProfit"],
    "Uber Eats": restaurant["UberEatsNetProfit"],
    "DoorDash": restaurant["DoorDashNetProfit"],
    "Self Delivery": restaurant["SelfDeliveryNetProfit"]
}

channel_df = pd.DataFrame(
    channel_profit.items(),
    columns=["Channel", "Net Profit"]
)

st.bar_chart(
    channel_df.set_index("Channel")
)
# -----------------------------
# Cost Sensitivity Analysis
# -----------------------------

st.subheader("📈 Commission Cost Sensitivity")

sensitivity_results = []

for rate in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]:

    scenario = input_data.copy()

    scenario["CommissionRate"] = rate

    predicted_profit = model.predict(scenario)[0]

    sensitivity_results.append({
        "Commission Rate": rate * 100,
        "Predicted Profit": predicted_profit
    })

sensitivity_df = pd.DataFrame(sensitivity_results)

st.line_chart(
    sensitivity_df.set_index("Commission Rate")
)

st.write(
    "This chart shows how changes in commission rate "
    "can affect predicted monthly profit."
)
# -----------------------------
# Optimization Recommendation
# -----------------------------

st.subheader("🎯 Profit Optimization Recommendation")

optimization_results = []

for rate in [0.10, 0.12, 0.14, 0.16, 0.18,
              0.20, 0.22, 0.24, 0.26, 0.28,
              0.30, 0.32, 0.34, 0.36, 0.38, 0.40]:

    scenario = input_data.copy()

    scenario["CommissionRate"] = rate

    predicted_profit = model.predict(scenario)[0]

    optimization_results.append({
        "CommissionRate": rate,
        "PredictedProfit": predicted_profit
    })

optimization_df = pd.DataFrame(optimization_results)

best_result = optimization_df.loc[
    optimization_df["PredictedProfit"].idxmax()
]

best_rate = best_result["CommissionRate"]
best_profit = best_result["PredictedProfit"]

st.success(
    f"Recommended Commission Rate: {best_rate * 100:.1f}%"
)

st.metric(
    "Maximum Predicted Profit",
    f"${best_profit:,.2f}"
)

st.write(
    "The recommendation is based on the Random Forest "
    "model and the tested commission-rate scenarios."
)