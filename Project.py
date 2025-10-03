import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
from fastapi import FastAPI
import uvicorn
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import os

# --- Define the absolute file path for the dataset ---
# Using a raw string (r"...") to handle Windows backslashes
FILEPATH = r"C:\Users\alesha.b.lv\Desktop\MLOps\airflow_env\customer_shopping_data.csv"

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Retail Analytics & MLOps API",
    description="An API to serve customer segmentation and sales insights."
)

# --- Helper Functions for Analysis ---

def load_and_process_data(filepath=FILEPATH):
    """Loads and preprocesses the customer shopping data."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"The file was not found at the specified path: {filepath}")
    
    df = pd.read_csv(filepath)
    df['invoice_date'] = pd.to_datetime(df['invoice_date'], format='%d/%m/%Y')
    df['total_price'] = df['quantity'] * df['price']
    return df

def calculate_rfm(df):
    """Calculates RFM scores and segments customers."""
    snapshot_date = df['invoice_date'].max() + dt.timedelta(days=1)
    rfm = df.groupby('customer_id').agg({
        'invoice_date': lambda date: (snapshot_date - date.max()).days,
        'invoice_no': 'nunique',
        'total_price': 'sum'
    })
    rfm.rename(columns={'invoice_date': 'Recency', 'invoice_no': 'Frequency', 'total_price': 'Monetary'}, inplace=True)
    
    rfm['R_Score'] = pd.qcut(rfm['Recency'], 4, labels=[4, 3, 2, 1])
    rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 4, labels=[1, 2, 3, 4])
    rfm['M_Score'] = pd.qcut(rfm['Monetary'], 4, labels=[1, 2, 3, 4])
    
    seg_map = {
        r'[1-2][1-2]': 'Hibernating', r'[1-2][3-4]': 'At Risk',
        r'3[1-2]': 'About To Sleep', r'33': 'Need Attention',
        r'[3-4][3-4]': 'Loyal Customers', r'41': 'Promising',
        r'4[2-3]': 'Potential Loyalists', r'44': 'Champions'
    }
    rfm['Segment'] = (rfm['R_Score'].astype(str) + rfm['F_Score'].astype(str)).replace(seg_map, regex=True)
    return rfm

# --- Load data on startup ---
data = load_and_process_data()

# --- API Endpoints ---

@app.get("/")
def read_root():
    return {"message": "Welcome to the Retail Analytics API. Go to /docs for endpoints."}

@app.get("/rfm-segmentation")
def get_rfm_segmentation():
    """Returns the RFM segmentation for all customers."""
    rfm_data = calculate_rfm(data)
    return rfm_data.to_dict(orient='index')

@app.get("/store-performance")
def get_store_performance():
    """Returns total sales aggregated by shopping mall."""
    performance = data.groupby('shopping_mall')['total_price'].sum().sort_values(ascending=False)
    return performance.to_dict()

@app.get("/seasonal-trends")
def get_seasonal_trends():
    """Returns total sales aggregated by month."""
    trends = data.set_index('invoice_date').resample('M')['total_price'].sum()
    return trends.to_dict()

@app.get("/payment-methods")
def get_payment_methods():
    """Returns the count of transactions for each payment method."""
    methods = data['payment_method'].value_counts()
    return methods.to_dict()

@app.post("/trigger-training")
def trigger_model_training():
    """
    Triggers a dummy model training job and logs it to MLflow.
    This simulates retraining a model to predict total spending based on age.
    """
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("Retail Spending Predictor")

    with mlflow.start_run():
        model_data = data[['age', 'total_price']].dropna()
        X = model_data[['age']]
        y = model_data['total_price']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = LinearRegression()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        mse = mean_squared_error(y_test, preds)

        mlflow.log_param("model_type", "LinearRegression")
        mlflow.log_metric("mse", mse)
        mlflow.sklearn.log_model(model, "spending-predictor-model")
        
        return {"message": "Model training complete.", "run_id": mlflow.active_run().info.run_id, "mse": mse}

def generate_visualizations():
    """Generates and saves plots for the key analyses."""
    print("Generating and saving initial visualizations...")
    
    rfm = calculate_rfm(data)
    plt.figure(figsize=(10, 6)); sns.barplot(x=rfm['Segment'].value_counts().index, y=rfm['Segment'].value_counts().values)
    plt.title('Customer Segmentation by RFM Score'); plt.xlabel('Segment'); plt.ylabel('Number of Customers'); plt.xticks(rotation=45)
    plt.tight_layout(); plt.savefig('rfm_segmentation.png')

    performance = data.groupby('shopping_mall')['total_price'].sum().sort_values(ascending=False)
    plt.figure(figsize=(10, 6)); performance.plot(kind='bar'); plt.title('Total Sales by Shopping Mall')
    plt.xlabel('Shopping Mall'); plt.ylabel('Total Sales'); plt.xticks(rotation=45, ha='right')
    plt.tight_layout(); plt.savefig('store_performance.png')

    trends = data.set_index('invoice_date').resample('M')['total_price'].sum()
    plt.figure(figsize=(10, 6)); trends.plot(kind='line', marker='o'); plt.title('Monthly Sales Trends')
    plt.xlabel('Month'); plt.ylabel('Total Sales'); plt.grid(True)
    plt.tight_layout(); plt.savefig('monthly_sales.png')

    methods = data['payment_method'].value_counts()
    plt.figure(figsize=(8, 8)); methods.plot(kind='pie', autopct='%1.1f%%', startangle=90)
    plt.title('Payment Method Distribution'); plt.ylabel('')
    plt.tight_layout(); plt.savefig('payment_methods.png')
    
    plt.close('all')
    print("Visualizations saved as .png files.")

# --- Main Execution Block ---
if __name__ == "__main__":
    generate_visualizations()
    uvicorn.run(app, host="0.0.0.0", port=8000)