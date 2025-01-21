import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, dash_table
from transformers import pipeline
import requests

# Setup sentiment analysis
sentiment_pipeline = pipeline("sentiment-analysis", model="yiyanghkust/finbert-tone")

# Initialize Dash app
app = Dash(__name__)

# Dataset size options
DATASET_OPTIONS = [90, 180, 270, 365]

# Fetch stock data
def fetch_stock_data(symbol, days):
    today = datetime.now()
    start_date = (today - timedelta(days=days)).strftime('%Y-%m-%d')
    stock_data = yf.download(symbol, start=start_date, end=today.strftime('%Y-%m-%d'), auto_adjust=True)
    
    if stock_data.empty:
        raise ValueError(f"No data fetched for {symbol}")

    # Calculate Average Price
    stock_data['Average_Price'] = (stock_data['High'] + stock_data['Low']) / 2
    stock_data.dropna(inplace=True)
    stock_data.reset_index(inplace=True)

    # Normalize the 'Date' column
    stock_data['Date'] = pd.to_datetime(stock_data['Date']).dt.tz_localize(None)
    return stock_data

# Fetch news data
def fetch_news_data(company):
    url = f"https://newsapi.org/v2/everything?q={company}&apiKey=d555c82c13af4e2eb3e94d67349fdf03"
    response = requests.get(url)
    if response.status_code != 200:
        raise ValueError("Failed to fetch news")

    news_data = response.json()["articles"][:5]
    headlines = [article["title"] for article in news_data]

    # Sentiment Analysis
    sentiments = sentiment_pipeline(headlines)
    return headlines, sentiments

# Train models and compare results
def train_and_compare_models(data, forecast_days=10):
    X = data[['Open', 'High', 'Low', 'Close', 'Volume', 'Average_Price']]
    y = data['Average_Price'].shift(-forecast_days)
    
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "XGBoost": XGBRegressor(objective='reg:squarederror', n_estimators=100, max_depth=5, learning_rate=0.1)
    }
    
    results = []
    best_model_name, best_dataset_size, best_mse = None, None, float('inf')
    best_forecast = None
    
    for days in DATASET_OPTIONS:
        subset = data.iloc[-days:]
        X_subset = subset[['Open', 'High', 'Low', 'Close', 'Volume', 'Average_Price']]
        y_subset = subset['Average_Price'].shift(-forecast_days).dropna()

        if len(y_subset) < forecast_days:
            continue  # Skip this dataset size if there's not enough data

        X_train, X_test, y_train, y_test = train_test_split(X_subset.iloc[:-forecast_days], y_subset, test_size=0.2, random_state=42)
        
        for name, model in models.items():
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            mse = mean_squared_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)

            results.append({
                "Dataset Size": days,
                "Model": name,
                "MSE": round(mse, 4),
                "R² Score": round(r2, 4)
            })

            # Choose the best model and dataset size
            if mse < best_mse:
                best_mse = mse
                best_model_name = name
                best_dataset_size = days
                best_forecast = model.predict(X_subset.tail(forecast_days))

    return results, best_model_name, best_dataset_size, best_forecast

# Dashboard layout
app.layout = html.Div([
    html.H1("Stock Price Prediction Dashboard", style={"textAlign": "center"}),

    html.Div([
        dcc.Input(id='stock-input', type='text', placeholder='Enter stock symbol (e.g., AAPL)', style={"width": "50%", "margin": "0 auto"}),
        html.Button('Submit', id='submit-button', n_clicks=0, style={"marginTop": "10px"}),

        html.Label("Select Dataset Size:"),
        dcc.Dropdown(
            id='dataset-size',
            options=[{"label": f"{days} Days", "value": days} for days in DATASET_OPTIONS],
            value=180,  # Default selection
            style={"width": "50%", "margin": "0 auto"}
        )
    ], style={"textAlign": "center", "marginBottom": "20px"}),

    html.Div(id='news-section', style={"marginBottom": "30px"}),

    dcc.Graph(id='stock-graph', style={"height": "70vh"}),

    html.Div(id='recommendation-section', style={"marginTop": "20px", "textAlign": "center", "fontSize": "16px"}),

    html.H3("Model Comparison Results", style={"textAlign": "center"}),
    dash_table.DataTable(id='model-comparison-table', style_table={'margin': 'auto'})
])

@app.callback(
    [Output('stock-graph', 'figure'),
     Output('news-section', 'children'),
     Output('recommendation-section', 'children'),
     Output('model-comparison-table', 'data'),
     Output('model-comparison-table', 'columns')],
    [Input('submit-button', 'n_clicks')],
    [Input('stock-input', 'value'), Input('dataset-size', 'value')]
)
def update_dashboard(n_clicks, stock_symbol, dataset_size):
    if n_clicks == 0 or not stock_symbol:
        return {}, "", "", [], []

    try:
        stock_data = fetch_stock_data(stock_symbol, dataset_size)
        news_headlines, sentiments = fetch_news_data(stock_symbol)

        results, best_model_name, best_dataset_size, best_forecast = train_and_compare_models(stock_data)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=stock_data['Date'], y=stock_data['Average_Price'], mode='lines', name='Historical'))
        
        if best_forecast is not None:
            future_dates = [stock_data['Date'].iloc[-1] + timedelta(days=i) for i in range(1, 11)]
            fig.add_trace(go.Scatter(x=future_dates, y=best_forecast, mode='lines', name='Forecast', line=dict(color='red')))
        
        fig.update_layout(title=f"{stock_symbol.upper()} Stock Price Prediction", xaxis_title="Date", yaxis_title="Price")

        news_html = [html.P(f"{headline} | Sentiment: {sentiment['label']} | Confidence: {sentiment['score']:.2f}")
                     for headline, sentiment in zip(news_headlines, sentiments)]

        suggested_price = stock_data['Average_Price'].iloc[-1]
        recommendation_html = html.Div([
            html.H3("Recommendation"),
            html.P(f"Best Model: {best_model_name} using {best_dataset_size} days of data."),
            html.P(f"Suggested price to act: {suggested_price:.2f}")
        ])

        return fig, news_html, recommendation_html, results, [{"name": col, "id": col} for col in results[0].keys()]

    except Exception as e:
        return {}, [html.Div(f"Error: {str(e)}")], "", [], []

# Run app
if __name__ == '__main__':
    app.run_server(debug=True)
