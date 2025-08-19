from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.io as pio
from statsmodels.tsa.seasonal import STL
from scipy.stats import ttest_ind
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
import time
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
from sklearn.impute import SimpleImputer
from statsmodels.tsa.statespace.sarimax import SARIMAX
app = Flask(__name__)
def simulate_sensitivity(product_id, data):
    product_data = data[data['product_id'] == product_id]

    if product_data.empty:
        return "No data available for the selected product."

    # Simulate a sudden change in sales (e.g., a 20% increase)
    product_data['quantity'] = product_data['quantity'] * 1.2

    # Run the forecasting model again
    ensemble_forecast, ensemble_mae, ensemble_rmse, y_test_index, future_dates, future_ensemble_forecast = train_and_forecast_ensemble(product_id, product_data)

    # Return the sensitivity results
    return {
        'ensemble_mae': ensemble_mae,
        'ensemble_rmse': ensemble_rmse,
        'future_dates': future_dates,
        'future_ensemble_forecast': future_ensemble_forecast
    }

 

def train_and_forecast_ensemble(product_id, data):
    product_data = data[data['product_id'] == product_id]
    
    if product_data.empty:
        return None, None, None, None, None
    
    feature_columns = [
        'month_sin', 'month_cos', 'day_of_week_sin', 'day_of_week_cos',
        'is_weekend', 'is_holiday', 'economic_indicator', 'interaction_term'
    ] + [f'lag_{lag}' for lag in range(1, 15)] + ['rolling_mean_7', 'rolling_std_7']
    
    X = product_data[feature_columns]
    y = product_data['quantity']
    
    # Splitting the data
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Impute NaN values in the training data
    imputer = SimpleImputer(strategy='mean')
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)
    
    # Check for any remaining NaNs and handle them
    if np.any(np.isnan(X_train)) or np.any(np.isnan(X_test)):
        X_train = np.nan_to_num(X_train)
        X_test = np.nan_to_num(X_test)
    
    # Train XGBoost model
    xgb_model = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9)
    xgb_model.fit(X_train, y_train)
    xgb_forecast = xgb_model.predict(X_test)
    
    # Train Linear Regression model
    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)
    linear_forecast = linear_model.predict(X_test)
    
    # Ensemble forecast
    ensemble_forecast = (xgb_forecast + linear_forecast) / 2
    
    # Evaluate Model
    ensemble_mae = mean_absolute_error(y_test, ensemble_forecast)
    ensemble_rmse = mean_squared_error(y_test, ensemble_forecast) ** 0.5

    
    # Create future data for next 2 months
    last_date = product_data.index.max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=60, freq='D')
    future_data = pd.DataFrame(index=future_dates)
    future_data['month'] = future_data.index.month
    future_data['day_of_week'] = future_data.index.dayofweek
    future_data['is_weekend'] = (future_data['day_of_week'] >= 5).astype(int)
    holidays = calendar().holidays(start=last_date, end=future_dates[-1])
    future_data['is_holiday'] = future_data.index.isin(holidays).astype(int)
    future_data['month_sin'] = np.sin(2 * np.pi * future_data['month'] / 12)
    future_data['month_cos'] = np.cos(2 * np.pi * future_data['month'] / 12)
    future_data['day_of_week_sin'] = np.sin(2 * np.pi * future_data['day_of_week'] / 7)
    future_data['day_of_week_cos'] = np.cos(2 * np.pi * future_data['day_of_week'] / 7)
    
    # Lag features and rolling statistics for future data (using the latest available data)
    for lag in range(1, 15):
        future_data[f'lag_{lag}'] = product_data['quantity'].shift(lag).iloc[-1]
    future_data['rolling_mean_7'] = product_data['quantity'].rolling(window=7).mean().iloc[-1]
    future_data['rolling_std_7'] = product_data['quantity'].rolling(window=7).std().iloc[-1]
    
    # Interaction feature for future data
    future_data['economic_indicator'] = np.random.normal(loc=100, scale=10, size=len(future_data))
    future_data['interaction_term'] = future_data['rolling_mean_7'] * future_data['economic_indicator']
    
    future_data = future_data[feature_columns]
    
    # Impute NaN values in the future data
    future_data = imputer.transform(future_data)
    
    # Check for any remaining NaNs and handle them
    if np.any(np.isnan(future_data)):
        future_data = np.nan_to_num(future_data)
    
    # Predict future sales
    future_xgb_forecast = xgb_model.predict(future_data)
    future_linear_forecast = linear_model.predict(future_data)
    future_ensemble_forecast = (future_xgb_forecast + future_linear_forecast) / 2
    
    return ensemble_forecast, ensemble_mae, ensemble_rmse, y_test.index, future_dates, future_ensemble_forecast

def get_explanation(coefficient):
    """Get explanation based on the coefficient value."""
    if coefficient > 500:
        return "Very high positive coefficient, indicating a strong seasonal increase in sales."
    elif 100 < coefficient <= 500:
        return "High positive coefficient, indicating a noticeable seasonal increase in sales."
    elif 10 < coefficient <= 100:
        return "Moderate positive coefficient, indicating a moderate seasonal increase in sales."
    elif 0 < coefficient <= 10:
        return "Slight positive coefficient, indicating a slight seasonal increase in sales."
    elif -10 <= coefficient < 0:
        return "Slight negative coefficient, indicating a slight seasonal decrease in sales."
    elif -100 <= coefficient < -10:
        return "Moderate negative coefficient, indicating a moderate seasonal decrease in sales."
    elif -500 <= coefficient < -100:
        return "High negative coefficient, indicating a noticeable seasonal decrease in sales."
    else:  # coefficient < -500
        return "Very high negative coefficient, indicating a strong seasonal decrease in sales."
def get_promotion_explanation(coefficient):
    """Get explanation based on the promotion impact coefficient value."""
    if coefficient > 1.5:
        return "Very high positive impact, indicating a strong increase in sales during promotions."
    elif 1.2 < coefficient <= 1.5:
        return "High positive impact, indicating a noticeable increase in sales during promotions."
    elif 1.1 < coefficient <= 1.2:
        return "Moderate positive impact, indicating a moderate increase in sales during promotions."
    elif 1.0 < coefficient <= 1.1:
        return "Slight positive impact, indicating a slight increase in sales during promotions."
    elif 0.9 < coefficient <= 1.0:
        return "No significant impact, indicating similar sales during promotions and non-promotions."
    elif 0.8 < coefficient <= 0.9:
        return "Slight negative impact, indicating a slight decrease in sales during promotions."
    elif 0.5 < coefficient <= 0.8:
        return "Moderate negative impact, indicating a moderate decrease in sales during promotions."
    else:  # coefficient <= 0.5
        return "High negative impact, indicating a strong decrease in sales during promotions."
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/seasonality', methods=['GET', 'POST'])
def seasonality():
    # Load the CSV file
    sales_data = pd.read_csv(r'D:\bYoussefH-01\Desktop\app_stage\my_flask_app\Ventes-par-depot.csv')
    
    # Convert 'invoice_date' to datetime format
    sales_data['invoice_date'] = pd.to_datetime(sales_data['invoice_date'], errors='coerce')
    
    # Set 'invoice_date' as the index
    sales_data.set_index('invoice_date', inplace=True)
    
    # Aggregate sales data by date
    daily_sales = sales_data['quantity'].resample('D').sum()

    # Ensure the time series has a proper frequency
    if daily_sales.index.freq is None:
        daily_sales = daily_sales.asfreq(pd.infer_freq(daily_sales.index))

    # Perform STL decomposition (Seasonal and Trend decomposition using Loess)
    stl = STL(daily_sales, seasonal=13)
    result = stl.fit()

    # Extract the seasonal component
    sales_data['seasonal'] = result.seasonal.reindex(sales_data.index)

    # Calculate seasonal coefficients for each product and depot
    seasonal_coefficients = sales_data.groupby(['product_id', 'depot_id', sales_data.index.month])['seasonal'].mean().reset_index()
    seasonal_coefficients.rename(columns={'seasonal': 'seasonal_coefficient'}, inplace=True)

    # Get unique product and depot options
    products = seasonal_coefficients['product_id'].unique()
    depots = seasonal_coefficients['depot_id'].unique()

    selected_product = request.form.get('product')
    selected_depot = request.form.get('depot')
    coefficient = None
    explanation = None

    if selected_product and selected_depot:
        selected_product = int(selected_product)
        selected_depot = int(selected_depot)
        
        # Filter the coefficients for the selected product and depot
        filtered_coefficients = seasonal_coefficients[
            (seasonal_coefficients['product_id'] == selected_product) & 
            (seasonal_coefficients['depot_id'] == selected_depot)
        ]
        
        if not filtered_coefficients.empty:
            # Get the highest seasonal coefficient
            best_month_data = filtered_coefficients.loc[filtered_coefficients['seasonal_coefficient'].idxmax()]
            coefficient = best_month_data['seasonal_coefficient']
            explanation = get_explanation(coefficient)
        else:
            explanation = "No data available for the selected product and depot combination."

    # Plot the decomposed components using Plotly
    original_fig = go.Figure(go.Scatter(x=daily_sales.index, y=daily_sales, mode='lines', name='Original'))
    original_fig.update_layout(title='Original Sales Data', xaxis_title='Date', yaxis_title='Quantity')

    trend_fig = go.Figure(go.Scatter(x=result.trend.index, y=result.trend, mode='lines', name='Trend'))
    trend_fig.update_layout(title='Trend', xaxis_title='Date', yaxis_title='Quantity')

    seasonal_fig = go.Figure(go.Scatter(x=result.seasonal.index, y=result.seasonal, mode='lines', name='Seasonality'))
    seasonal_fig.update_layout(title='Seasonality', xaxis_title='Date', yaxis_title='Quantity')

    residual_fig = go.Figure(go.Scatter(x=result.resid.index, y=result.resid, mode='lines', name='Residuals'))
    residual_fig.update_layout(title='Residuals', xaxis_title='Date', yaxis_title='Quantity')

    # Convert plots to HTML
    original_plot = pio.to_html(original_fig, full_html=False)
    trend_plot = pio.to_html(trend_fig, full_html=False)
    seasonal_plot = pio.to_html(seasonal_fig, full_html=False)
    residual_plot = pio.to_html(residual_fig, full_html=False)

    return render_template('seasonality.html', products=products, depots=depots, coefficient=coefficient, explanation=explanation,
                           original_plot=original_plot, trend_plot=trend_plot, seasonal_plot=seasonal_plot, residual_plot=residual_plot)
@app.route('/promotions', methods=['GET', 'POST'])
def promotions():
    # Load the CSV file
    sales_data = pd.read_csv(r'D:\bYoussefH-01\Desktop\app_stage\my_flask_app\Ventes-par-depot.csv')

    # Convert 'invoice_date' to datetime format
    sales_data['invoice_date'] = pd.to_datetime(sales_data['invoice_date'], errors='coerce')

    # Ensure the 'status' column exists and contains valid values
    if 'status' not in sales_data.columns:
        return "Error: The 'status' column is missing from the dataset."

    # Map status values to meaningful categories
    status_mapping = {0: 'Non-Promotion', 1: 'Promotion', 2: 'Free Product', 3: 'Pack Sale'}
    sales_data['status'] = sales_data['status'].map(status_mapping)

    # Ensure 'Promotion' and 'Non-Promotion' are valid categories
    if not set(['Promotion', 'Non-Promotion']).issubset(set(sales_data['status'].unique())):
        return "Error: The 'status' column does not contain the required 'Promotion' and 'Non-Promotion' values."

    # Group sales data by promotion status and calculate mean sales quantity
    promotion_analysis = sales_data.groupby('status')['quantity'].mean().reset_index()
    promotion_analysis.columns = ['Promotion Status', 'Average Quantity Sold']

    # Perform t-test to compare sales during promotion and non-promotion periods
    promo_sales = sales_data[sales_data['status'] == 'Promotion']['quantity']
    non_promo_sales = sales_data[sales_data['status'] == 'Non-Promotion']['quantity']
    t_stat, p_value = ttest_ind(promo_sales, non_promo_sales, equal_var=False)

    # Calculate the promotion impact coefficient for each product and depot
    promotion_coefficients = sales_data.groupby(['product_id', 'depot_id', 'status'])['quantity'].mean().unstack().reset_index()
    promotion_coefficients['promotion_impact_coefficient'] = promotion_coefficients['Promotion'] / promotion_coefficients['Non-Promotion']

    # Get unique product and depot options
    products = promotion_coefficients['product_id'].unique()
    depots = promotion_coefficients['depot_id'].unique()

    selected_product = request.form.get('product')
    selected_depot = request.form.get('depot')
    coefficient = None
    explanation = None
    non_promo_quantity = None
    promo_quantity = None
    free_product_quantity = None
    pack_sale_quantity = None

    if selected_product and selected_depot:
        selected_product = int(selected_product)
        selected_depot = int(selected_depot)
        
        # Filter the coefficients for the selected product and depot
        filtered_coefficients = promotion_coefficients[
            (promotion_coefficients['product_id'] == selected_product) & 
            (promotion_coefficients['depot_id'] == selected_depot)
        ]
        
        if not filtered_coefficients.empty:
            coefficient = filtered_coefficients['promotion_impact_coefficient'].values[0]
            explanation = get_promotion_explanation(coefficient)
            non_promo_quantity = filtered_coefficients['Non-Promotion'].values[0]
            promo_quantity = filtered_coefficients['Promotion'].values[0]
            free_product_quantity = filtered_coefficients['Free Product'].values[0]
            pack_sale_quantity = filtered_coefficients['Pack Sale'].values[0]
        else:
            coefficient = "No data available for the selected product and depot combination."

    # Plot the promotion analysis using Plotly
    promotion_fig = go.Figure(go.Bar(x=promotion_analysis['Promotion Status'], y=promotion_analysis['Average Quantity Sold']))
    promotion_fig.update_layout(title='Average Quantity Sold by Promotion Status', xaxis_title='Promotion Status', yaxis_title='Average Quantity Sold')

    # Convert plot to HTML
    promotion_plot = pio.to_html(promotion_fig, full_html=False)

    return render_template('promotions.html', promotion_plot=promotion_plot, t_stat=t_stat, p_value=p_value, products=products, depots=depots, coefficient=coefficient, explanation=explanation, non_promo_quantity=non_promo_quantity, promo_quantity=promo_quantity, free_product_quantity=free_product_quantity, pack_sale_quantity=pack_sale_quantity)
@app.route('/combined', methods=['GET', 'POST'])
def combined():
    # Load the CSV file
    sales_data = pd.read_csv(r'D:\bYoussefH-01\Desktop\app_stage\my_flask_app\Ventes-par-depot.csv')

    # Convert 'invoice_date' to datetime format
    sales_data['invoice_date'] = pd.to_datetime(sales_data['invoice_date'], errors='coerce')

    # Ensure the 'status' column exists and contains valid values
    if 'status' not in sales_data.columns:
        return "Error: The 'status' column is missing from the dataset."

    # Map status values to meaningful categories
    status_mapping = {0: 'Non-Promotion', 1: 'Promotion', 2: 'Free Product', 3: 'Pack Sale'}
    sales_data['status'] = sales_data['status'].map(status_mapping)

    # Extract month and year for seasonality analysis
    sales_data['month'] = sales_data['invoice_date'].dt.to_period('M')

    # Group sales data by month, year, and promotion status
    combined_analysis = sales_data.groupby(['month', 'status'])['quantity'].sum().reset_index()

    # Plot the combined analysis using Plotly
    combined_fig = go.Figure()

    for status in sales_data['status'].unique():
        filtered_data = combined_analysis[combined_analysis['status'] == status]
        combined_fig.add_trace(go.Scatter(x=filtered_data['month'].astype(str), y=filtered_data['quantity'], mode='lines+markers', name=status))

    combined_fig.update_layout(title='Combined Impact of Seasonality and Promotions on Sales', xaxis_title='Month', yaxis_title='Quantity Sold')

    # Convert plot to HTML
    combined_plot = pio.to_html(combined_fig, full_html=False)

    # Get unique product options
    products = sales_data['product_id'].unique()
    
    selected_product = request.form.get('product')
    product_status_plots = []
    status_messages = []

    if selected_product:
        selected_product = int(selected_product)
        product_data = sales_data[sales_data['product_id'] == selected_product]

        for status in sales_data['status'].unique():
            status_data = product_data[product_data['status'] == status]
            if not status_data.empty:
                status_fig = go.Figure()
                filtered_data = status_data.groupby('month')['quantity'].sum().reset_index()
                status_fig.add_trace(go.Scatter(x=filtered_data['month'].astype(str), y=filtered_data['quantity'], mode='lines+markers', name=status))
                status_fig.update_layout(title=f'Sales Trends for Product {selected_product} - {status}', xaxis_title='Month', yaxis_title='Quantity Sold')
                product_status_plots.append(pio.to_html(status_fig, full_html=False))
                status_messages.append('')
            else:
                status_messages.append(f'No sales in {status} for Product {selected_product}')
                product_status_plots.append(None)

    return render_template('combined.html', combined_plot=combined_plot, products=products, selected_product=selected_product, product_status_plots=product_status_plots, status_messages=status_messages, zip=zip)
@app.route('/forecasting', methods=['GET', 'POST'])
def forecasting():
    # Load the CSV file
    sales_data = pd.read_csv(r'D:\bYoussefH-01\Desktop\app_stage\my_flask_app\Ventes-par-depot.csv')
    sales_data['invoice_date'] = pd.to_datetime(sales_data['invoice_date'], errors='coerce')
    sales_data.set_index('invoice_date', inplace=True)

    # Add date-based features
    sales_data['month'] = sales_data.index.month
    sales_data['day_of_week'] = sales_data.index.dayofweek
    sales_data['is_weekend'] = (sales_data['day_of_week'] >= 5).astype(int)
    holidays = calendar().holidays(start=sales_data.index.min(), end=sales_data.index.max())
    sales_data['is_holiday'] = sales_data.index.isin(holidays).astype(int)

    # Cyclical encoding
    sales_data['month_sin'] = np.sin(2 * np.pi * sales_data['month'] / 12)
    sales_data['month_cos'] = np.cos(2 * np.pi * sales_data['month'] / 12)
    sales_data['day_of_week_sin'] = np.sin(2 * np.pi * sales_data['day_of_week'] / 7)
    sales_data['day_of_week_cos'] = np.cos(2 * np.pi * sales_data['day_of_week'] / 7)

    # Lag features and rolling stats
    for lag in range(1, 15):
        sales_data[f'lag_{lag}'] = sales_data['quantity'].shift(lag)
    sales_data['rolling_mean_7'] = sales_data['quantity'].rolling(window=7).mean()
    sales_data['rolling_std_7'] = sales_data['quantity'].rolling(window=7).std()

    # Synthetic external feature
    np.random.seed(42)
    sales_data['economic_indicator'] = np.random.normal(loc=100, scale=10, size=len(sales_data))

    # Interaction term
    sales_data['interaction_term'] = sales_data['rolling_mean_7'] * sales_data['economic_indicator']

    # Drop rows with NaNs (due to lags/rolling)
    sales_data.dropna(inplace=True)

    # Available products
    products = sales_data['product_id'].unique()
    
    # Get selected product from form
    selected_product = request.form.get('product')
    selected_product_str = str(selected_product) if selected_product else ""
    
    # Initialize variables
    ensemble_forecast = ensemble_mae = ensemble_rmse = y_test_index = future_dates = future_ensemble_forecast = None

    # Forecast if product is selected
    if selected_product:
        selected_product_int = int(selected_product)
        ensemble_forecast, ensemble_mae, ensemble_rmse, y_test_index, future_dates, future_ensemble_forecast = train_and_forecast_ensemble(selected_product_int, sales_data)

    # Plot actual sales
    actual_sales_fig = go.Figure()
    if selected_product:
        product_sales = sales_data[sales_data['product_id'] == int(selected_product)]
        monthly_sales = product_sales['quantity'].resample('M').sum()
        actual_sales_fig.add_trace(go.Scatter(x=monthly_sales.index, y=monthly_sales.values, mode='lines', name='Actual Sales'))
        actual_sales_fig.update_layout(title=f'Actual Sales for Product ID: {selected_product}', xaxis_title='Date', yaxis_title='Sales Quantity')
    actual_sales_plot = pio.to_html(actual_sales_fig, full_html=False)

    # Plot forecast
    future_forecast_fig = go.Figure()
    if future_ensemble_forecast is not None:
        future_forecast_fig.add_trace(go.Scatter(x=future_dates, y=future_ensemble_forecast, mode='lines', name='Future Ensemble Forecast'))
        future_forecast_fig.update_layout(title=f'Future Forecast for Product ID: {selected_product}', xaxis_title='Date', yaxis_title='Sales Quantity')
    future_forecast_plot = pio.to_html(future_forecast_fig, full_html=False)

    return render_template(
        'forecasting.html',
        products=products,
        selected_product=selected_product_str,  # Safe for comparison in template
        actual_sales_plot=actual_sales_plot,
        future_forecast_plot=future_forecast_plot,
        ensemble_mae=ensemble_mae,
        ensemble_rmse=ensemble_rmse
    )


@app.route('/sensitivity', methods=['GET', 'POST'])
def sensitivity():
    # Load the CSV file
    sales_data = pd.read_csv(r'D:\bYoussefH-01\Desktop\app_stage\my_flask_app\Ventes-par-depot.csv')
    sales_data['invoice_date'] = pd.to_datetime(sales_data['invoice_date'], errors='coerce')
    sales_data.set_index('invoice_date', inplace=True)

    # Adding more features
    sales_data['month'] = sales_data.index.month
    sales_data['day_of_week'] = sales_data.index.dayofweek
    sales_data['is_weekend'] = (sales_data['day_of_week'] >= 5).astype(int)
    holidays = calendar().holidays(start=sales_data.index.min(), end=sales_data.index.max())
    sales_data['is_holiday'] = sales_data.index.isin(holidays).astype(int)

    # Cyclical encoding for month and day_of_week
    sales_data['month_sin'] = np.sin(2 * np.pi * sales_data['month'] / 12)
    sales_data['month_cos'] = np.cos(2 * np.pi * sales_data['month'] / 12)
    sales_data['day_of_week_sin'] = np.sin(2 * np.pi * sales_data['day_of_week'] / 7)
    sales_data['day_of_week_cos'] = np.cos(2 * np.pi * sales_data['day_of_week'] / 7)

    # Adding lag features and rolling statistics
    for lag in range(1, 15):
        sales_data[f'lag_{lag}'] = sales_data['quantity'].shift(lag)
    sales_data['rolling_mean_7'] = sales_data['quantity'].rolling(window=7).mean()
    sales_data['rolling_std_7'] = sales_data['quantity'].rolling(window=7).std()

    # Adding a synthetic external feature (e.g., random economic indicator)
    np.random.seed(42)
    sales_data['economic_indicator'] = np.random.normal(loc=100, scale=10, size=len(sales_data))

    # Interaction features
    sales_data['interaction_term'] = sales_data['rolling_mean_7'] * sales_data['economic_indicator']

    sales_data.dropna(inplace=True)
    
    products = sales_data['product_id'].unique()
    selected_product = request.form.get('product')
    scenario = request.form.get('scenario')
    ensemble_forecast, ensemble_mae, ensemble_rmse, y_test_index, future_dates, future_ensemble_forecast, adjusted_future_ensemble_forecast = [None]*7

    if selected_product and scenario:
        selected_product = int(selected_product)
        ensemble_forecast, ensemble_mae, ensemble_rmse, y_test_index, future_dates, future_ensemble_forecast = train_and_forecast_ensemble(selected_product, sales_data)
        
        # Apply scenario-based disturbance
        adjusted_future_ensemble_forecast = future_ensemble_forecast.copy()
        if scenario == 'pandemic':
            adjusted_future_ensemble_forecast[:30] *= np.random.uniform(0.5, 0.7, size=30)  # Decrease
            adjusted_future_ensemble_forecast[30:] *= np.random.uniform(0.8, 1.0, size=30)
        elif scenario == 'economic_recession':
            adjusted_future_ensemble_forecast *= np.random.uniform(0.6, 0.9, size=len(adjusted_future_ensemble_forecast))
        elif scenario == 'economic_boom':
            adjusted_future_ensemble_forecast *= np.random.uniform(1.1, 1.5, size=len(adjusted_future_ensemble_forecast))
        elif scenario == 'new_product_launch':
            adjusted_future_ensemble_forecast[:15] *= np.random.uniform(0.8, 1.1, size=15)
            adjusted_future_ensemble_forecast[15:] *= np.random.uniform(1.2, 1.5, size=45)
        elif scenario == 'seasonal_peak':
            adjusted_future_ensemble_forecast *= np.random.uniform(1.2, 1.5, size=len(adjusted_future_ensemble_forecast))
        elif scenario == 'promotion':
            adjusted_future_ensemble_forecast *= np.random.uniform(1.3, 1.6, size=len(adjusted_future_ensemble_forecast))
        elif scenario == 'supply_chain_disruption':
            adjusted_future_ensemble_forecast *= np.random.uniform(0.4, 0.8, size=len(adjusted_future_ensemble_forecast))
        elif scenario == 'regulatory_change':
            adjusted_future_ensemble_forecast *= np.random.uniform(0.7, 1.1, size=len(adjusted_future_ensemble_forecast))
        elif scenario == 'competitive_pressure':
            adjusted_future_ensemble_forecast *= np.random.uniform(0.5, 0.9, size=len(adjusted_future_ensemble_forecast))

    return render_template('sensitivity.html', products=products, selected_product=selected_product, scenario=scenario,
                           sensitivity_results={
                               'ensemble_mae': ensemble_mae,
                               'ensemble_rmse': ensemble_rmse,
                               'future_dates': future_dates.tolist() if future_dates is not None else None,
                               'future_ensemble_forecast': future_ensemble_forecast.tolist() if future_ensemble_forecast is not None else None,
                               'adjusted_future_ensemble_forecast': adjusted_future_ensemble_forecast.tolist() if adjusted_future_ensemble_forecast is not None else None
                           })


if __name__ == '__main__':
    app.run(debug=True)
