import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
from prophet import Prophet
from datetime import datetime

# Define models
models = [
    LogisticRegression(),
    SVC(kernel='poly', probability=True),
    XGBClassifier(),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    GradientBoostingClassifier(),
    AdaBoostClassifier(),
    KNeighborsClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis(),
    LinearDiscriminantAnalysis(),
    ExtraTreesClassifier()
]

# Helper function for forecasting with Prophet
def forecast_prophet(df, periods=365*10):
    # Ensure correct column names
    df = df.rename(columns={'Date': 'ds', 'Close': 'y'})
    
    # Check the dataframe structure
    print(df.head())
    print(df.columns)
    
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    
    return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

# Tesla Data Analysis
tesla_df = pd.read_csv('/home/kali/Python Projects/tesla.csv')
print("Tesla Data Exploration:")
print(tesla_df.head())
print(tesla_df.shape)
print(tesla_df.describe())
print(tesla_df.info())

# Plot Tesla close price
plt.figure(figsize=(15, 5))
plt.plot(tesla_df['close'])
plt.title('Tesla Close Price', fontsize=25)
plt.ylabel('Price in Dollars')
plt.show()

# Handle missing values in Tesla data
print(tesla_df.isnull().sum())

# Convert 'date' column to datetime in Tesla data
tesla_df['date'] = pd.to_datetime(tesla_df['date'], format='%m/%d/%Y')
tesla_df['month'] = tesla_df['date'].dt.month
tesla_df['day'] = tesla_df['date'].dt.day
tesla_df['year'] = tesla_df['date'].dt.year

# Tesla Feature Engineering
tesla_df['is_quarter_end'] = np.where(tesla_df['month'] % 3 == 0, 1, 0)
tesla_df['open-close'] = tesla_df['open'] - tesla_df['close']
tesla_df['low-high'] = tesla_df['low'] - tesla_df['high']
tesla_df['target'] = np.where(tesla_df['close'].shift(-1) > tesla_df['close'], 1, 0)

# Tesla Distribution plots
features = ['open', 'high', 'low', 'close', 'volume']
plt.subplots(figsize=(20, 10))
for i, col in enumerate(features):
    plt.subplot(2, 3, i + 1)
    sb.histplot(tesla_df[col], kde=True)
plt.show()

# Tesla Box plots
plt.subplots(figsize=(20, 10))
for i, col in enumerate(features):
    plt.subplot(2, 3, i + 1)
    sb.boxplot(x=tesla_df[col])
plt.show()

# Tesla Grouped data for year-based visualization
data_grouped = tesla_df.groupby('year').mean()
plt.subplots(figsize=(20, 10))
for i, col in enumerate(['open', 'high', 'low', 'close']):
    plt.subplot(2, 2, i + 1)
    data_grouped[col].plot.bar()
    plt.title(col)
plt.show()

# Tesla Pie chart for target distribution
plt.pie(tesla_df['target'].value_counts().values, labels=[0, 1], autopct='%1.1f%%')
plt.title('Tesla Target Distribution')
plt.show()

# Tesla Heatmap of correlations
plt.figure(figsize=(10, 10))
sb.heatmap(tesla_df.corr() > 0.9, annot=True, cbar=False)
plt.title('Tesla Correlation Heatmap')
plt.show()

# Prepare Tesla features and target
tesla_features = tesla_df[['open-close', 'low-high', 'is_quarter_end']]
tesla_target = tesla_df['target']

# Standardize Tesla features
scaler = StandardScaler()
tesla_features = scaler.fit_transform(tesla_features)

# Split Tesla data into training and validation sets
X_train_tesla, X_valid_tesla, Y_train_tesla, Y_valid_tesla = train_test_split(tesla_features, tesla_target, test_size=0.1, random_state=2022)
print(X_train_tesla.shape, X_valid_tesla.shape)

# Train and evaluate models for Tesla
for model in models:
    model.fit(X_train_tesla, Y_train_tesla)
    print(f'Tesla - {model.__class__.__name__} : ')
    print('Training ROC AUC : ', metrics.roc_auc_score(Y_train_tesla, model.predict_proba(X_train_tesla)[:, 1]))
    print('Validation ROC AUC : ', metrics.roc_auc_score(Y_valid_tesla, model.predict_proba(X_valid_tesla)[:, 1]))
    print()

# Bitcoin Data Analysis
bitcoin_df = pd.read_csv('/home/kali/Python Projects/bitcoin.csv')
print("Bitcoin Data Exploration:")
print(bitcoin_df.head())
print(bitcoin_df.shape)
print(bitcoin_df.describe())
print(bitcoin_df.info())

# Plot Bitcoin output volume
plt.figure(figsize=(15, 5))
plt.plot(bitcoin_df['btc_output_volume'])
plt.title('Bitcoin Output Volume', fontsize=25)
plt.ylabel('Volume')
plt.show()

# Handle missing values in Bitcoin data
print(bitcoin_df.isnull().sum())

# Convert 'Date' column to datetime in Bitcoin data
bitcoin_df['Date'] = pd.to_datetime(bitcoin_df['Date'], format='%m/%d/%Y')
bitcoin_df['month'] = bitcoin_df['Date'].dt.month
bitcoin_df['day'] = bitcoin_df['Date'].dt.day
bitcoin_df['year'] = bitcoin_df['Date'].dt.year

# Bitcoin Feature Engineering
bitcoin_df['is_quarter_end'] = np.where(bitcoin_df['month'] % 3 == 0, 1, 0)
bitcoin_df['btc_market_price-btc_output_volume'] = bitcoin_df['btc_market_price'] - bitcoin_df['btc_output_volume']
bitcoin_df['btc_n_transactions_excluding_chains_longer_than_100-btc_avg_block_size'] = bitcoin_df['btc_n_transactions_excluding_chains_longer_than_100'] - bitcoin_df['btc_avg_block_size']
bitcoin_df['target'] = np.where(bitcoin_df['btc_output_volume'].shift(-1) > bitcoin_df['btc_output_volume'], 1, 0)

# Bitcoin Distribution plots
features = ['btc_market_price', 'btc_avg_block_size', 'btc_n_transactions_excluding_chains_longer_than_100', 'btc_output_volume']
plt.subplots(figsize=(20, 10))
for i, col in enumerate(features):
    plt.subplot(2, 3, i + 1)
    sb.histplot(bitcoin_df[col], kde=True)
plt.show()

# Bitcoin Box plots
plt.subplots(figsize=(20, 10))
for i, col in enumerate(features):
    plt.subplot(2, 3, i + 1)
    sb.boxplot(x=bitcoin_df[col])
plt.show()

# Bitcoin Grouped data for year-based visualization
data_grouped = bitcoin_df.groupby('year').mean()
plt.subplots(figsize=(20, 10))
for i, col in enumerate(['btc_market_price', 'btc_avg_block_size', 'btc_n_transactions_excluding_chains_longer_than_100', 'btc_output_volume']):
    plt.subplot(2, 2, i + 1)
    data_grouped[col].plot.bar()
    plt.title(col)
plt.show()

# Bitcoin Pie chart for target distribution
plt.pie(bitcoin_df['target'].value_counts().values, labels=[0, 1], autopct='%1.1f%%')
plt.title('Bitcoin Target Distribution')
plt.show()

# Bitcoin Heatmap of correlations
plt.figure(figsize=(10, 10))
sb.heatmap(bitcoin_df.corr() > 0.9, annot=True, cbar=False)
plt.title('Bitcoin Correlation Heatmap')
plt.show()

# Prepare Bitcoin features and target
bitcoin_features = bitcoin_df[['btc_market_price-btc_output_volume', 'btc_n_transactions_excluding_chains_longer_than_100-btc_avg_block_size', 'is_quarter_end']]
bitcoin_target = bitcoin_df['target']

# Standardize Bitcoin features
scaler = StandardScaler()
bitcoin_features = scaler.fit_transform(bitcoin_features)

# Split Bitcoin data into training and validation sets
X_train_bitcoin, X_valid_bitcoin, Y_train_bitcoin, Y_valid_bitcoin = train_test_split(bitcoin_features, bitcoin_target, test_size=0.1, random_state=2022)
print(X_train_bitcoin.shape, X_valid_bitcoin.shape)

# Train and evaluate models for Bitcoin
for model in models:
    model.fit(X_train_bitcoin, Y_train_bitcoin)
    print(f'Bitcoin - {model.__class__.__name__} : ')
    print('Training ROC AUC : ', metrics.roc_auc_score(Y_train_bitcoin, model.predict_proba(X_train_bitcoin)[:, 1]))
    print('Validation ROC AUC : ', metrics.roc_auc_score(Y_valid_bitcoin, model.predict_proba(X_valid_bitcoin)[:, 1]))
    print()

# Confusion matrices
for model in models:
    # Tesla
    cm_tesla = confusion_matrix(Y_valid_tesla, model.predict(X_valid_tesla))
    plt.figure(figsize=(10, 7))
    sb.heatmap(cm_tesla, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(Y_valid_tesla), yticklabels=np.unique(Y_valid_tesla))
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title(f'Tesla Confusion Matrix for {model.__class__.__name__}')
    plt.show()
    
    # Bitcoin
    cm_bitcoin = confusion_matrix(Y_valid_bitcoin, model.predict(X_valid_bitcoin))
    plt.figure(figsize=(10, 7))
    sb.heatmap(cm_bitcoin, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(Y_valid_bitcoin), yticklabels=np.unique(Y_valid_bitcoin))
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title(f'Bitcoin Confusion Matrix for {model.__class__.__name__}')
    plt.show()

# Microsoft Data Analysis and Forecasting
microsoft = pd.read_csv('/home/kali/Python Projects/microsoft.csv')
print("Microsoft Data Exploration:")
print(microsoft.head())
print(microsoft.shape)
print(microsoft.describe())
print(microsoft.info())

# Convert 'Date' column to datetime format
microsoft['Date'] = pd.to_datetime(microsoft['Date'], format='%d-%b-%y', errors='coerce')
microsoft = microsoft.dropna(subset=['Date'])

# Plot the Microsoft data
plt.figure(figsize=(14, 7))
plt.plot(microsoft['Date'], microsoft['Open'], color="blue", label="Open")
plt.plot(microsoft['Date'], microsoft['Close'], color="green", label="Close")
plt.plot(microsoft['Date'], microsoft['Volume'], color="red", label="Volume")
plt.title("Microsoft Stock Data")
plt.legend()
plt.xlabel('Date')
plt.ylabel('Value')
plt.show()

# Correlation heatmap for Microsoft data
sb.heatmap(microsoft.corr(), annot=True, cbar=False)
plt.show()

# Forecast Microsoft stock prices for the next 10 years
start_date = datetime(2025, 1, 1)
end_date = datetime(2035, 12, 31)
periods = (end_date - start_date).days

# Prepare Microsoft data for Prophet
microsoft_prophet_df = microsoft[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
print("Microsoft Prophet DataFrame:")
print(microsoft_prophet_df.head())
print(microsoft_prophet_df.columns)

# Forecasting with Prophet
forecast_microsoft = forecast_prophet(microsoft_prophet_df, periods=periods)

# Plot the forecast results for Microsoft
plt.figure(figsize=(14, 7))
plt.plot(microsoft['Date'], microsoft['Close'], color='b', label='Historical Data')
plt.plot(forecast_microsoft['ds'], forecast_microsoft['yhat'], color='r', linestyle='--', label='Forecast')
plt.fill_between(forecast_microsoft['ds'], forecast_microsoft['yhat_lower'], forecast_microsoft['yhat_upper'], color='red', alpha=0.2)
plt.title('Microsoft Stock Price Forecast from 2025 to 2035')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.show()

# Prepare Tesla data for Prophet
tesla_prophet_df = tesla_df[['date', 'close']].rename(columns={'date': 'ds', 'close': 'y'})
print("Tesla Prophet DataFrame:")
print(tesla_prophet_df.head())
print(tesla_prophet_df.columns)

# Forecast Tesla stock prices for the next 10 years
forecast_tesla = forecast_prophet(tesla_prophet_df, periods=periods)

# Plot the results for Tesla
plt.figure(figsize=(14, 7))
plt.plot(tesla_prophet_df['ds'], tesla_prophet_df['y'], color='b', label='Historical Data')
plt.plot(forecast_tesla['ds'], forecast_tesla['yhat'], color='r', linestyle='--', label='Forecast')
plt.fill_between(forecast_tesla['ds'], forecast_tesla['yhat_lower'], forecast_tesla['yhat_upper'], color='red', alpha=0.2)
plt.title('Tesla Stock Price Forecast from 2025 to 2035')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.show()

# Prepare Bitcoin data for Prophet
bitcoin_prophet_df = bitcoin_df[['Date', 'btc_market_price']].rename(columns={'Date': 'ds', 'btc_market_price': 'y'})
print("Bitcoin Prophet DataFrame:")
print(bitcoin_prophet_df.head())
print(bitcoin_prophet_df.columns)

# Forecast Bitcoin prices for the next 10 years
forecast_bitcoin = forecast_prophet(bitcoin_prophet_df, periods=periods)

# Plot the results for Bitcoin
plt.figure(figsize=(14, 7))
plt.plot(bitcoin_prophet_df['ds'], bitcoin_prophet_df['y'], color='b', label='Historical Data')
plt.plot(forecast_bitcoin['ds'], forecast_bitcoin['yhat'], color='r', linestyle='--', label='Forecast')
plt.fill_between(forecast_bitcoin['ds'], forecast_bitcoin['yhat_lower'], forecast_bitcoin['yhat_upper'], color='red', alpha=0.2)
plt.title('Bitcoin Price Forecast from 2025 to 2035')
plt.xlabel('Date')
plt.ylabel('Market Price')
plt.legend()
plt.show()
