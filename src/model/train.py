import pandas as pd
import yfinance as yf
from datetime import date, timedelta
import joblib
from sklearn.ensemble import GradientBoostingRegressor

today = date.today()
start_date = (today - timedelta(days=730)).strftime("%Y-%m-%d")
end_date = today.strftime("%Y-%m-%d")

data = yf.download(
    'BTC-USD',
    start=start_date,
    end=end_date,
    progress=False,
    auto_adjust=False
)
data["Date"] = data.index

data = data[["Date", "Close"]]
data.reset_index(drop=True, inplace=True)

data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]

data['Date'] = pd.to_datetime(data['Date'])
data['Close'] = pd.to_numeric(data['Close'], errors='coerce')
data = data.dropna(subset=['Close'])

data['Close_lag1'] = data['Close'].shift(1)
data['Close_lag2'] = data['Close'].shift(2)
data['Close_lag3'] = data['Close'].shift(3)
data['Close_lag4'] = data['Close'].shift(4)
data['Close_lag5'] = data['Close'].shift(5)
data = data.dropna()

X = data[['Close_lag1', 'Close_lag2', 'Close_lag3', 'Close_lag4', 'Close_lag5']]
y = data['Close']

#model type was changed from AutoTS to GradientBoostingRegressor because AutoTS does not support continuation of the training
model = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=3,
    random_state=3,
    warm_start=True
)
model.fit(X, y)

joblib.dump(model, "joblib/model_v1_BTC.joblib")
print("Model trained and saved as model_v1_BTC.joblib")
