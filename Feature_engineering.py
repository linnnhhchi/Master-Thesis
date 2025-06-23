# %%
import pandas as pd
import numpy as np
import holidays
import matplotlib.pyplot as plt
import seaborn as sns
from pmdarima import auto_arima
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# %% Load final data 
df = pd.read_csv('final_data1.csv')

df["duration"] = pd.to_timedelta(df["duration"])
df["duration"] = (df["duration"].dt.total_seconds()/ 60).round(3)

# Convert time into year, month, day, hour and week day
def extract_datetime(df, time1, time2):
    df[time1] = pd.to_datetime(df[time1], errors="coerce", utc=False)
    df[time2] = pd.to_datetime(df[time2], errors="coerce", utc=False)
    
    df["year"] = df[time1].dt.year.astype("Int32")
    df["month"] = df[time1].dt.month.astype("Int32")
    df["day"] = df[time1].dt.day.astype("Int32")
    df["hour"] = df[time1].dt.hour.astype("Int64")
    df["day_of_week"] = df[time1].dt.day_name()
    return df
df = extract_datetime(df, 'start_date/time', 'end_date/time')
df["date"] = pd.to_datetime(df[["year", "month", "day", 'hour']])
df = df.sort_values(by=['date', 'hour'])
df= df.drop_duplicates(subset=['stationname', 'start_date/time', 'end_date/time', 'duration', 'charged_energy'])

station_counts = df['stationname'].value_counts()
stations_to_drop = station_counts[station_counts < 10].index
df = df[~df['stationname'].isin(stations_to_drop)]

# %% 4.2. Temporal features 
## Weekend indicator (1 = Weekday and 0 = Weekend)
df["weekend"] = df["day_of_week"].isin(["Saturday", "Sunday"]).map({True: "Yes", False: "No"})

## Holiday/Nation events (0 = Fales & 1 = True)
country_holidays = holidays.country_holidays('US')  
def holiday(date):
    if date in country_holidays:
        return 'yes'
    else:
        return 'no'
df['holiday'] = df['date'].dt.date.apply(holiday)

## Lockdown and COVID timeframe (0 = Fales & 1 = True)
def check_covid(df, date, year):
    covid_years = [2020, 2021]
    df['covid'] = (df[year].isin(covid_years))
    return df
df = check_covid(df, 'date', 'year')

## Apply time slots
def assign_time_slot(hour):
    if 0 <= hour < 5:
        return 'midnight'
    elif 5 <= hour < 8:
        return 'morning'
    elif 8 <= hour < 10:
        return 'morning rush'
    elif 10 <= hour < 15:
        return 'afternoon'
    elif 15 <= hour < 18:
        return 'afternoon rush'
    else:
        return 'evening'

df['time_slot'] = df['hour'].apply(assign_time_slot)

#time_slot_order = ['midnight', 'morning', 'morning rush', 'afternoon', 'afternoon rush', 'evening']
#df['time_slot'] = pd.Categorical(df['time_slot'], categories=time_slot_order, ordered=True)

# %% Log transformation 
def transform_log(df): 
    for col in df.select_dtypes('float'):
        skew_val = df[col].skew()
        if skew_val > 1: 
            df[col] = np.log1p(df[col])
        elif skew_val < 0:  
            df[col] = PowerTransformer(method='yeo-johnson').fit_transform(df[[col]])
    return df
df = transform_log(df)

# %% Hour transformation 
def transform_temporal_features(df):
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    return df

df = transform_temporal_features(df)

# %% Drop columns
df.set_index('date', inplace=True)
columns_to_drop = [
    'Unnamed: 0', 'stationname', 'start_date/time', 'end_date/time', 'address', 'city', 'weekday',
    'state/province', "energy_avg", "energy_std", 'day', 'hour', 'zipcode', 'latitude', 'longitude', 'timestamp_hour', 'month', 
    'duration', 'weekday']
df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

# %%
Q1 = df['charged_energy'].quantile(0.25)
Q3 = df['charged_energy'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df = df[(df['charged_energy'] >= lower_bound) & (df['charged_energy'] <= upper_bound)]


# %% Feature correlations
numeric_cols = df.select_dtypes('float64')
correlation_matrix = numeric_cols.corr()
plt.figure(figsize=(20, 15))
sns.heatmap(correlation_matrix, annot=True, cmap="Blues", fmt=".2f", linewidths=0.8, linecolor='black')
plt.title("Correlation Heatmap of Numeric Features", fontsize=20)
plt.xticks(fontsize=14, rotation=45, ha="right")  
plt.yticks(fontsize=14, rotation=0) 
plt.tight_layout()
plt.show()

# %% 5. Experiment setup for spatial feature inclusion
X = df.drop(columns=['charged_energy'])
y = df['charged_energy']

label_encoder = LabelEncoder()
X['land_types'] = label_encoder.fit_transform(X['land_types'])

train_size = int(len(X) * 0.8)  
val_size = int(len(X) * 0.1)   
test_size = len(X) - train_size - val_size 

X_train, X_val, X_test = X[:train_size], X[train_size:train_size + val_size], X[train_size + val_size:]
y_train, y_val, y_test = y[:train_size], y[train_size:train_size + val_size], y[train_size + val_size:]

# %% Standardize numeric features
numeric_features = ['year', 'temperature',
       'dewpoint', 'road_density', 'commercial_density', 'residential_density',
       'recreation_density', 'highway_proximity', 'public_transport_proximity',
       'evcs_proximity', 'center_proximity', 'parking_density', 'year', 'month_sin', 'month_cos', 'hour_sin', 'hour_cos']
multiple = ['land_types']
binary = list(set(X.columns) - set(numeric_features) - set(multiple))

preprocessor = ColumnTransformer(transformers=[('num', MinMaxScaler(), numeric_features), 
                                               ('cat', OneHotEncoder(),binary)])

X_train = preprocessor.fit_transform(X_train, y_train)
X_val = preprocessor.transform(X_val)
X_test = preprocessor.transform(X_test)


# %% Hyperparameters
## ARIMA Baseline
column_to_test = df['charged_energy']  
result = adfuller(column_to_test)
print(f'p-value: {result[1]}')
if result[1] > 0.05:
    print("Data is non-stationary. Consider differencing.")
else:
    print("Data is stationary.")

if y_train.isnull().any():
    print("Handling missing values...")
    y_train = y_train.fillna(method='ffill') 
adf_test = adfuller(y_train)
if adf_test[1] > 0.05:
    print("Series is non-stationary, differencing...")
    y_train = y_train.diff().dropna()
model = auto_arima(y_train, seasonal=True, m=24, trace=True)
model.summary()

#  %% ACF plot
energy_series = df['charged_energy'].resample('H').mean().fillna(0)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
plot_acf(energy_series, lags=24, ax=axes[0], alpha=0.05) 
axes[0].set_title('Autocorrelation')
plot_pacf(energy_series, lags=48, ax=axes[1], method='ywm', alpha=0.05)
axes[1].set_title('Partial Autocorrelation')

plt.tight_layout()
plt.savefig('ACF-PACF.png', dpi=300)
plt.show()

# %% Baseline ARIMA
model = ARIMA(y_train, order=(1,1,2))
arima= model.fit()
print(arima.summary())
val_pred = arima.forecast(steps=len(y_val))
test_pred = arima.forecast(steps=len(y_test))

rmse_val = np.sqrt(mean_squared_error(y_val, val_pred))
mae_val = mean_absolute_error(y_val, val_pred)
r2_val = r2_score(y_val, val_pred)

rmse_test = np.sqrt(mean_squared_error(y_test, test_pred))
mae_test = mean_absolute_error(y_test, test_pred)
r2_test = r2_score(y_test, test_pred)

print(f"Validation Set: RMSE = {rmse_val:.4f}, MAE = {mae_val:.4f}, R2= {r2_val:.4f}")
print(f"Test Set: RMSE = {rmse_test:.4f}, MAE = {mae_test:.4f}, R2= {r2_test:.4f}")












# %%
