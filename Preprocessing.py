# %%   1. Concatenate datasets
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import silhouette_score

# %% Load datasets
boulder = pd.read_excel('Boulder.xlsx')
cary = pd.read_excel('Cary.xlsx')
palo = pd.read_excel('Palo Alto.xlsx')


boulder = boulder.drop(['lat', 'long'], axis=1)

# %%
def clean_column_names(df):
    df.columns = (df.columns
                  .str.strip() 
                  .str.lower() 
                  .str.replace("longtitude", "longitude")
                  .str.replace("startdate/time", "start_date/time")
                  .str.replace("zip code", "zipcode")  
                  .str.replace("enddate/time", "end_date/time") 
                  .str.replace("chargedenergy (kwh)", "charged_energy")  
                  .str.replace("address", "address"))
    return df

boulder = clean_column_names(boulder)
palo = clean_column_names(palo)
cary = clean_column_names(cary)
boulder['stationname'] = boulder['stationname'].apply(lambda x: x.replace(" ", ""))
palo['stationname'] = palo['stationname'].apply(lambda x: x.replace(" ", ""))
cary['stationname'] = cary['stationname'].apply(lambda x: x.replace(" ", ""))

# %% Format the time 
def fix_slash_dates(date_str):
    if pd.isna(date_str) or date_str.strip() == '':
        return None  
    date_str = date_str.strip()  
    if '/' in date_str:  
        try:
            return pd.to_datetime(date_str, dayfirst=True).strftime('%Y-%m-%d %H:%M:%S')
        except ValueError:
            return None  
    
    return date_str

boulder['start_date/time'] = boulder['start_date/time'].apply(fix_slash_dates)
boulder['end_date/time'] = boulder['end_date/time'].apply(fix_slash_dates)

boulder["start_date/time"] = pd.to_datetime(boulder["start_date/time"], errors="coerce", dayfirst=True)
boulder["end_date/time"] = pd.to_datetime(boulder["end_date/time"], errors="coerce", dayfirst=True)

palo["start_date/time"] = pd.to_datetime(palo["start_date/time"], errors="coerce", dayfirst=True)
palo["end_date/time"] = pd.to_datetime(palo["end_date/time"], errors="coerce", dayfirst=True)

# Merge the weather datasets
boulder['timestamp_hour'] = boulder['start_date/time'].dt.round('H')
palo['timestamp_hour'] = palo['start_date/time'].dt.round('H')

boulder = boulder[boulder['city'] == 'Boulder'].merge(b_weather, on='timestamp_hour', how='left')
palo = palo[palo['city'] == 'Palo Alto'].merge(p_weather, on='timestamp_hour', how='left')

# %%
combined_data = pd.concat([boulder, palo], ignore_index=True)
combined_data = combined_data.sort_values(by="start_date/time")
combined_data = combined_data.reset_index(drop=True) 
combined_data.to_csv('combined_data.csv', index=False)

# %%
train = pd.read_csv('combined_data.csv', index_col=False)

# %%  2. EDA
# Convert duration to minutes
train["duration"] = pd.to_timedelta(train["duration"])
train["duration"] = (train["duration"].dt.total_seconds()/ 3600)

#  Convert time into year, month, day, hour and week day
def extract_datetime(df, time1, time2):
    df[time1] = pd.to_datetime(df[time1], errors="coerce", utc=False)
    df[time2] = pd.to_datetime(df[time2], errors="coerce", utc=False)
    
    df["year"] = df[time1].dt.year.astype("Int32")
    df["month"] = df[time1].dt.month.astype("Int32")
    df["day"] = df[time1].dt.day.astype("Int32")
    df["hour"] = df[time1].dt.hour.astype("Int64")
    df["day_of_week"] = df[time1].dt.weekday.astype("Int64") # 0 = Monday & 6 = Sunday
    df['weekday'] = df[time1].dt.day_name()
    return df
train = extract_datetime(train, 'start_date/time', 'end_date/time')
train = train.drop_duplicates(subset=['stationname', 'start_date/time', 'end_date/time', 'duration', 'charged_energy'])

# %% Energy load distribution 
plt.figure(figsize=(10, 6))
sns.histplot(train['charged_energy'], bins=30, kde=True)
plt.xlabel('Energy Consumption (kWh)')
plt.tight_layout()
#plt.savefig('Avg_energy.jpg', dpi=300)
plt.show()


# %%
# Analyze energy load in each station in each city
energy_agg = train.groupby(['city', 'stationname'])['charged_energy'].sum().reset_index()
cities = energy_agg["city"].unique()
cmap = plt.get_cmap("tab10")
colors = [cmap(i) for i in range(len(cities))]
fig, axes = plt.subplots(1, len(cities), figsize=(30, 10), sharey=True)
for i, city in enumerate(cities):
    ax = axes[i]
    city_data = energy_agg[energy_agg["city"] == city]

    ax.bar(city_data["stationname"], city_data["charged_energy"], color=colors[i])
    ax.set_xlabel("Station Name")
    ax.set_ylabel("Total Energy Load")
    ax.tick_params(axis="x", rotation=90) 
plt.tight_layout()
#plt.savefig('station_usage.jpg', dpi=600)
plt.show()

# %%
# Analyze energy load in each day in each city
train["Date"] = pd.to_datetime(train[["year", "month", "day"]])
energy_year = train.groupby(["city", "Date"])["charged_energy"].sum().reset_index()
cities = energy_year["city"].unique()
fig, axs = plt.subplots(len(cities), 1, figsize=(20, 10), constrained_layout=True)
if len(cities) == 1:
    axs = [axs]
for i, city in enumerate(cities):
    city_data = energy_year[energy_year["city"] == city].set_index("Date")

    axs[i].plot(city_data.index, city_data["charged_energy"], linestyle="-", color='tab:blue')
    axs[i].set_ylabel("Total Charged Energy (kWh)", fontsize=18)
    axs[i].grid(True)
    axs[i].tick_params(axis='x', rotation=45)
    if i == 1:  
        axs[i].set_xlabel("Date", fontsize=18)
plt.show()


# %%
# Analyze energy load in each month 
month_dict = {1: "January", 2: "February", 3: "March", 4: "April", 5: "May", 6: "June", 7: "July", 
              8: "August", 9: "September", 10: "October", 11: "November", 12: "December"}

train["month2"] = train["month"].map(month_dict)
month_energy = train.groupby("month2")["charged_energy"].sum().reset_index()

# Ensure months are sorted correctly
month_energy["month2"] = pd.Categorical(month_energy["month2"],
    categories=["January", "February", "March", "April", "May", "June", 
                "July", "August", "September", "October", "November", "December"],
    ordered=True)

month_energy = month_energy.sort_values("month2")
plt.figure(figsize=(12, 6))
sns.barplot(data=month_energy, x="month2", y="charged_energy")

plt.xlabel("Month")
plt.ylabel("Total Charged Energy (kWh)")
plt.xticks(rotation=0)
plt.tight_layout()
#plt.savefig('monthly_demand.jpg', dpi=300)
plt.show()

# %% Group by weekday and sum the energy
energy_by_day = train.groupby('weekday')['charged_energy'].sum().reindex([
    'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])

plt.figure(figsize=(12, 6))
energy_by_day.plot(kind='bar')
plt.ylabel('Total Energy (kWh)')
plt.xlabel('Day of the Week')
plt.xticks(rotation=0)
plt.tight_layout()
#plt.savefig('weekly_demand.jpg', dpi=300)
plt.show()

# Heatmap of energy load across week days and hours 
heatmap_data = train.groupby(['weekday', 'hour'])['charged_energy'].sum().unstack(fill_value=0)
heatmap_data = heatmap_data.reindex(["Monday", "Tuesday", "Wednesday", "Thursday", 
                                     "Friday", "Saturday", "Sunday"])
plt.figure(figsize=(15, 10))
sns.heatmap(heatmap_data, cmap="YlGnBu", annot=False, fmt=".1f") 
plt.xlabel("Hour of Day")
plt.ylabel("Day of Week")
plt.xticks(rotation=0) 
plt.tight_layout()
#plt.savefig('daily_consumption.jpg', dpi=300)
plt.show()

# %% Heat map accross stations
zipcode_data = train.groupby('zipcode').agg({'charged_energy': 'sum', 'hour': 'mean','day_of_week': lambda x: x.mode()[0]}).reset_index()

scaler = StandardScaler()
scaled_features = scaler.fit_transform(zipcode_data[['charged_energy', 'hour', 'day_of_week']])

k_values = range(4, min(11, len(scaled_features)))  
silhouette_scores = []

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(scaled_features)

    silhouette = silhouette_score(scaled_features, labels)
    silhouette_scores.append(silhouette)
    
optimal_k = k_values[np.argmax(silhouette_scores)]
kmeans = KMeans(n_clusters=optimal_k, random_state=0, n_init=10)
zipcode_data['cluster'] = kmeans.fit_predict(scaled_features)

zipcode_data = zipcode_data.merge(train[['zipcode', 'stationname']], on='zipcode', how='left').drop_duplicates()

selected_cluster = np.random.choice(zipcode_data['cluster'].unique())
cluster_zipcodes = zipcode_data[zipcode_data['cluster'] == selected_cluster]['zipcode'].unique()

if len(cluster_zipcodes) < 4:
    remaining_zipcodes = zipcode_data[~zipcode_data['zipcode'].isin(cluster_zipcodes)]['zipcode'].unique()
    np.random.shuffle(remaining_zipcodes)
    selected_zipcodes = list(cluster_zipcodes) + list(remaining_zipcodes[:4 - len(cluster_zipcodes)])
else:
    selected_zipcodes = np.random.choice(cluster_zipcodes, 4, replace=False)

weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

fig, axs = plt.subplots(2, 2, figsize=(15, 10))
for i, code in enumerate(selected_zipcodes):
    station_subset = train[train['zipcode'] == code].groupby(['day_of_week', 'hour'])['charged_energy'].sum().unstack()
    
    # Map index to weekday names
    station_subset.index = station_subset.index.map(lambda x: weekday_order[int(x)] if pd.notna(x) else x)
    station_subset = station_subset.reindex(weekday_order)  

    sns.heatmap(station_subset, cmap="YlGnBu", ax=axs[i // 2, i % 2])
    axs[i // 2, i % 2].set_title(f"Zipcode: {code}")
    axs[i // 2, i % 2].set_xlabel("Hour")
    axs[i // 2, i % 2].set_ylabel("Day of Week")
    axs[i // 2, i % 2].tick_params(axis='x', rotation=0)
    axs[i // 2, i % 2].tick_params(axis='y', rotation=0)

plt.tight_layout(rect=[0, 0, 1, 0.96])  
#plt.savefig('zipcode_consumption.jpg', dpi=300)
plt.show()


# %%
# Comparation of charged energy and charging duration on each day
energy_duration = train.groupby('Date').agg({'charged_energy': 'sum', 'duration': 'sum'}).reset_index()
fig, ax1 = plt.subplots(figsize=(15, 6)) 

ax1.plot(energy_duration['Date'], energy_duration['charged_energy'], label="Charged Energy (kWh)", color='blue')
ax1.set_xlabel("Date", fontsize=14)
ax1.set_ylabel("Total Charged Energy (kWh)", fontsize=14)
ax1.tick_params(axis='y')

ax2 = ax1.twinx()
ax2.plot(energy_duration['Date'], energy_duration['duration'], label="Charging Duration Avergae (hour)", color='orange')
ax2.set_ylabel("Charging Duration (hour)", fontsize=14)
ax2.tick_params(axis='y')
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%Y'))
plt.xticks(rotation=0)
plt.tight_layout()
plt.grid(True)
#plt.savefig('duration_relation_energy.jpg', dpi=300)
plt.show()


# %% Decomposition
train["Date"] = pd.to_datetime(train[["year", "month", "day"]])

energy_year = train.groupby("Date")["charged_energy"].sum()
decomposition = seasonal_decompose(energy_year, model="additive", period=30)

trend = decomposition.trend.dropna()
seasonal = decomposition.seasonal.dropna()
residual = decomposition.resid.dropna()

seasonal_smooth = seasonal.rolling(window=365, center=True).mean()  
energy_year_aligned = energy_year.loc[trend.index]

fig, axs = plt.subplots(4, 1, figsize=(25, 15))
axs[0].plot(energy_year_aligned.index, energy_year_aligned, linestyle="-", color="tab:blue", label="Original")
axs[0].legend()
axs[0].grid(True)

axs[1].plot(trend.index, trend, linestyle="-", color='tab:blue', label="Trend")
axs[1].legend()
axs[1].grid(True)

axs[2].plot(seasonal_smooth.index, seasonal_smooth, linestyle="-",  color='tab:blue', label="Seasonality")
axs[2].legend()
axs[2].grid(True)

axs[3].plot(residual.index, residual, linestyle="-", color='tab:blue', label="Residuals")
axs[3].legend()
axs[3].grid(True)

plt.tight_layout()
plt.grid(True)
#plt.savefig('decomposition.jpg', dpi=300)
plt.show()


# %% 3. Data cleaning
# Ouliers
plt.figure(figsize=(50, 8))
sns.boxplot(data=train, x='stationname', y='charged_energy')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Drop the stations that are less than 10-time uses
station_counts = train['stationname'].value_counts()
stations_to_drop = station_counts[station_counts < 10].index
train = train[~train['stationname'].isin(stations_to_drop)]





# %%
train = train.sort_values('start_date/time')
train.set_index('start_date/time', inplace=True)

# %%
daily_energy = train['charged_energy'].resample('D').mean()

rolling_7 = daily_energy.rolling(window=7).mean()
rolling_30 = daily_energy.rolling(window=30).mean()

plt.figure(figsize=(20, 10))
plt.plot(daily_energy, label='Daily Avg', alpha=0.4,color='gray')
plt.plot(rolling_7, label='7-Day Rolling Avg',linewidth=1.5)
plt.plot(rolling_30, label='30-Day Rolling Avg', color='orange',linewidth= 0.6)
plt.xlabel('Date')
plt.ylabel('Charged Energy (kWh)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('rolling_avg.jpg', dpi=600)
plt.show()

# %%
data = pd.read_csv('combined_data.csv')
data['zipcode'] = data['zipcode'].astype(str)

boulder_data = data[data['city'].str.lower() == 'boulder']
paloalto_data = data[data['city'].str.lower() == 'palo alto']

gdf = gpd.read_file("/Users/linhchi/.Trash/tl_2024_us_zcta520/tl_2024_us_zcta520.shp")
gdf['postcode'] = gdf['ZCTA5CE20'].astype(str)

boulder_zips = boulder_data['zipcode'].unique()
paloalto_zips = paloalto_data['zipcode'].unique()


gdf_boulder = gdf[gdf['postcode'].isin(boulder_zips)]
gdf_paloalto = gdf[gdf['postcode'].isin(paloalto_zips)]

merged_boulder = gdf_boulder.merge(
    boulder_data[['zipcode', 'charged_energy']],
    left_on='postcode',
    right_on='zipcode',
    how='left'
)
merged_boulder['charged_energy'] = merged_boulder['charged_energy'].fillna(0)

merged_paloalto = gdf_paloalto.merge(
    paloalto_data[['zipcode', 'charged_energy']],
    left_on='postcode',
    right_on='zipcode',
    how='left')
merged_paloalto['charged_energy'] = merged_paloalto['charged_energy'].fillna(0)

merged_boulder = merged_boulder.to_crs(epsg=4326)
merged_paloalto = merged_paloalto.to_crs(epsg=4326)

fig, ax = plt.subplots(1, 2, figsize=(16, 8))
merged_boulder.plot(
    column='charged_energy',
    cmap='Blues',
    linewidth=0.6,
    edgecolor='0.8',
    ax=ax[0],
    legend=True,
    legend_kwds={'label': "Charging Energy (kWh)", 'shrink': 0.6})
ax[0].set_title("Boulder")
ax[0].set_xlabel("Longitude")
ax[0].set_ylabel("Latitude")

merged_paloalto.plot(
    column='charged_energy',
    cmap='Blues',
    linewidth=0.6,
    edgecolor='0.8',
    ax=ax[1],
    legend=True,
    legend_kwds={'label': "Charging Energy (kWh)", 'shrink': 0.6})
ax[1].set_title("Palo Alto")
ax[1].set_xlabel("Longitude")
ax[1].set_ylabel("Latitude")

plt.tight_layout()
plt.savefig('boulder_and_paloalto_heatmaps.jpg', dpi=300)
plt.show()










# %%
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

train = pd.read_csv('feature_data.csv', index_col=False)
train["date"] = pd.to_datetime(train["date"], errors='coerce')  
train["hour"] = train['date'].dt.hour 

label_encoder = LabelEncoder()
train['day_of_week'] = label_encoder.fit_transform(train['day_of_week'])
label_encoder = LabelEncoder()
train['land_types'] = label_encoder.fit_transform(train['land_types'])

# Assuming train DataFrame is already loaded
zipcode_data = train.groupby('land_types').agg({
    'charged_energy': 'sum', 
    'hour': 'mean',
    'day_of_week': lambda x: x.mode()[0]
}).reset_index()

# Encoding 'day_of_week' for numerical scaling
label_encoder = LabelEncoder()
zipcode_data['day_of_week_encoded'] = label_encoder.fit_transform(zipcode_data['day_of_week'])

# Scaling features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(zipcode_data[['charged_energy', 'hour', 'day_of_week_encoded']])

# Find optimal k for KMeans
k_values = range(4, min(11, len(scaled_features)))
silhouette_scores = []

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(scaled_features)
    silhouette = silhouette_score(scaled_features, labels)
    silhouette_scores.append(silhouette)

optimal_k = k_values[np.argmax(silhouette_scores)]
kmeans = KMeans(n_clusters=optimal_k, random_state=0, n_init=10)
zipcode_data['cluster'] = kmeans.fit_predict(scaled_features)

# Merge with original train to get land_types and remove duplicates
zipcode_data = zipcode_data.merge(train[['land_types']], on='land_types', how='left').drop_duplicates()

# Select a random cluster
selected_cluster = np.random.choice(zipcode_data['cluster'].unique())
cluster_zipcodes = zipcode_data[zipcode_data['cluster'] == selected_cluster]['land_types'].unique()

# Ensure 4 zipcodes are selected
if len(cluster_zipcodes) < 4:
    remaining_zipcodes = zipcode_data[~zipcode_data['land_types'].isin(cluster_zipcodes)]['land_types'].unique()
    np.random.shuffle(remaining_zipcodes)
    selected_zipcodes = list(cluster_zipcodes) + list(remaining_zipcodes[:4 - len(cluster_zipcodes)])
else:
    selected_zipcodes = np.random.choice(cluster_zipcodes, 4, replace=False)

# Weekday order
weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# Create subplots for each selected zipcode
fig, axs = plt.subplots(2, 2, figsize=(15, 10))
for i, code in enumerate(selected_zipcodes):
    station_subset = train[train['land_types'] == code].groupby(['day_of_week', 'hour'])['charged_energy'].sum().unstack()
    
    # Map index to weekday names and reindex
    station_subset.index = station_subset.index.map(lambda x: weekday_order[int(x)] if pd.notna(x) else "Unknown")
    station_subset = station_subset.reindex(weekday_order)

    # Plot heatmap
    sns.heatmap(station_subset, cmap="YlGnBu", ax=axs[i // 2, i % 2], cbar_kws={'label': 'Energy Consumption (kWh)'})
    axs[i // 2, i % 2].set_title(f"Zone: {code}")
    axs[i // 2, i % 2].set_xlabel("Hour")
    axs[i // 2, i % 2].set_ylabel("Day of Week")
    axs[i // 2, i % 2].tick_params(axis='x', rotation=0)
    axs[i // 2, i % 2].tick_params(axis='y', rotation=0)

plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for space
plt.savefig('landtypes_consumption.jpg', dpi=600)
plt.show()