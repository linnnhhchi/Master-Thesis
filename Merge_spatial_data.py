# %%
import osmnx as ox
import numpy as np
import geopandas as gpd
from geopy.distance import geodesic
import pandas as pd

# %%
centers = {"Boulder": (40.015, -105.2705), "Palo Alto": (37.4419, -122.1430)}

def extract_ev_features(lat, lon, dist=300):
    try:
        # Validate latitude and longitude ranges
        if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
            return {"road_density": np.nan, "highway_proximity": np.nan, "commercial_density": np.nan, "residential_density": np.nan, "public_transport_proximity": np.nan, "ev_station_proximity": np.nan, "parking_density": np.nan, "park_density": np.nan, "center_proximity": np.nan}

        point = (lat, lon)
        
        def get_feature_count(tag, value):
            try:
                features = ox.features_from_point(point, tags={tag: value}, dist=dist)
                return len(features) if not features.empty else 0
            except:
                return 0
        
        def get_nearest_distance(tag, value, max_dist=1000):
            try:
                features = ox.features_from_point(point, tags={tag: value}, dist=max_dist)
                return features.geometry.apply(lambda x: geodesic((lat, lon), (x.centroid.y, x.centroid.x)).meters).min() if not features.empty else np.nan
            except:
                return np.nan
        
        # Calculate distance to the nearest city center
        city_center = min(centers.values(), key=lambda c: geodesic(point, c).meters)
        center_proximity = geodesic(point, city_center).meters

        # Extract features safely
        road_density = get_feature_count("highway", True)
        highway_proximity = get_nearest_distance("highway", ["motorway", "trunk", "primary", "secondary"], 500)
        commercial_density = get_feature_count("landuse", "commercial")
        residential_density = get_feature_count("landuse", "residential")
        public_transport_proximity = get_nearest_distance("highway", "bus_stop", 500)
        ev_station_proximity = get_nearest_distance("amenity", "charging_station", 1000)
        parking_density = get_feature_count("amenity", "parking")
        park_density = get_feature_count("leisure", "park")

        return {
            "road_density": road_density,
            "highway_proximity": highway_proximity,
            "commercial_density": commercial_density,
            "residential_density": residential_density,
            "public_transport_proximity": public_transport_proximity,
            "ev_station_proximity": ev_station_proximity,
            "parking_density": parking_density,
            "park_density": park_density,
            "center_proximity": center_proximity}
    
    except Exception as e:
        print(f"Error processing lat: {lat}, lon: {lon} -> {e}")
        return {"road_density": np.nan, "highway_proximity": np.nan, "commercial_density": np.nan, "residential_density": np.nan, "public_transport_proximity": np.nan, "ev_station_proximity": np.nan, "parking_density": np.nan, "park_density": np.nan, "center_proximity": np.nan}

spatial = pd.read_csv("locations.csv")
spatial['latitude'] = pd.to_numeric(spatial['latitude'], errors='coerce')
spatial['longitude'] = pd.to_numeric(spatial['longitude'], errors='coerce')
spatial = spatial.dropna(subset=['latitude', 'longitude'])
spatial = spatial[(spatial['longitude'] < -100) & (spatial['longitude'] > -130)]
features_df = spatial.apply(lambda row: pd.Series(extract_ev_features(row['latitude'], row['longitude'])), axis=1)
spatial = pd.concat([spatial, features_df], axis=1)
spatial.to_csv("station_zones.csv", index=False)


# %%
combined_data = pd.read_csv("combined_data.csv")
station_zones = pd.read_csv("station_zones.csv")

combined_data['stationname'] = combined_data['stationname'].str.strip()
station_zones['stationname'] = station_zones['stationname'].str.strip()

station_zones = station_zones.drop_duplicates(subset=['stationname'])
station_zones = station_zones.drop(['latitude', 'longitude', 'address'], axis=1)
merged_data = combined_data.merge(station_zones, on=["stationname"], how="left")
merged_data.to_csv("final_data.csv", index=False)

# %%
merged_data = pd.read_csv('final_data.csv')
null_poi_rows = merged_data[merged_data[['road_density', 'highway_proximity', 'commercial_density', 'residential_density',
       'public_transport_proximity', 'ev_station_proximity', 'parking_density', 'park_density', 'center_proximity']].isnull().any(axis=1)]
null_poi_info = null_poi_rows[['stationname', 'address', 'latitude', 'longitude']].drop_duplicates()
null_poi_info.to_csv("null_poi_density_stations.csv", index=False)

# %%
merged_data = pd.read_csv("final_data.csv") 
null_filled_data = pd.read_csv("null_poi_density_stations.csv")  
merged_data = merged_data.merge(
    null_filled_data[['stationname','road_density','highway_proximity','commercial_density',
                      'residential_density','public_transport_proximity','ev_station_proximity','parking_density', 'park_density', 'center_proximity']],
    on=['stationname'],
    how='left',
    suffixes=('', '_filled'))
for col in ['road_density','highway_proximity','commercial_density',
                      'residential_density','public_transport_proximity','ev_station_proximity','parking_density', 'park_density', 'center_proximity']:
    merged_data[col] = merged_data[col].fillna(merged_data[f"{col}_filled"])
    merged_data.drop(columns=[f"{col}_filled"], inplace=True) 
merged_data.to_csv("final_data.csv", index=False)

