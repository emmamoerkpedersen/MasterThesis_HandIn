import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
import numpy as np

# Read input files
catchments = gpd.read_file('oplande.gpkg', engine="pyogrio")
stations_of_interest = gpd.read_file('Stations_45_46_46.shp', engine="pyogrio")
stations_of_interest = stations_of_interest.drop_duplicates(subset=['StedID'])
rain_stations = pd.read_csv('DMIStations.csv')


# Filter rain stations to only include Synop and Pluvio types as these holds precipitation data
rain_stations = rain_stations[rain_stations['type'].isin(['Synop', 'Pluvio'])]
# Create geometry for stations of interest from x and y coordinates
stations_geometry = [Point(xy) for xy in zip(stations_of_interest['x-koordina'], 
                                           stations_of_interest['y-koordina'])]
stations_of_interest = gpd.GeoDataFrame(stations_of_interest, 
                                      geometry=stations_geometry, 
                                      crs=catchments.crs)

# Convert rain stations DataFrame to GeoDataFrame
# Create points from lat/long and convert to the same CRS as catchments
geometry = [Point(xy) for xy in zip(rain_stations['longitude'], rain_stations['latitude'])]
rain_stations_gdf = gpd.GeoDataFrame(rain_stations, 
                                    geometry=geometry, 
                                    crs="EPSG:4326")  # WGS84 lat/long
# Reproject to match catchments CRS
rain_stations_gdf = rain_stations_gdf.to_crs(catchments.crs)

# Find closest rain station within the same catchment for each station of interest
results = []

for idx, station in stations_of_interest.iterrows():
    # Find which catchment contains the station of interest
    containing_catchment = catchments[catchments.contains(station.geometry)]
    
    if containing_catchment.empty:
        print(f"Warning: Station {station.name} is not within any catchment")
        continue
        
    # Find rain stations within the same catchment
    rain_stations_in_catchment = rain_stations_gdf[rain_stations_gdf.within(containing_catchment.iloc[0].geometry)]
    
    if rain_stations_in_catchment.empty:
        print(f"Warning: No rain stations found in the catchment containing station {station.name}")
        continue
    
    # Calculate distances to all rain stations in the catchment
    distances = rain_stations_in_catchment.geometry.distance(station.geometry)
    
    # Find the closest station
    closest_station_idx = distances.idxmin()
    closest_station = rain_stations_in_catchment.loc[closest_station_idx]
    
    results.append({
        'Station_of_Interest': station['StedID'],
        'Closest_Rain_Station': f"{closest_station['stationId']:05d}",
        'Distance_m': distances[closest_station_idx],
        'Catchment_ID': containing_catchment.iloc[0].name
    })

# Convert results to DataFrame
results_df = pd.DataFrame(results)
# Save results to CSV
output_path = '/Users/emmamork/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/Master Thesis/MasterThesis/closest_rain_stations.csv'
results_df.to_csv(output_path, index=False)
print(f"Analysis complete. Results saved to '{output_path}'")

