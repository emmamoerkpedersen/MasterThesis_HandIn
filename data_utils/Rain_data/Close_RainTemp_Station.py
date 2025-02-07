import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
import numpy as np

## Script to find the closest rain station to each station of interest. 
## The results are saved to a csv file. 

def load_and_prepare_data(catchments_path, stations_path, dmi_stations_path):
    """
    Load and prepare the input data files
    """
    # Read input files
    catchments = gpd.read_file(catchments_path, engine="pyogrio")
    stations_of_interest = gpd.read_file(stations_path, engine="pyogrio")
    stations_of_interest = stations_of_interest.drop_duplicates(subset=['StedID'])
    dmi_stations = pd.read_csv(dmi_stations_path)
    
    # Filter stations for rain and temperature data
    rain_stations = dmi_stations[dmi_stations['precip_past1h'].notna() & (dmi_stations['precip_past1h'] != '-')]
    temp_stations = dmi_stations[dmi_stations['temp_mean_past1h'].notna() & (dmi_stations['temp_mean_past1h'] != '-')]
    
    # Create geometry for stations of interest
    stations_geometry = [Point(xy) for xy in zip(stations_of_interest['x-koordina'], 
                                               stations_of_interest['y-koordina'])]
    stations_of_interest = gpd.GeoDataFrame(stations_of_interest, 
                                          geometry=stations_geometry, 
                                          crs=catchments.crs)
    
    # Convert DMI stations DataFrames to GeoDataFrames
    def create_geodataframe(df):
        geometry = [Point(xy) for xy in zip(df['longitude'], df['latitude'])]
        gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
        return gdf.to_crs(catchments.crs)
    
    rain_stations_gdf = create_geodataframe(rain_stations)
    temp_stations_gdf = create_geodataframe(temp_stations)
    
    return catchments, stations_of_interest, rain_stations_gdf, temp_stations_gdf

def find_closest_rain_stations(catchments, stations_of_interest, rain_stations_gdf):
    """
    Find the closest rain station within the same catchment for each station of interest
    """
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
            'Closest_Rain_Station': f"{int(closest_stat