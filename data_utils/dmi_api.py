import json
import os
from datetime import datetime
#import geopandas as gpd
import numpy as np
import pandas as pd
import requests

# Get the MasterThesis directory (parent directory of data_utils)
master_thesis_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class MetObsAPI:
    """
    Class for fetching meteorological observation data from the DMI MetObs API.
    
    Attributes
    ----------
    api_key : str
        Authentication key for the DMI API
    base_url : str
        Base URL for the MetObs API endpoint
    """

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://dmigw.govcloud.dk/v2/metObs/collections/"

    def get_precipitation_from_station(self, station_id: str, time_from=None, time_to=None, limit: int = 10000) -> tuple[pd.DataFrame, list]:
        """Get hourly precipitation data from a given station.

        Parameters
        ----------
        station_id : str
            The station identifier
        time_from : datetime, optional
            Start time for data retrieval
        time_to : datetime, optional
            End time for data retrieval
        limit : int, default=10000
            Maximum number of data points to fetch (max 300000)

        Returns
        -------
        tuple[pd.DataFrame, list]
            DataFrame containing precipitation data and station coordinates [lon, lat]
        
        Raises
        ------
        ValueError
            If the limit exceeds 300000 or if the API response is invalid
        """
        if limit >= 300000:
            raise ValueError("The limit must be less than 300000")
            
        time_arg = self.construct_datetime(time_from, time_to)
        url = f"{self.base_url}observation/items?stationId={station_id}&parameterId=precip_past1h&datetime={time_arg}&limit={limit}&api-key={self.api_key}"
        
        response = requests.get(url).json()
        features = response.get("features")
        
        if not features:
            raise ValueError(f"No data received for station {station_id}")
            
        location = features[0].get("geometry").get("coordinates")
        data = [[feat.get("properties").get("observed"), 
                feat.get("properties").get("value")] 
               for feat in features]
        
        data = pd.DataFrame(data, columns=["datetime", "precipitation (mm)"])
        data["datetime"] = pd.to_datetime(data["datetime"])
        return data.set_index("datetime").sort_index(), location

    def get_temperature_from_station(self, station_id: str, time_from=None, time_to=None, limit: int = 10000) -> tuple[pd.DataFrame, list]:
        """Get hourly temperature data from a given station.

        Parameters
        ----------
        station_id : str
            The station identifier
        time_from : datetime, optional
            Start time for data retrieval
        time_to : datetime, optional
            End time for data retrieval
        limit : int, default=10000
            Maximum number of data points to fetch (max 300000)

        Returns
        -------
        tuple[pd.DataFrame, list]
            DataFrame containing temperature data and station coordinates [lon, lat]
        
        Raises
        ------
        ValueError
            If the limit exceeds 300000 or if the API response is invalid
        """
        if limit >= 300000:
            raise ValueError("The limit must be less than 300000")
            
        time_arg = self.construct_datetime(time_from, time_to)
        url = f"{self.base_url}observation/items?stationId={station_id}&parameterId=temp_mean_past1h&datetime={time_arg}&limit={limit}&api-key={self.api_key}"
        
        response = requests.get(url).json()
        features = response.get("features")
        
        if not features:
            raise ValueError(f"No data received for station {station_id}")
            
        location = features[0].get("geometry").get("coordinates")
        data = [[feat.get("properties").get("observed"), 
                feat.get("properties").get("value")] 
               for feat in features]
        
        data = pd.DataFrame(data, columns=["datetime", "temperature (C)"])
        data["datetime"] = pd.to_datetime(data["datetime"])
        return data.set_index("datetime").sort_index(), location

    @staticmethod
    def construct_datetime(time_from=None, time_to=None) -> str:
        """Construct datetime string for API query.

        Parameters
        ----------
        time_from : datetime, optional
            Start time
        time_to : datetime, optional
            End time

        Returns
        -------
        str
            Formatted datetime string for API query
        """
        if time_from is None and time_to is None:
            return None
        elif time_from is None and time_to is not None:
            return f"../{time_to.isoformat()}Z"
        elif time_from is not None and time_to is None:
            return f"{time_from.isoformat()}Z/.."
        else:
            return f"{time_from.isoformat()}Z/{time_to.isoformat()}Z"


def process_station_data(client: MetObsAPI, station_type: str):
    """Process and save data for all stations of a given type.
    
    Parameters
    ----------
    client : MetObsAPI
        Initialized API client
    station_type : str
        Either 'rain' or 'temperature'
    """
    # Build path to read from Rain_data folder
    input_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                             'Rain_data', 
                             'closest_rain_temp_stations.csv')
    
    column_name = 'Closest_Rain_Station' if station_type == 'rain' else 'Closest_Temp_Station'
    stations_df = pd.read_csv(input_path, dtype={column_name: str})
    station_ids = stations_df[column_name].unique()
    
    for station_id in station_ids:
        print(f"Processing station {station_id}")
        try:
            if station_type == 'rain':
                data, location = client.get_precipitation_from_station(
                    str(station_id),
                    datetime(1990, 1, 1),
                    datetime(2025, 1, 1),
                    limit=299999
                )
                # Remove negative precipitation values
                data = data[data['precipitation (mm)'] >= 0]
                prefix = 'RainData'
            else:
                data, location = client.get_temperature_from_station(
                    str(station_id),
                    datetime(1990, 1, 1),
                    datetime(2025, 1, 1),
                    limit=299999
                )
                prefix = 'TempData'
            
            output_filename = f'{prefix}_{station_id}.csv'
            output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                     'Rain_data', 
                                     output_filename)
            data.to_csv(output_path)
            print(f"Saved data to {output_path}")
            
        except Exception as e:
            print(f"Error processing station {station_id}: {str(e)}")
            continue


if __name__ == "__main__":
    my_api_key = "762c2c0b-caa9-4c96-a8a7-4eb2c719b359"
    client = MetObsAPI(my_api_key)
    
    # Process both rain and temperature stations
    process_station_data(client, 'rain')
    process_station_data(client, 'temperature')
