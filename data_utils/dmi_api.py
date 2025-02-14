import json
import os
from datetime import datetime
#import geopandas as gpd
import numpy as np
import pandas as pd
import requests



class MetObsAPI:
    """
    Class for fetching data from the MetObs API
    """

    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://dmigw.govcloud.dk/v2/metObs/collections/"

    def get_data_from_station(self, station_id: str, parameter_id: str, time_from=None, time_to=None, limit=10000) -> tuple:
        """Get data from a given station for a specific parameter

        Parameters
        ----------
        station_id : str
            The station id
        parameter_id : str
            The parameter to fetch (e.g., 'precip_past1h' or 'temp_mean_past1h')
        time_from : datetime
        time_to : datetime
        limit : int
            The number of data points to fetch. The maximum are 300000 at a time.

        Returns
        -------
        tuple
            (DataFrame with the data, location coordinates)
        """
        assert limit < 300000, "The limit is too large"
        time_arg = self.construct_datetime(time_from, time_to)
        url = f"{self.base_url}observation/items?stationId={station_id}&parameterId={parameter_id}&datetime={time_arg}&limit={limit}&api-key={self.api_key}"
        response = requests.get(url).json()
        features = response.get("features", [])
        
        if not features:
            raise ValueError(f"No data found for station {station_id} with parameter {parameter_id}")
            
        location = features[0].get("geometry", {}).get("coordinates")
        if not location:
            raise ValueError(f"No location data found for station {station_id}")
            
        data = []
        for feature in features:
            properties = feature.get("properties", {})
            if "observed" in properties and "value" in properties:
                data.append([properties["observed"], properties["value"]])
        
        if not data:
            raise ValueError(f"No valid data points found for station {station_id}")
            
        column_name = "precipitation (mm)" if parameter_id == "precip_past1h" else "temperature (C)"
        data = pd.DataFrame(data, columns=["datetime", column_name])
        data["datetime"] = pd.to_datetime(data["datetime"])
        data = data.set_index("datetime").sort_index()
        return data, location

    @staticmethod
    def construct_datetime(time_from=None, time_to=None):
        if time_from is None and time_to is None:
            return None
        elif time_from is None and time_to is not None:
            return f"../{time_to.isoformat()}Z"
        elif time_from is not None and time_to is None:
            return f"{time_from.isoformat()}Z/.."
        else:
            return f"{time_from.isoformat()}Z/{time_to.isoformat()}Z"

def get_data_dir():
    """Returns the path to the data directory"""
    # First check if there's an environment variable set
    data_dir = os.getenv('DMI_DATA_DIR')
    if data_dir:
        return data_dir

    # Otherwise, use the Rain_data directory relative to the script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, 'Rain_data')

if __name__ == "__main__":
    # Example of how to use the API
    my_api_key = "9d13ca23-5a3d-47fc-a63d-b91009a1879a"
    client = MetObsAPI(my_api_key)
    
    # Use the get_data_dir function
    data_dir = get_data_dir()
    input_path = os.path.join(data_dir, 'closest_rain_temp_stations.csv')
    
    # Read CSV with station IDs as int64
    stations_df = pd.read_csv(input_path, dtype={
        'Closest_Rain_Station': str,
        'Closest_Temp_Station': str
    })
    
    # Process rain stations
    rain_station_ids = stations_df['Closest_Rain_Station'].unique()
    for station_id in rain_station_ids:
        print(f"Processing rain station {station_id}")
        try:
            # Convert station_id to string with leading zeros
            formatted_station_id = f"{station_id}"
            data, location = client.get_data_from_station(
                formatted_station_id,
                "precip_past1h",
                datetime(1990, 1, 1), 
                datetime(2025, 1, 1), 
                limit=299999
            )
            
            # Delete negative values
            data = data[data['precipitation (mm)'] >= 0]
            
            # Save to CSV using formatted station 
            output_filename = f'RainData_{formatted_station_id}.csv'
            output_path = os.path.join(data_dir, output_filename)
            data.to_csv(output_path)
            print(f"Saved rain data to {output_path}")
            
        except Exception as e:
            print(f"Error processing rain station {station_id}: {str(e)}")
            continue
    
    # Process temperature stations
    temp_station_ids = stations_df['Closest_Temp_Station'].unique()
    for station_id in temp_station_ids:
        print(f"Processing temperature station {station_id}")
        try:
            # Convert station_id to string with leading zeros
            formatted_station_id = f"{station_id}"
            data, location = client.get_data_from_station(
                formatted_station_id,
                "temp_mean_past1h",
                datetime(1990, 1, 1), 
                datetime(2025, 1, 1), 
                limit=299999
            )
            
            # Save to CSV using formatted station ID
            output_filename = f'TempData_{formatted_station_id}.csv'
            output_path = os.path.join(data_dir, output_filename)
            data.to_csv(output_path)
            print(f"Saved temperature data to {output_path}")
            
        except Exception as e:
            print(f"Error processing temperature station {station_id}: {str(e)}")
            continue


# import matplotlib.pyplot as plt


# plt.figure(figsize=(15, 8))
# plt.plot(data)
# plt.xlabel('Time [days]')
# plt.ylabel('Water level [m]')
# #plt.legend((key[i],))
# plt.show()

test = pd.read_csv('/Users/emmamork/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/Master Thesis/MasterThesis/data_utils/Rain_data/closest_rain_stations.csv')