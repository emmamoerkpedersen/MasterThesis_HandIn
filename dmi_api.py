import json
import os
from datetime import datetime
#import geopandas as gpd
import numpy as np
import pandas as pd
import requests

# Get the MasterThesis directory (parent directory of the script)
master_thesis_dir = os.path.dirname(os.path.abspath(__file__))

class MetObsAPI:
    """
    Class for fetching data from the MetObs API
    """

    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://dmigw.govcloud.dk/v2/metObs/collections/"

    def get_precipitation_from_station(self, station_id: str, time_from=None, time_to=None, limit=10000) -> dict:
        """Get precipitation data from a given station

        Parameters
        ----------
        station_id : str
            The station id
        time_from  : datetime
        time_to    : datetime
        limit      : int
            The number of data points to fetch. The maximum are 300000 at a time.

        Returns
        -------
        dict
            The response from the API
        """
        assert limit < 300000, "The limit is too large"
        time_arg = self.construct_datetime(time_from, time_to)
        url = f"{self.base_url}observation/items?stationId={station_id}&parameterId=precip_past1h&datetime={time_arg}&limit={limit}&api-key={self.api_key}"
        response = requests.get(url).json()
        response = response.get("features")
        location = response[0].get("geometry").get("coordinates")
        data = []
        for i in range(len(response)):
            tmp = response[i].get("properties")
            data.append([tmp["observed"], tmp["value"]])
        data = pd.DataFrame(data, columns=["datetime", "precipitation (mm)"])
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


if __name__ == "__main__":
    # Example of how to use the API
    my_api_key = "762c2c0b-caa9-4c96-a8a7-4eb2c719b359"
    client = MetObsAPI(my_api_key)
    
    # Build path to read from data folder
    input_path = os.path.join(master_thesis_dir, 'data', 'closest_rain_stations.csv')
    stations_df = pd.read_csv(input_path, dtype={'Closest_Rain_Station': str})
    station_ids = stations_df['Closest_Rain_Station'].unique()
    
    # Loop through each station and save its data
    for station_id in station_ids:
        print(f"Processing station {station_id}")
        try:
            data, location = client.get_precipitation_from_station(
                str(station_id),  # Convert to string in case IDs are numeric
                datetime(1990, 1, 1), 
                datetime(2025, 1, 7), 
                limit=299999
            )
            
            # Delete negative values
            # data = data[data['precipitation (mm)'] >= 0]
            
            # Save to CSV with station ID in filename
            output_filename = f'RainData_{station_id}.csv'
            # Build path to save in data folder
            output_path = os.path.join(master_thesis_dir, 'data', output_filename)
            data.to_csv(output_path)
            print(f"Saved data to {output_path}")
            
        except Exception as e:
            print(f"Error processing station {station_id}: {str(e)}")
            continue


# import matplotlib.pyplot as plt


# plt.figure(figsize=(15, 8))
# plt.plot(data)
# plt.xlabel('Time [days]')
# plt.ylabel('Water level [m]')
# #plt.legend((key[i],))
# plt.show()

