import sys
from pathlib import Path

# Add the parent directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

from data_utils.data_loading import load_all_station_data



def preprocess_data():
    All_station_data = load_all_station_data()

    for station_data in All_station_data:
        print(station_data)

print(preprocess_data())



