import pandas as pd
import os
from pathlib import Path
import pickle
from typing import Dict, Any

def load_vst_file(file_path):
    """Load a VST file with multiple encoding attempts."""
    encodings = ['utf-8', 'latin1', 'cp1252', 'iso-8859-1']
    
    for encoding in encodings:
        try:
            df = pd.read_csv(file_path, 
                           encoding=encoding,
                           delimiter=';',
                           decimal=',',
                           skiprows=3,
                           names=['Date', 'Value'])
            
            df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y %H:%M')
            df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
            
            # Set Date as index
            df.set_index('Date', inplace=True)
            df.index.name = 'Date'  # Ensure index is named Date

            
            return df
        except UnicodeDecodeError:
            continue
        except Exception as e:
            print(f"Error loading {file_path}: {str(e)}")
    
    return None

def load_vinge_file(file_path):
    """Load and process a VINGE file."""
    try:
        # Read the file with the correct delimiter and decimal separator
        df = pd.read_csv(file_path,
                        delimiter='\t',
                        encoding='latin1',
                        decimal=',',
                        quotechar='"',
                        on_bad_lines='skip')
        
        # Clean up the column names (remove any quotes and spaces)
        df.columns = df.columns.str.strip().str.replace('"', '')
        
        # Convert Date column to datetime
        df['Date'] = pd.to_datetime(df['Date'], format='%d.%m.%Y %H:%M')
        
        # Convert water level to mm (from cm)
        # First convert to string, then handle the comma decimal separator
        df['W.L [cm]'] = df['W.L [cm]'].astype(str)
        df['W.L [cm]'] = pd.to_numeric(df['W.L [cm]'].str.replace(',', '.'), errors='coerce') * 10
        
        # Filter out rows with NaN values in essential columns
        df = df.dropna(subset=['Date', 'W.L [cm]'])
        
        # Filter years if needed
        df = df[df['Date'].dt.year >= 1990]
        
        # Set Date as index
        df.set_index('Date', inplace=True)
        df.index.name = 'Date'  # Ensure index is named Date
        
        # Keep only the water level column
        df = df[['W.L [cm]']]
        
        # Sort by date
        df = df.sort_index()
        
        if len(df) == 0:
            print(f"Warning: No valid data found in {file_path}")
            return None
            
        return df
        
    except Exception as e:
        print(f"Error loading VINGE file {file_path}: {str(e)}")
        return None

def load_folder_data(folder_path):
    """Load all relevant data files from a folder."""
    data = {
        'vst_raw': None,
        'vst_edt': None,
        'vinge': None
    }
    
    # Load VST files
    vst_raw_path = os.path.join(folder_path, 'VST_RAW.txt')
    vst_edt_path = os.path.join(folder_path, 'VST_EDT.txt')
    vinge_path = os.path.join(folder_path, 'VINGE.txt')
    
    data['vst_raw'] = load_vst_file(vst_raw_path)
    data['vst_edt'] = load_vst_file(vst_edt_path)
    data['vinge'] = load_vinge_file(vinge_path)
    
    return data

def load_all_folders(base_path, folders):
    """Load data from multiple folders."""
    all_data = {}
    
    for folder in folders:
        folder_path = os.path.join(base_path, folder)
        all_data[folder] = load_folder_data(folder_path)
    
    return all_data

def prepare_data_for_error_detection(file_path: str) -> pd.DataFrame:
    """
    Prepare VST_RAW data specifically for error detection pipeline.
    
    Args:
        file_path: Path to VST_RAW.txt
    
    Returns:
        DataFrame with:
        - Consistent 15-min intervals
        - DateTime index
        - Single 'Value' column
        - No initial missing values
    """
    # Load raw data using existing function
    raw_data = load_vst_file(file_path)  # Now already has Date as index
    
    # Ensure consistent time intervals using the index
    clean_data = raw_data.resample('15T').asfreq()
    
    # Basic validation
    assert not clean_data['Value'].isna().any(), "Missing values in clean data"
    assert clean_data.index.is_monotonic_increasing, "Time index not monotonic"
    
    return clean_data

def get_data_path():
    """Get the path to the data directory."""
    # Get the current file's directory and navigate to the sample data
    current_dir = Path(__file__).parent  # We're already in '0. Data'
    return current_dir / "Sample data"

def get_plot_path():
    """Get the path to the plots directory."""
    current_dir = Path(__file__).parent  # We're already in '0. Data'
    plot_dir = current_dir / "plots"
    plot_dir.mkdir(exist_ok=True)
    return plot_dir

def load_rainfall_data(data_dir: Path, station_id: int) -> pd.DataFrame:
    """Load rainfall data for a specific measuring station by finding its closest rain station."""
    current_dir = Path(__file__).parent
    rain_data_dir = current_dir / 'Rain_data'
    
    # First, read the mapping file to find the closest rain station
    mapping_file = rain_data_dir / 'closest_rain_temp_stations.csv'
    try:
        mapping_df = pd.read_csv(mapping_file)
        closest_station = mapping_df[mapping_df['Station_of_Interest'] == station_id]['Closest_Rain_Station'].iloc[0]
        
        station_id_padded = f"{int(closest_station):05d}"
        rain_file = rain_data_dir / f'RainData_{station_id_padded}.csv'
        
        if rain_file.exists():
            df = pd.read_csv(rain_file)
            df['Date'] = pd.to_datetime(df['datetime'])  # Convert datetime to Date
            df.drop('datetime', axis=1, inplace=True)  # Drop the original datetime column
            df.set_index('Date', inplace=True)  # Use Date as index
            return df
        else:
            print(f"Rainfall data file not found for closest rain station {closest_station}")
            return None
            
    except Exception as e:
        print(f"Error loading rainfall data for measuring station {station_id}: {str(e)}")
        return None

def load_temperature_data(data_dir: Path, station_id: int) -> pd.DataFrame:
    """Load temperature data for a specific measuring station by finding its closest temperature station."""
    current_dir = Path(__file__).parent
    temp_data_dir = current_dir / 'Rain_data'
    
    mapping_file = temp_data_dir / 'closest_rain_temp_stations.csv'
    try:
        mapping_df = pd.read_csv(mapping_file)
        closest_station = mapping_df[mapping_df['Station_of_Interest'] == station_id]['Closest_Temp_Station'].iloc[0]
        
        station_id_padded = f"{int(closest_station):05d}"
        temp_file = temp_data_dir / f'TempData_{station_id_padded}.csv'
        
        if temp_file.exists():
            df = pd.read_csv(temp_file)
            df['Date'] = pd.to_datetime(df['datetime'])  # Convert datetime to Date
            df.drop('datetime', axis=1, inplace=True)  # Drop the original datetime column
            df.set_index('Date', inplace=True)  # Use Date as index
            return df
        else:
            print(f"Temperature data file not found for closest temperature station {closest_station}")
            return None
            
    except Exception as e:
        print(f"Error loading temperature data for measuring station {station_id}: {str(e)}")
        return None

def load_all_station_data() -> dict:
    """
    Main function to load and consolidate all station data.
    
    Returns:
        dict: Dictionary with station names as keys and DataFrames containing:
            - vst_raw: Raw VST data
            - vst_edt: Edited VST data
            - vinge: VINGE data
            - rainfall: Associated rainfall data (if available)
            - temperature: Associated temperature data (if available)
    """
    data_path = get_data_path()
    stations = [d for d in data_path.iterdir() if d.is_dir()]
    
    all_station_data = {}
    
    for station_path in stations:
        station_name = station_path.name
        
        # Load the station's data using existing function
        station_data = load_folder_data(station_path)
        
        # Try to load associated rainfall and temperature data if station ID can be extracted
        try:
            station_id = int(station_name.split('_')[0])  # Assuming format: "XXXXX_StationName"
            rainfall_data = load_rainfall_data(data_path, station_id)
            temperature_data = load_temperature_data(data_path, station_id)
            
            if rainfall_data is not None:
                station_data['rainfall'] = rainfall_data
            if temperature_data is not None:
                station_data['temperature'] = temperature_data
                
        except (ValueError, IndexError):
            print(f"Could not extract station ID from {station_name}, skipping rainfall and temperature data")
        
        all_station_data[station_name] = station_data
    
    return all_station_data

def save_data_Dict(data: Dict[str, Any], filename: str = 'preprocessed_data.pkl') -> None:
    """
    Save preprocessed data to a pickle file.
    
    Args:
        data: Dictionary containing preprocessed data
        filename: Name of the file to save the data to
    """
    # Get the universal path using the existing get_data_path function
    output_dir = get_data_path()
    output_path = output_dir / filename
    
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)
    print(f"Saved data to {output_path}")

