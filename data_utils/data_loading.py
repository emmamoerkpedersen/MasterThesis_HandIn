import pandas as pd
import os
from pathlib import Path

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
            
            return df
        except UnicodeDecodeError:
            continue
        except Exception as e:
            print(f"Error loading {file_path}: {str(e)}")
    
    return None

def load_vinge_file(file_path):
    """Load and process a VINGE file."""
    try:
        # First try reading with all columns
        df = pd.read_csv(file_path,
                        delimiter='\t',
                        encoding='latin1',
                        decimal=',',
                        quotechar='"',
                        on_bad_lines='skip')
        
        # If that fails, try reading only the needed columns
        if df is None or df.empty:
            df = pd.read_csv(file_path,
                           delimiter='\t',
                           encoding='latin1',
                           decimal=',',
                           quotechar='"',
                           usecols=['Date', 'W.L [cm]'],
                           on_bad_lines='skip')

        # Try multiple date formats
        date_formats = [
            '%d.%m.%Y %H:%M',  # Standard format
            '%Y-%m-%d %H:%M:%S',  # ISO format
            '%d-%m-%Y %H:%M',  # Alternative format
            '%d/%m/%Y %H:%M'   # Another common format
        ]

        for date_format in date_formats:
            try:
                df['Date'] = pd.to_datetime(df['Date'], format=date_format)
                break  # If successful, exit the loop
            except:
                continue
        
        # If none of the specific formats worked, let pandas try to guess
        if not pd.api.types.is_datetime64_any_dtype(df['Date']):
            try:
                df['Date'] = pd.to_datetime(df['Date'])
            except:
                print(f"Error: Failed to parse dates in {file_path}")
                return None
        
        # Convert water level to mm and filter years
        df['W.L [cm]'] = pd.to_numeric(df['W.L [cm]'], errors='coerce') * 10  # Convert to mm
        df = df[df['Date'].dt.year >= 1990]
        
        # Drop any rows with NaN values
        df = df.dropna(subset=['Date', 'W.L [cm]'])
        
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
    raw_data = load_vst_file(file_path)
    
    # Ensure consistent time intervals
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
    """Load rainfall data for a specific measuring station by finding its closest rain station.
    
    Args:
        data_dir: Path to the data directory
        station_id: ID of the measuring station (Station_of_Interest)
        
    Returns:
        DataFrame with rainfall data from the closest rain station, or None if not found
    """
    current_dir = Path(__file__).parent
    rain_data_dir = current_dir / 'Rain_data'
    
    # First, read the mapping file to find the clos