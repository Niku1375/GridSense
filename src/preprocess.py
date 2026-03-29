import pandas as pd
import glob
import os
from sklearn.preprocessing import LabelEncoder
from utils import save_artifact

def load_all_data(data_folder="data"):
    """Reads both CSV and Parquet files, extracts Region, and combines them."""
    csv_files = glob.glob(f"{data_folder}/*_hourly.csv")
    parquet_files = glob.glob(f"{data_folder}/*_hourly.parquet")
    all_files = csv_files + parquet_files
    
    if not all_files:
        raise FileNotFoundError(f"No data files found in {data_folder}/")
        
    df_list = []
    for file in all_files:
        region_name = os.path.basename(file).split('_')[0]
        
        # Smart loading based on file extension
        if file.endswith('.csv'):
            df = pd.read_csv(file)
        elif file.endswith('.parquet'):
            df = pd.read_parquet(file)
            
        # Standardize columns (Assuming col 0 is Datetime, col 1 is Demand)
        target_col_name = df.columns[1] 
        df = df.rename(columns={target_col_name: 'Demand_MW', df.columns[0]: 'Datetime'})
        
        df['Region'] = region_name
        df_list.append(df)
        
    return pd.concat(df_list, ignore_index=True)

def engineer_features(df, is_training=True):
    """Creates time and categorical features."""
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    if is_training:
        df = df.sort_values('Datetime') # Crucial for time-series!
    
    df['hour'] = df['Datetime'].dt.hour
    df['dayofweek'] = df['Datetime'].dt.dayofweek
    df['month'] = df['Datetime'].dt.month
    
    # Handle the Text-to-Number Region Encoding
    if is_training:
        le = LabelEncoder()
        df['Region_Code'] = le.fit_transform(df['Region'])
        save_artifact(le, 'artifacts/region_encoder.pkl')
    else:
        from utils import load_artifact
        le = load_artifact('artifacts/region_encoder.pkl')
        df['Region_Code'] = le.transform(df['Region'])
        
    return df