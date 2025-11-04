"""
Dataset module for loading and preprocessing event logs.
"""
import pandas as pd
import pm4py
from pm4py.objects.conversion.log import converter as log_converter


class DataSet:
    """Handles loading and preprocessing of event log data."""
    
    def __init__(self, xes_file_path):
        """
        Initialize dataset loader.
        
        Args:
            xes_file_path (str): Path to XES event log file
        """
        self.xes_file_path = xes_file_path
        self.log = None
        self.df = None

    def import_files(self):
        """
        Import event log from XES file.
        
        Returns:
            EventLog: PM4Py event log object
        """
        self.log = pm4py.read_xes(self.xes_file_path)
        print(f"Log imported with {len(self.log)} traces.")
        return self.log

    def convert_to_df(self):
        """
        Convert event log to pandas DataFrame with standardized column names.
        
        Returns:
            pd.DataFrame: Preprocessed event log DataFrame
        """
        if self.log is None:
            self.log = self.import_files()
            
        # Convert to DataFrame
        df = log_converter.apply(self.log, variant=log_converter.Variants.TO_DATA_FRAME)
        
        # Standardize column names
        col_map = {
            'case:concept:name': 'case_id',
            'concept:name': 'activity',
            'org:resource': 'resource',
            'time:timestamp': 'timestamp'
        }
        
        for old, new in col_map.items():
            if old in df.columns:
                df = df.rename(columns={old: new})
        
        # Check required columns
        required = ['case_id', 'activity', 'resource']
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Select and clean data
        use_cols = ['case_id', 'activity', 'resource', 'timestamp'] if 'timestamp' in df.columns else ['case_id', 'activity', 'resource']
        df = df[use_cols].dropna(subset=['case_id', 'activity', 'resource']).copy()
        
        self.df = df
        print(f"DataFrame created with {len(df)} events")
        return df