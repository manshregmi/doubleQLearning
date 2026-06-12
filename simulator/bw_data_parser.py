import numpy as np
import pandas as pd
from bisect import bisect_left
from typing import List, Tuple, Optional

class BandwidthTracker:
    def __init__(self, bandwidth_data: List[Tuple[float, float]]):
        """
        Args:
            bandwidth_data: List of (timestamp_seconds, bandwidth_mbps) tuples
        """
        # Filter out entries with None/empty bandwidth
        valid_data = [(t, bw) for t, bw in bandwidth_data if bw is not None and not pd.isna(bw)]
        
        if not valid_data:
            raise ValueError("No valid bandwidth data provided")
        
        # Sort by timestamp
        valid_data.sort(key=lambda x: x[0])
        
        self.timestamps = np.array([t[0] for t in valid_data])
        self.bandwidths = np.array([bw for bw in valid_data])
        
        # Normalize timestamps to start from 0 for easier use
        self.min_timestamp = self.timestamps[0]
        self.normalized_timestamps = self.timestamps - self.min_timestamp
        
        print(f"Loaded {len(self.timestamps)} bandwidth samples")
        print(f"Time range: {self.timestamps[0]:.2f} to {self.timestamps[-1]:.2f} seconds")
        print(f"Bandwidth range: {self.bandwidths.min():.1f} - {self.bandwidths.max():.1f} Mbps")
        
    def get_bandwidth_at_time(self, time_seconds: float, use_normalized: bool = False) -> float:
        """
        Get bandwidth at specific time using linear interpolation
        
        Args:
            time_seconds: Current simulation time in seconds
            use_normalized: If True, time_seconds is already normalized to start at 0
            
        Returns:
            Interpolated bandwidth in Mbps
        """
        # Convert to normalized time if needed
        if not use_normalized:
            query_time = time_seconds - self.min_timestamp
        else:
            query_time = time_seconds
        
        # Handle edge cases
        if query_time <= self.normalized_timestamps[0]:
            return self.bandwidths[0]
        if query_time >= self.normalized_timestamps[-1]:
            return self.bandwidths[-1]
        
        # Find the interval containing the time
        idx = bisect_left(self.normalized_timestamps, query_time)
        
        # Get surrounding timestamps and bandwidths
        t0, t1 = self.normalized_timestamps[idx-1], self.normalized_timestamps[idx]
        b0, b1 = self.bandwidths[idx-1], self.bandwidths[idx]
        
        # Linear interpolation
        ratio = (query_time - t0) / (t1 - t0)
        bandwidth = b0 + ratio * (b1 - b0)
        
        return bandwidth

def load_bandwidth_data_from_csv(csv_path: str) -> List[Tuple[float, float]]:
    """
    Load bandwidth data from CSV file
    
    Args:
        csv_path: Path to CSV file
        
    Returns:
        List of (timestamp_seconds, bandwidth_mbps) tuples
    """
    df = pd.read_csv(csv_path)
    
    # Clean data: convert empty strings to NaN, then drop NaN
    df['bandwidth_mbps'] = pd.to_numeric(df['bandwidth_mbps'], errors='coerce')
    df = df.dropna(subset=['bandwidth_mbps'])
    
    # Use the 'timestamp' column (seconds with decimal)
    data = list(zip(df['timestamp'].values, df['bandwidth_mbps'].values))
    
    print(f"Loaded {len(data)} valid bandwidth samples")
    return data