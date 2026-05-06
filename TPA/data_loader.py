# data_loader.py
import numpy as np
import glob
import re
import os
from config import COLUMNS, FILE_PATTERN

def extract_temperature(filename):
    """Extracts the integer temperature from the filename."""
    match = re.search(r"Broadened(\d+)C", os.path.basename(filename))
    return int(match.group(1)) if match else None

def load_spectra_data(data_dir="data"):
    """
    Finds all matching files and averages the PMT counts that occur 
    at the exact same wavelength within each file.
    """
    search_path = os.path.join(data_dir, FILE_PATTERN)
    files = glob.glob(search_path)
    
    if not files:
        print(f"Warning: No files found matching {search_path}")
        return {}

    dataset = {}
    
    for filepath in files:
        temp = extract_temperature(filepath)
        if temp is None:
            continue
            
        # 1. Load raw data
        raw_data = np.loadtxt(filepath)
        wavelength = raw_data[:, COLUMNS["wavelength"]]
        pmt = raw_data[:, COLUMNS["pmt"]]
        
        # 2. Clean out zero-counts before averaging so they don't drag the mean down
        mask = pmt > 0
        wavelength = wavelength[mask]
        pmt = pmt[mask]
        
        # 3. Round wavelengths slightly to handle floating-point jitter
        # (e.g., treats 822.123451 and 822.123452 as the same point)
        rounded_wl = np.round(wavelength, decimals=5)
        
        # 4. Find the unique wavelengths and their grouping indices
        unique_wl, inverse_indices = np.unique(rounded_wl, return_inverse=True)
        
        # 5. Sum the PMT counts for each unique wavelength, then divide by how many there were
        sum_pmt = np.bincount(inverse_indices, weights=pmt)
        counts_per_wl = np.bincount(inverse_indices)
        avg_pmt = sum_pmt / counts_per_wl
        
        dataset[temp] = {
            "wavelength": unique_wl,
            "pmt": avg_pmt
        }
        
    return dataset