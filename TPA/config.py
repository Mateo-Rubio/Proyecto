# config.py

# Data columns based on your file structure:
# power (mW) | wavelength (nm) | PMT singles (counts) | frequency (THz) | photodiode voltage (V)
COLUMNS = {
    "power": 0, 
    "wavelength": 1, 
    "pmt": 2, 
    "frequency": 3, 
    "voltage": 4
}

FILE_PATTERN = "DopplerBroadened*C-822nm-3.5V-10mHz-500ms.txt"