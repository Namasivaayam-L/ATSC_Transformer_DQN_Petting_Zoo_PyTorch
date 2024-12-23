# https://gemini.google.com/app/894e2be26d6e4b0b

import numpy as np

# Fixed bin edges (calculated once)
bin_edges = np.array([0, 10, 20, 30, 40, 50])  # 5 bins

data1 = [5, 12, 25, 35, 42]
hist1, _ = np.histogram(data1, bins=bin_edges)
print("Histogram 1:", hist1, "Length:", len(hist1))  # Output: [1 1 1 1 1] Length: 5

data2 = [2, 8, 15, 22, 28, 33, 38, 45, 48]
hist2, _ = np.histogram(data2, bins=bin_edges)
print("Histogram 2:", hist2, "Length:", len(hist2))  # Output: [2 2 2 2 2] Length: 5

data3 = []  # Empty data
hist3, _ = np.histogram(data3, bins=bin_edges)
print("Histogram 3 (Empty):", hist3, "Length:", len(hist3)) # Output: [0 0 0 0 0] Length: 5

import numpy as np

def create_waiting_time_bins(lane_wt, bin_edges):
    """Creates bins for waiting times.

    Args:
        lane_wt (dict): Lane waiting times.
        bin_edges (list/np.array): Fixed bin edges.

    Returns:
        dict: Binned waiting times.
        Raises ValueError if bin_edges is None or empty.
    """

    if bin_edges is None or not bin_edges:
        raise ValueError("bin_edges must be provided and not empty.")
    
    bin_edges = np.array(bin_edges) #Convert to numpy array for efficiency
    num_bins = len(bin_edges) - 1 #Number of bins is always len(edges) - 1

    binned_lane_wt = {}
    for lane_name, waiting_times in lane_wt.items():
        if not waiting_times:
            binned_lane_wt[lane_name] = [0] * num_bins # Correct way to pad
            continue
        binned_wt, _ = np.histogram(waiting_times, bins=bin_edges)
        binned_lane_wt[lane_name] = binned_wt.tolist()

        # Sanity Check (for extra safety during development):
        if len(binned_lane_wt[lane_name]) != num_bins:
            raise ValueError(f"Bin length mismatch for lane {lane_name}. Expected {num_bins}, got {len(binned_lane_wt[lane_name])}. Check bin_edges.")

    return binned_lane_wt

# Example usage (correct way):
lane_wt = {
    'lane1': [12, 40, 32],
    'lane2': [5, 10, 8],
    'lane3': [],
    'lane4': [100, 110]
}

# Calculate bin edges ONCE
all_waiting_times = [wt for lane in lane_wt.values() for wt in lane] #Flatten the list
min_wt = min(all_waiting_times) if all_waiting_times else 0
max_wt = max(all_waiting_times) if all_waiting_times else 100 # Default max if no data
num_bins = 5
bin_edges = np.linspace(min_wt, max_wt, num_bins + 1)
print("Bin Edges:", bin_edges)

binned_data = create_waiting_time_bins(lane_wt, bin_edges)
print("Binned Data:", binned_data)

# Example to show error if bin_edges is not provided
try:
    create_waiting_time_bins(lane_wt, None)
except ValueError as e:
    print(e)

# Example to show error if bin length is not consistent
try:
    lane_wt_new = {'lane1':[1,2,3,4,5,6,7,8,9,10]}
    new_bin_edges = np.array([0,1,2,3,4])
    create_waiting_time_bins(lane_wt_new, new_bin_edges)
except ValueError as e:
    print(e)