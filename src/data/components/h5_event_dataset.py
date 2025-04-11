import torch
import torch.utils.data
import h5py as h5
import numpy as np
import atexit
from typing import List, Optional

opened_h5_files = []

def close_h5_files():
    """Closes all HDF5 files opened by workers."""
    for f in opened_h5_files:
        try:
            f.close()
        except Exception as e:
            print(f"Warning: Could not close HDF5 file: {e}")
    opened_h5_files.clear()


atexit.register(close_h5_files)


class H5SparseEventDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for loading sparse event data from an HDF5 file.

    Opens the HDF5 file lazily in each worker process using SWMR mode.
    Converts sparse PMT hits (ID, time, photons) into a dense
    (NUM_PMT_IDS, NUM_TIME_BINS) tensor for each event.
    """
    def __init__(
        self,
        h5_path: str,
        h5_dataset_name: str = "pmt",
        num_pmt_ids: int = 64,
        time_min: int = -18_000_000,  # Min time in 100 ps units
        time_max: int = 18_000_000,   # Max time in 100 ps units
        time_bin_width: int = 10_000, # Bin width in 100 ps units (1 μs)
    ):
        """
        Initialize a dataset handling sparse event data from an HDF5 file.

        Args:
            h5_path: Path to the HDF5 file
            h5_dataset_name: Name of the dataset inside the HDF5 file
            num_pmt_ids: Number of PMT IDs to consider
            time_min: Minimum time (in 100 ps units)
            time_max: Maximum time (in 100 ps units)
            time_bin_width: Width of time bins (in 100 ps units)
        """
        self.h5_path = h5_path
        self.h5_dataset_name = h5_dataset_name
        self.num_pmt_ids = num_pmt_ids
        self.time_min = time_min
        self.time_max = time_max
        self.time_bin_width = time_bin_width
        self.num_time_bins = (time_max - time_min) // time_bin_width

        self.h5_file = None # Will be opened lazily in __getitem__ per worker
        self.pmt_dataset = None

        # Get the total number of events without loading all data
        with h5.File(self.h5_path, 'r') as f:
            self._len = len(f[self.h5_dataset_name])

    def _open_h5(self):
        """Opens the HDF5 file in SWMR mode for the current worker."""
        try:
            # libver='latest' is recommended for SWMR
            self.h5_file = h5.File(
                self.h5_path, 
                'r', 
                libver='latest', 
                swmr=True
            )
            self.pmt_dataset = self.h5_file[self.h5_dataset_name]
            opened_h5_files.append(self.h5_file) # Register for cleanup
        except Exception as e:
            print(f"Error opening HDF5 file {self.h5_path} in worker: {e}")
            raise

    def __len__(self):
        """Return the total number of events."""
        return self._len

    def __getitem__(self, idx):
        """Loads, processes, and returns a single event as a dense tensor."""
        if self.h5_file is None:
            self._open_h5()

        # Load raw data for the event
        # Using [idx:idx+1] and then [0] is sometimes more robust for h5py datasets
        event_data = self.pmt_dataset[idx:idx+1][0]

        # Extract fields
        # Ensure they are numpy arrays for filtering
        pmt_ids = np.array(event_data["id"], dtype=np.int64)
        pmt_times = np.array(event_data["t"], dtype=np.int64)
        pmt_photons = np.array(event_data["nphotons"], dtype=np.float32)

        # Filter hits outside the desired time range and valid PMT IDs
        valid_time_mask = (pmt_times >= self.time_min) & (pmt_times < self.time_max)
        valid_id_mask = (pmt_ids >= 0) & (pmt_ids < self.num_pmt_ids)
        valid_mask = valid_time_mask & valid_id_mask
        assert valid_mask.all(), f"Invalid hits in event {idx}, {pmt_ids[~valid_mask]}"

        pmt_ids_filt = pmt_ids[valid_mask]
        pmt_times_filt = pmt_times[valid_mask]
        pmt_photons_filt = pmt_photons[valid_mask]

        if len(pmt_ids_filt) == 0:
            # Handle events with no valid hits within the time range
            return torch.zeros(
                (self.num_pmt_ids, self.num_time_bins), 
                dtype=torch.float32
            )

        time_indices = ((pmt_times_filt - self.time_min) // self.time_bin_width).astype(
            np.int64
        )

        # [[row_indices], [col_indices]]
        coords = torch.tensor(
            np.stack([pmt_ids_filt, time_indices]), 
            dtype=torch.int64
        )
        values = torch.tensor(pmt_photons_filt, dtype=torch.float32)

        sparse_event_tensor = torch.sparse_coo_tensor(
            indices=coords,
            values=values,
            size=(self.num_pmt_ids, self.num_time_bins),
            dtype=torch.float32,
        )
        dense_event_tensor = sparse_event_tensor.to_dense()

        return dense_event_tensor[..., :16_000]