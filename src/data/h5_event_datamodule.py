from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import os

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split

from .components.h5_event_dataset import H5SparseEventDataset


class H5EventDataModule(LightningDataModule):
    """LightningDataModule for H5 event data.

    This DataModule handles sparse event data stored in H5 files, where each event contains
    PMT hits (ID, time, photons). It converts these sparse representations into dense
    2D tensors (PMT_ID × Time) for training neural networks.

    A `LightningDataModule` implements 7 key methods:
    - prepare_data (optional)
    - setup
    - train_dataloader
    - val_dataloader
    - test_dataloader
    - predict_dataloader
    - teardown (optional)
    """

    def __init__(
        self,
        data_dir: str = "../data/FM/",
        train_file: str = "train/kazu_broken.h5",
        val_file: Optional[str] = None,
        test_file: str = "test/kazu_broken.h5",
        h5_dataset_name: str = "pmt",
        num_pmt_ids: int = 128,
        time_min: int = -18_000_000,  # in 100 ps units
        time_max: int = 18_000_000,   # in 100 ps units
        time_bin_width: int = 10_000, # in 100 ps units (1 μs)
        train_val_split: Tuple[float, float] = (0.8, 0.2), # Only used if val_file is None
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
    ) -> None:
        """Initialize the H5EventDataModule.

        Args:
            data_dir: The base directory containing H5 data files
            train_file: The file path relative to data_dir for training
            val_file: Optional file path relative to data_dir for validation
                     (if None, a portion of the training data will be used)
            test_file: The file path relative to data_dir for testing
            h5_dataset_name: The dataset name inside the H5 files
            num_pmt_ids: Number of PMT IDs to consider
            time_min: Minimum time in 100 ps units
            time_max: Maximum time in 100 ps units
            time_bin_width: Width of time bins in 100 ps units
            train_val_split: Proportion of training data to use for training and validation
                            (only used if val_file is None)
            batch_size: The batch size
            num_workers: The number of workers for dataloaders
            pin_memory: Whether to pin memory (beneficial for GPU training)
        """
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.train_file_path = os.path.join(data_dir, train_file)
        self.test_file_path = os.path.join(data_dir, test_file)
        self.val_file_path = os.path.join(data_dir, val_file) if val_file else None

        self.data_train = None
        self.data_val = None
        self.data_test = None

        self.batch_size_per_device = batch_size

    def prepare_data(self) -> None:
        """Verify file existence. 
        
        This method is called only from a single process, so it is safe
        to verify file existence and other setup operations.
        """
        # Check that files exist
        if not os.path.exists(self.train_file_path):
            raise FileNotFoundError(f"Training file not found: {self.train_file_path}")
        
        if self.val_file_path and not os.path.exists(self.val_file_path):
            raise FileNotFoundError(f"Validation file not found: {self.val_file_path}")
            
        if not os.path.exists(self.test_file_path):
            raise FileNotFoundError(f"Test file not found: {self.test_file_path}")

    def setup(self, stage: Optional[str] = None) -> None:
        """Set up datasets for training, validation, and testing.
        
        Args:
            stage: The current stage ('fit', 'validate', 'test', or 'predict')
        """
        # Calculate the batch size per device
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by "
                    f"the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        # Only set up datasets that are needed for the current stage
        if stage == "fit" or stage is None:
            if self.data_train is None:
                # Create training dataset
                self.data_train = H5SparseEventDataset(
                    h5_path=self.train_file_path,
                    h5_dataset_name=self.hparams.h5_dataset_name,
                    num_pmt_ids=self.hparams.num_pmt_ids,
                    time_min=self.hparams.time_min,
                    time_max=self.hparams.time_max,
                    time_bin_width=self.hparams.time_bin_width,
                )
                
                # Either use separate validation file or split training data
                if self.val_file_path:
                    self.data_val = H5SparseEventDataset(
                        h5_path=self.val_file_path,
                        h5_dataset_name=self.hparams.h5_dataset_name,
                        num_pmt_ids=self.hparams.num_pmt_ids,
                        time_min=self.hparams.time_min,
                        time_max=self.hparams.time_max,
                        time_bin_width=self.hparams.time_bin_width,
                    )
                else:
                    # Split training data for validation
                    train_length = int(len(self.data_train) * self.hparams.train_val_split[0])
                    val_length = len(self.data_train) - train_length
                    
                    self.data_train, self.data_val = random_split(
                        dataset=self.data_train,
                        lengths=[train_length, val_length],
                        generator=torch.Generator().manual_seed(42),
                    )

        if stage == "test" or stage is None:
            if self.data_test is None:
                # Create test dataset
                self.data_test = H5SparseEventDataset(
                    h5_path=self.test_file_path,
                    h5_dataset_name=self.hparams.h5_dataset_name,
                    num_pmt_ids=self.hparams.num_pmt_ids,
                    time_min=self.hparams.time_min,
                    time_max=self.hparams.time_max,
                    time_bin_width=self.hparams.time_bin_width,
                )

    def train_dataloader(self) -> DataLoader:
        """Create and return the training dataloader.
        
        Returns:
            DataLoader for training
        """
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            persistent_workers=(self.hparams.num_workers > 0),
        )

    def val_dataloader(self) -> DataLoader:
        """Create and return the validation dataloader.
        
        Returns:
            DataLoader for validation
        """
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            persistent_workers=(self.hparams.num_workers > 0),
        )

    def test_dataloader(self) -> DataLoader:
        """Create and return the test dataloader.
        
        Returns:
            DataLoader for testing
        """
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            persistent_workers=(self.hparams.num_workers > 0),
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        """Clean up after each stage.
        
        This method will be called from all processes when the stage ends.
        
        Args:
            stage: The stage being torn down
        """
        # The atexit handler will close the H5 files
        pass 