# General imports
import numpy as np
from tqdm import tqdm

# Torch imports
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Telecommunication imports
import DeepMIMOv3
from thesis.scenario_props import *

import warnings
warnings.filterwarnings('ignore')

# Configuration Constants
DEFAULT_NUM_UE_ANTENNAS = 1
DEFAULT_SUBCARRIER_SPACING = 30e3  # Hz
DEFAULT_BS_ROTATION = np.array([0, 0, -135])  # (x, y, z) degrees
DEFAULT_NUM_PATHS = 20

class DeepMIMOGenerator:
    """
    Represents a DeepMIMO dataset generator

    Attributes:
        scenario_name (str): The scenario name used in the dataset
        bs_idx (int): The index of the active base station
        scenario_folder (str): Path to to the directory with the scenarios
        params (dict): A dict with DeepMIMO parameters
        all_channels (np.array): Array with all channels in the scenario [K, N, SC]
        num_total_users (int): Total number of users in the dataset [K]
        user_gains (np.array): Gains of all users [K,]
        h_spatial (np.array): Normalized center carrier of each user [K, N]
    """
    def __init__(self,
                 scenario_name: str = 'city_6_miami',
                 bs_idx: int = 1,
                 scenario_folder: str = "./scenarios"):
        """
        Initialize the DeepMIMO dataset generator.
        Generates the channels, gains and normalized center carrier.
        
        Inputs:
            scenario_name (str): The name of the selected scenario
            bs_idx (int): The index of the active base station
            scenario_folder (str): Path to to the directory with the scenarios
        """
        self.scenario_name = scenario_name
        self.bs_idx = bs_idx
        self.scenario_folder = scenario_folder
        self.params, self.n_subcarriers, self.n_ant_bs = self.get_parameters(scenario_name, bs_idx, scenario_folder)

        self.all_channels, self.num_total_users = self.get_channels()
        self.user_gains, self.h_spatial = self.get_matrices()

    def get_parameters(self,
                       scenario: str,
                       bs_idx: int = 1,
                       scenario_folder: str = "./scenarios") -> dict:
        """
        Constructs the parameter dictionary for DeepMIMOv3 data generation.

        Inputs:
            scenario (str): The name of the scenario (e.g., 'city_6_miami_v1').
            bs_idx (int): The index of the active base station.

        Outputs:
            parameters (dict): A dictionary of parameters compatible with DeepMIMOv3.generate_data.
        """
        # 1. Retrieves scenario-specific properties (e.g., antenna counts)
        scenario_configs = scenario_prop()
        
        # 2. Start with default DeepMIMO parameters
        parameters = DeepMIMOv3.default_params()

        # 3. Basic configuration
        parameters['dataset_folder'] = scenario_folder
        # Assumes scenario format is 'name_vX' and extracts the base name
        parameters['scenario'] = scenario.split("_v")[0]
        parameters['active_BS'] = np.array([bs_idx])
        parameters['enable_BS2BS'] = False
        parameters['num_paths'] = DEFAULT_NUM_PATHS

        # 3. Scenario-specific configuration
        n_ant_bs = scenario_configs[scenario]['n_ant_bs']
        n_subcarriers = scenario_configs[scenario]['n_subcarriers']
        user_rows_config = scenario_configs[scenario]['n_rows']

        if isinstance(user_rows_config, int):
            parameters['user_rows'] = np.arange(user_rows_config)
        else: # Assumes a tuple or list [start, end]
            parameters['user_rows'] = np.arange(user_rows_config[0], user_rows_config[1])

        # 4. BS antenna configuration
        parameters['bs_antenna']['shape'] = np.array([n_ant_bs, 1])  # [Horizontal, Vertical]
        parameters['bs_antenna']['rotation'] = DEFAULT_BS_ROTATION

        # 5. UE antenna configuration
        parameters['ue_antenna']['shape'] = np.array([DEFAULT_NUM_UE_ANTENNAS, 1])

        # 6. OFDM configuration
        parameters['OFDM']['subcarriers'] = n_subcarriers
        parameters['OFDM']['selected_subcarriers'] = np.arange(n_subcarriers)
        parameters['OFDM']['bandwidth'] = 0.02 # 20 MHz for 16 or 32 carriers # 100 MHz for 64 or 128 carriers# (DEFAULT_SUBCARRIER_SPACING * n_subcarriers) / 1e9  # GHz

        return parameters, n_subcarriers, n_ant_bs
    
    def get_channels(self) -> tuple[np.ndarray, int]:
        """
        Generates the all channels using DeepMIMO.
        Removes users without a path to the BS.
        Removes the user antenna dimension, since all scenarios are generated with only one antenna in the UE.

        Inputs:

        Outputs:
            all_channels (np.ndarray): Array with all channels in the scenario [K, N, SC]
            num_total_users (int): Total number of users in the dataset [K]
        """
        # 1. Users DeepMIMO to generated the data
        deepmimo_data = DeepMIMOv3.generate_data(self.params)

        # 2. Removes users without a path to the BS
        # idxs = np.where(deepmimo_data[0]['user']['LoS'] != -1)[0]
        idxs = np.where(deepmimo_data[0]['user']['LoS'] == 1)[0]
        cleaned_deepmimo_data = deepmimo_data[0]['user']['channel'][idxs]
        # 3. Removes the UE antenna dimension
        all_channels = cleaned_deepmimo_data.squeeze()

        num_total_users = all_channels.shape[0]

        return all_channels, num_total_users
    
    def get_matrices(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Generates the gains and spatial signatures matrices
        
        Inputs:

        Outputs:
            user_gains (np.ndarray): Gains of all users [K,]
            h_spatial (np.ndarray): Normalized center carrier of each user [K, N]
        """
        # 1. Gains (Average Power over subcarriers)
        user_gains = np.linalg.norm(self.all_channels, axis=(1, 2))**2 / self.n_subcarriers

        # 2. Spatial Signatures (Normalized Center Subcarrier)
        mid_sub = self.n_subcarriers // 2
        h_spatial_raw = self.all_channels[:, :, mid_sub]
        norms = np.linalg.norm(h_spatial_raw, axis=1, keepdims=True)
        h_spatial = h_spatial_raw / (norms + 1e-9)

        return user_gains, h_spatial

    def get_valid_mask_for_user(self,
                                target_user_idx: int,
                                min_corr: float,
                                max_corr: float,
                                max_gain_ratio: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Generates a mask with the valid users based on gain and correlation conditions

        Inputs:
            target_iser_idx (int): The index of the compared user
            min_corr (float): Minimum correlation between users
            max_corr (float): Maximum correlation beween users
            max_gain_ratio (float): Maximum gain ratio between users

        Outputs:
            mask_corr (np.ndarray): The mask of users with valid correlation [K,]
            mask_gain (np.ndarray): The mask of users with valid gain [K,]
        """
        # 1. Vectorized Correlation
        target_vec = self.h_spatial[target_user_idx]
        corrs = np.abs(self.h_spatial @ target_vec.conj())
        mask_corr = (corrs >= min_corr) & (corrs <= max_corr)

        # 2. Vectorized Gain Ratio 
        target_gain = self.user_gains[target_user_idx]
        all_gains = self.user_gains

        g_min = np.minimum(target_gain, all_gains) + 1e-9
        g_max = np.maximum(target_gain, all_gains)
        ratios = g_max / g_min
        mask_gain = ratios <= max_gain_ratio

        return mask_corr & mask_gain
    
    def generate_dataset(self,
                         num_samples: int,
                         num_users: int,
                         min_corr: float = 0.5,
                         max_corr: float = 0.9,
                         max_gain_ratio: float = 20.0) -> tuple[np.ndarray, np.ndarray]:
        """
        Sample users and generate a dataset with minimum and maximum correlation between users
        and a maximum gain ratio between users.

        Inputs:
            num_samples (int): Number of samples in the final dataset
            num_users (int): Number of users in each sample
            min_corr (float): Minimum correlation between users
            max_corr (float): Maximum correlation beween users
            max_gain_ratio (float): Maximum gain ratio between users

        Outputs:
            dataset_H (np.array): The final channel dataset [B, K, N, SC]
            indices (np.array): The indices of selected users [B, K]
        """
        dataset_indices = []
        pbar = tqdm(total=num_samples, desc="Generating Scenarios")

        attempts = 0
        # 1. Keeps running while the number of samples is not achieved
        while len(dataset_indices) < num_samples:
            # 2. Random Start
            current_group = [np.random.randint(0, self.num_total_users - 1)]
            # 3. Init Mask (All users valid except self)
            candidate_mask = np.ones(self.num_total_users, dtype=bool)
            candidate_mask[current_group[0]] = False

            group_failed = False
            # 4. Greedily add users
            for _ in range(num_users - 1):
                last_added = current_group[-1]

                # 5. Update candidates based on compatibility with the newest member
                step_mask = self.get_valid_mask_for_user(last_added, min_corr, max_corr, max_gain_ratio)
                candidate_mask &= step_mask
                # 6. Get valid indices and check if the group is possible
                valid_indices = np.where(candidate_mask)[0]
                if len(valid_indices) == 0:
                    group_failed = True
                    break
                # 7. Randomly selects a new user
                next_user = np.random.choice(valid_indices)
                current_group.append(next_user)
                candidate_mask[next_user] = False
            # 8. Adds the new group of users to dataset
            if not group_failed:
                dataset_indices.append(current_group)
                pbar.update(1)
            else:
                attempts += 1
        # 9. Gets the channels from the indices
        pbar.close()
        indices = np.array(dataset_indices)
        dataset_H = self.all_channels[indices] # Shape: (Samples, Users, Antennas, Subcarriers)

        return dataset_H, indices

class Tokenizer:
    """
    A Tokenizer that generates tokens for the LWM from wireless channels

    Attributes:
        patch_rows (int): The number of rows used in each patch
        patch_cols (int): The number of columns used in each patch
        cls_value (float): The value that represents the CLS token
        scale_factor (int): The scale factor for normalization
    """
    def __init__(self,
                 patch_rows: int,
                 patch_cols: int,
                 cls_value: float = 0.2,
                 scale_factor: int = 1e6):
        """
        Constrcutor of the tokenizer

        Inputs:
            patch_rows (int): The number of rows used in each patch
            patch_cols (int): The number of columns used in each patch
            cls_value (float): The value that represents the CLS token
            scale_factor (int): The scale factor for normalization
        """
        self.patch_rows = patch_rows
        self.patch_cols = patch_cols
        self.cls_value = cls_value
        self.scale_factor = scale_factor

    def patching(self,
                 x_complex: torch.Tensor) -> torch.Tensor:
        """
        Generates patches from the complex channels
        
        Inputs:
            x_complex (torch.Tensor): The complex wireless channel [B, M, N, SC] or [B, N, SC]

        Outputs:
            patches (torch.Tensor): The final patches [B, Patches, Features]
        """
        # Step 1: Dimension check
        # Remove Singleton dimension - Currently, the LWM model uses only one antenna in the UE
        if x_complex.ndim == 4:
            x_complex = x_complex[:, 0, :, :]

        batch_size, n_rows, n_cols = x_complex.shape

        # Step 2: Split into real and imaginary parts and interleave them
        x_real = x_complex.real
        x_imag = x_complex.imag
        x_interleaved = torch.stack((x_real, x_imag), dim=-1).flatten(start_dim=2)

        # 3. Calculate Padding
        current_rows = x_interleaved.shape[1]
        current_cols = x_interleaved.shape[2]
        patch_width_flat = self.patch_cols * 2 # Real+Imag width

        pad_rows = (self.patch_rows - (current_rows % self.patch_rows)) % self.patch_rows
        pad_cols = (patch_width_flat - (current_cols % patch_width_flat)) % patch_width_flat

        # 4. Apply Padding
        if pad_rows > 0 or pad_cols > 0:
            x_interleaved = F.pad(x_interleaved, (0, pad_cols, 0, pad_rows), value=0)

        # 5. Unfold (Create Patches)
        # Shape: (B, n_pr, Width, h)
        patches = x_interleaved.unfold(dimension=1, size=self.patch_rows, step=self.patch_rows)
        # Shape: (B, n_pr, n_pc, h, w)
        patches = patches.unfold(dimension=2, size=patch_width_flat, step=patch_width_flat)

        # 6. Flatten to Sequence
        # Permute to (B, n_pr, n_pc, h, w)
        # We need to keep this permutation logic consistent for folding back
        patches = patches.contiguous()

        # 7. Flatten grid (n_pr, n_pc) into Num_Patches
        patches = patches.flatten(start_dim=1, end_dim=2)

        # 8. Flatten pixels (h, w) into Features
        patches = patches.flatten(start_dim=2)

        return patches
    
    def tokenizing(self,
                   patches: torch.Tensor) -> torch.Tensor:
        """
        Generates tokens used in the LWM from tokens.
        Basically, prepends a CLS token in the beginning of the token sequence

        Inputs:
            patches (torch.Tensor): The patches used to produce tokens [B, Patches, Features]

        Outputs:
            tokens (torch.Tensor): The sequence of tokens [B, Sequence Length, Features]
        """
        batch_size = patches.shape[0]
        features = patches.shape[-1]
        device = patches.device

        # 1. Create CLS token batch
        cls_tokens = torch.full(
            (batch_size, 1, features), 
            self.cls_value, 
            device=device, 
            dtype=patches.dtype
        )

        # 2. Prepend CLS
        return torch.cat([cls_tokens, patches], dim=1)
    
    def __call__(self,
                 x_complex: torch.Tensor) -> torch.Tensor:
        """
        Transform the complex wireless channel into a sequence of tokens and multiplies for normalization.
        
        Inputs:
            x_complex (torch.Tensor): The complex wireless channel [B, K, N, SC] or [B, N, SC]

        Outputs:
            tokens (torch.Tensor): The sequence of tokens [B, K, Sequence Length, Features] or [B, Sequence Length, Features]
        """
        input_ndim = x_complex.ndim 
        x_complex = x_complex * self.scale_factor # Scale factor for LWM
        # 1. Handle dimensions
        if input_ndim == 4:
            # Case A: (Batch, Users, M, S)
            batch_dim, user_dim, n_rows, n_cols = x_complex.shape
            # Flatten Batch and User together for processing
            # New shape: (B*Users, M, S)
            x_processing = x_complex.reshape(-1, n_rows, n_cols)

        elif input_ndim == 3:
            # Case B: (Batch, M, S)
            x_processing = x_complex 
        else:
            raise ValueError(f"Expected 3D or 4D input, got {x_complex.shape}")
        # 2. Process
        patches = self.patching(x_processing)
        tokens = self.tokenizing(patches)
        # 3. Restore original shape
        if input_ndim == 4:
            # Un-flatten (Batch*Users) -> (Batch, Users)
            # Current: (Batch*Users, Sequence, Dim)
            # New shape: (Batch, Users, Sequence, Dim)
            seq_len = tokens.shape[1]
            token_dim = tokens.shape[2]

            tokens = tokens.view(batch_dim, user_dim, seq_len, token_dim)

        return tokens
