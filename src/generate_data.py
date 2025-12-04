from scenario_props import *
import numpy as np
import DeepMIMOv3
import torch
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')

# --- Configuration Constants ---
DEFAULT_NUM_UE_ANTENNAS = 1
DEFAULT_SUBCARRIER_SPACING = 30e3  # Hz
DEFAULT_BS_ROTATION = np.array([0, 0, -135])  # (x, y, z) degrees
DEFAULT_NUM_PATHS = 20
DATASET_FOLDER = './scenarios'

# Define a constant for the scaling factor
CHANNEL_SCALING_FACTOR = 1e6

def patch_maker(original_ch, patch_rows, patch_cols):
    n_samples, _, n_rows, n_cols = original_ch.shape

    # Setp 1: Remove Singleton dimension - Currently, the LWM model uses only one antenna in the UE
    original_ch = original_ch[:, 0]

    # Step 2: Split into real and imaginary parts and interleave them
    flat_real = original_ch.real
    flat_imag = original_ch.imag

    # Interleave real and imaginary parts along the last axis
    interleaved = np.empty((n_samples, n_rows, n_cols * 2), dtype=np.float32)
    interleaved[:, :, 0::2] = flat_real
    interleaved[:, :, 1::2] = flat_imag

    # Step 3: Compute the number of patches along rows and columns
    n_patches_rows = int(np.ceil(n_rows / patch_rows))
    n_patches_cols = int(np.ceil(n_cols / patch_cols))

    # Step 4: Pad the matrix if necessary to make it divisible by patch size
    padded_rows = n_patches_rows * patch_rows - n_rows
    padded_cols = n_patches_cols * patch_cols - n_cols
    if padded_rows > 0 or padded_cols > 0:
        interleaved = np.pad(
            interleaved,
            ((0, 0), (0, padded_rows), (0, padded_cols * 2)),  # Double padding for interleaved axis
            mode='constant',
            constant_values=0,
        )

    # Step 5: Create patches by dividing into blocks
    n_samples, padded_rows, padded_cols = interleaved.shape
    padded_cols //= 2  # Adjust for interleaving (real and imaginary parts count as one)
    patches = []

    for i in range(0, padded_rows, patch_rows):
        for j in range(0, padded_cols, patch_cols):
            patch = interleaved[:, i:i + patch_rows, j * 2:(j + patch_cols) * 2]
            patches.append(patch.reshape(n_samples, -1))  # Flatten each patch

    # Step 6: Stack patches to form the final array
    patches = np.stack(patches, axis=1)  # Shape: (num_samples, n_patches, patch_rows * patch_cols * 2)

    return patches

def tokenizer(patches):

    patches = [patch for patch in patches]

    grouped_data_2 = []

    for user_idx in range(len(patches)):
        patch_size = patches[user_idx].shape[1]

        word2id = {
            '[CLS]': 0.2 * np.ones((patch_size))
        }

        tokens = patches[user_idx]
        sample = np.vstack((word2id['[CLS]'], tokens))
        sample = torch.tensor(sample)
        
        grouped_data_2.append(sample)
     
    normalized_grouped_data = torch.stack(grouped_data_2, dim=0)
    
    return normalized_grouped_data

def get_parameters(scenario: str, bs_idx: int = 1) -> dict:
    """Constructs the parameter dictionary for DeepMIMOv3 data generation.

    Args:
        scenario: The name of the scenario (e.g., 'city_6_miami_v1').
        bs_idx: The index of the active base station.

    Returns:
        A dictionary of parameters compatible with DeepMIMOv3.generate_data.
    """
    # Retrieves scenario-specific properties (e.g., antenna counts)
    scenario_configs = scenario_prop()
    
    # Start with default DeepMIMO parameters
    parameters = DeepMIMOv3.default_params()

    # --- Base Configuration ---
    parameters['dataset_folder'] = DATASET_FOLDER
    # Assumes scenario format is 'name_vX' and extracts the base name
    parameters['scenario'] = scenario.split("_v")[0]
    parameters['active_BS'] = np.array([bs_idx])
    parameters['enable_BS2BS'] = False
    parameters['num_paths'] = DEFAULT_NUM_PATHS

    # --- Scenario-Specific Configuration ---
    n_ant_bs = scenario_configs[scenario]['n_ant_bs']
    n_subcarriers = scenario_configs[scenario]['n_subcarriers']
    user_rows_config = scenario_configs[scenario]['n_rows']

    if isinstance(user_rows_config, int):
        parameters['user_rows'] = np.arange(user_rows_config)
    else: # Assumes a tuple or list [start, end]
        parameters['user_rows'] = np.arange(user_rows_config[0], user_rows_config[1])

    # --- Antenna and OFDM Configuration ---
    parameters['bs_antenna']['shape'] = np.array([n_ant_bs, 1])  # [Horizontal, Vertical]
    parameters['bs_antenna']['rotation'] = DEFAULT_BS_ROTATION
    parameters['ue_antenna']['shape'] = np.array([DEFAULT_NUM_UE_ANTENNAS, 1])
    parameters['OFDM']['subcarriers'] = n_subcarriers
    parameters['OFDM']['selected_subcarriers'] = np.arange(n_subcarriers)
    parameters['OFDM']['bandwidth'] = (DEFAULT_SUBCARRIER_SPACING * n_subcarriers) / 1e9  # GHz

    return parameters

def deepmimo_data_cleaning(deepmimo_data):
    """Cleans DeepMIMO data by removing users without a LoS path and scales channel coefficients.

    Args:
        deepmimo_data (dict): The raw data dictionary returned by DeepMIMOv3.generate_data.

    Returns:
        np.ndarray: Cleaned and scaled channel data for users with a valid path.
                    The channel coefficients are multiplied by CHANNEL_SCALING_FACTOR
                    for numerical stability in subsequent processing (e.g., ML models).
    """
    # Identify users with a Line-of-Sight (LoS) path (LoS != -1 indicates a valid path)
    valid_user_indices = np.where(deepmimo_data['user']['LoS'] != -1)[0]

    # Select channel data only for valid users
    cleaned_channels = deepmimo_data['user']['channel'][valid_user_indices]

    # Scale the channel coefficients for numerical stability
    return cleaned_channels * CHANNEL_SCALING_FACTOR

def deepmimo_data_gen(scenario_names: list[str], bs_idxs: list[int] | None = None) -> list[dict]:
    """Generates DeepMIMO channel data for multiple scenarios and base stations.

    Args:
        scenario_names: A list of scenario name strings to generate data for.
        bs_idxs: A list of base station indices to use for each scenario.
                 Defaults to [1, 2, 3] if not provided.

    Returns:
        A list of dictionaries, where each dictionary contains the 'scenario'
        identifier and the corresponding 'channels' data (np.ndarray).
    """
    if bs_idxs is None:
        bs_idxs = [1, 2, 3]

    deepmimo_data = []
    
    # Create a list of all (scenario, bs) pairs to iterate over
    generation_tasks = [(name, idx) for name in scenario_names for idx in bs_idxs]

    # Use tqdm for a user-friendly progress bar
    print(f"Generating data for {len(generation_tasks)} scenario-BS pairs...")
    for scenario_name, bs_idx in tqdm(generation_tasks, desc="Data Generation"):
        parameters = get_parameters(scenario_name, bs_idx)
        # The [0] index selects the user data from the DeepMIMO output
        raw_deepmimo_data = DeepMIMOv3.generate_data(parameters)[0]
        cleaned_channels = deepmimo_data_cleaning(raw_deepmimo_data)
        deepmimo_data.append({"scenario": f"{scenario_name} - BS{bs_idx}", "channels": cleaned_channels})

    return deepmimo_data

def sample(deepmimo_data: list[dict], N_samples: int, n_users: int) -> list[dict]:
    """Generates samples by randomly selecting a scenario and a subset of users' channels.

    Each sample consists of channel data for 'n_users' randomly chosen users
    from a randomly selected scenario.

    Args:
        deepmimo_data: A list of dictionaries, where each dict contains
                       'scenario' (str) and 'channels' (np.ndarray) data.
        N_samples: The total number of samples to generate.
        n_users: The number of users (channels) to select for each sample.

    Returns:
        A list of dictionaries, each representing a sample with 'scenario' and
        'channels' (np.ndarray of shape (n_users, ...)).
    """
    samples = []

    for _ in tqdm(range(N_samples), desc="Sampling"):
        # Randomly select a scenario from the deepmimo_data list
        scenario_idx = np.random.randint(0, len(deepmimo_data))
        selected_scenario_data = deepmimo_data[scenario_idx]
        
        # Randomly select 'n_users' channel indices from the chosen scenario
        num_available_users = selected_scenario_data["channels"].shape[0] # Use shape[0] for number of users
        ue_idxs = np.random.choice(num_available_users, n_users, replace=False) # Use np.random.choice for unique indices
        
        selected_channels = selected_scenario_data["channels"][ue_idxs]

        samples.append({"scenario": selected_scenario_data["scenario"], "channels": selected_channels})

    return samples

def create_dataset(scenario_names: list[str] = None,
                   bs_idxs: list[int] = None,
                   n_samples:int = 100,
                   n_users: int = 4,
                   patch_rows: int = 4,
                   patch_cols: int = 4):
    """

    """
    deepmimo_data = deepmimo_data_gen(scenario_names, bs_idxs)
    dataset = sample(deepmimo_data, n_samples, n_users)

    # 2. Prepare all data for batch inference
    all_tokens = []
    for item in tqdm(dataset, desc="Preparing data"):
        # Prepare input tokens for the LWM model
        patches = patch_maker(item["channels"], patch_rows, patch_cols)
        tokens = tokenizer(patches)
        all_tokens.append(tokens)

        # Reshape channel data for the next stage (the regressor model)
        # Original shape: (K, 1, M, S) -> (4, 1, 16, 32)
        # Squeezed shape: (K, M, S) -> (4, 16, 32)
        # Transposed shape: (S, K, M) -> (32, 4, 16)
        item["channels"] = item["channels"].squeeze().transpose(2, 0, 1)

    # Stack all tokens into a single tensor for efficient processing
    # Shape changes from a list of [N_USERS, ...] tensors to one [N_SAMPLES * N_USERS, ...] tensor
    all_tokens_tensor = torch.cat(all_tokens, dim=0)
    all_channels_array = np.array([d['channels'] for d in dataset]) # (N, S, K, M)

    return all_tokens_tensor, all_channels_array

