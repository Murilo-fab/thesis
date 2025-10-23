from scenario_props import *
import numpy as np
import DeepMIMOv3
from collections import defaultdict
import torch
from tqdm import tqdm
from inference import lwm_inference
from power_allocation import power_allocation

import warnings
warnings.filterwarnings('ignore')

def get_parameters(scenario, bs_idx=1):
    # For default, the model currently uses only one antenna for the UE
    n_ant_ue = 1
    scs = 30e3
    # Retrieves default scenario properties
    row_column_users = scenario_prop()
    # Selected scenario and folder
    parameters = DeepMIMOv3.default_params()
    parameters['dataset_folder'] = './scenarios'
    parameters['scenario'] = scenario.split("_v")[0]

    n_ant_bs = row_column_users[scenario]['n_ant_bs']
    n_subcarriers = row_column_users[scenario]['n_subcarriers']
    parameters['active_BS'] = np.array([bs_idx])
    
    if isinstance(row_column_users[scenario]['n_rows'], int):
        parameters['user_rows'] = np.arange(row_column_users[scenario]['n_rows'])
    else:
        parameters['user_rows'] = np.arange(row_column_users[scenario]['n_rows'][0],
                                            row_column_users[scenario]['n_rows'][1])

    parameters['bs_antenna']['shape'] = np.array([n_ant_bs, 1]) # Horizontal, Vertical 
    parameters['bs_antenna']['rotation'] = np.array([0,0,-135]) # (x,y,z)
    parameters['ue_antenna']['shape'] = np.array([n_ant_ue, 1])
    parameters['enable_BS2BS'] = False
    parameters['OFDM']['subcarriers'] = n_subcarriers
    parameters['OFDM']['selected_subcarriers'] = np.arange(n_subcarriers)
    
    parameters['OFDM']['bandwidth'] = scs * n_subcarriers / 1e9
    parameters['num_paths'] = 20

    return parameters

def deepmimo_data_cleaning(deepmimo_data):
    # Remove users without a path between the TX and RX
    idxs = np.where(deepmimo_data['user']['LoS'] != -1)[0]
    cleaned_deepmimo_data = deepmimo_data['user']['channel'][idxs]
    return np.array(cleaned_deepmimo_data) * 1e6

def deepmimo_data_gen(scenario_names=None, bs_idxs=[1,2,3]):
    deepmimo_data = []
    # For each BS, generates the data
    for scenario_name in scenario_names:
        for bs_idx in bs_idxs:
            parameters = get_parameters(scenario_name, bs_idx)
            deepMIMO_dataset = DeepMIMOv3.generate_data(parameters)[0]
        
            deepmimo_data.append(deepMIMO_dataset)
        # Those properties will be useful during the sampling
    return deepmimo_data

def sample_users(dataset, N):
    # Selectes N random users
    idxs = np.random.randint(0, len(dataset), N)
    return dataset[idxs]

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

def sample(deepmimo_data, N_samples, n_bs_antenna, Ptot=1, sigma2=10e-3):
    channels = []
    labels = np.zeros((N_samples, n_bs_antenna), dtype=float)

    for i in tqdm(range(N_samples), desc="Sampling"):
        scenario_idx = np.random.randint(0, len(deepmimo_data))
        ue_idxs = np.random.randint(0, len(deepmimo_data[scenario_idx]), n_bs_antenna)

        users = deepmimo_data[scenario_idx][ue_idxs]

        V_wmmse, p_alloc = power_allocation(users, Ptot, sigma2)

        channels.append(users)

        labels[i, :] = p_alloc

    return channels, labels

def generate_dataset(model=None, input_types=["cls_emb"], device="cpu", batch_size=32, scenario_names=None, bs_idxs=[1,2,3], N_samples=1000, n_bs_antenna=8, n_rows=4, n_columns=4):
    # Generate data with DeepMIMO for the selected scenario with active BS
    deepmimo_data = deepmimo_data_gen(scenario_names, bs_idxs)
    # Clean the data to exclude users without a path between TX and RX 
    cleaned_deepmimo_data = [deepmimo_data_cleaning(deepmimo_data[scenario_idx]) for scenario_idx in range(len(deepmimo_data))]
    # Choose users
    selected_channels, labels = sample(cleaned_deepmimo_data, N_samples, n_bs_antenna)
    outputs = {}
    for input_type in input_types:
        embeddings = []
        for channels in tqdm(selected_channels, desc=f"Inference - {input_type}"):
            # Create patches
            patch_list = patch_maker(channels, n_rows, n_columns)
            # Create tokens
            tokens = tokenizer(patch_list)
            # Perform embedding
            embeddings.append(lwm_inference(model, tokens, input_type, device, batch_size))
        embeddings = torch.stack(embeddings)
        outputs[input_type] = embeddings
    return torch.from_numpy(np.stack(selected_channels)).float(), outputs, torch.from_numpy(labels).float()


import warnings
warnings.filterwarnings('ignore')

def get_parameters(scenario, bs_idx=1):
    # For default, the model currently uses only one antenna for the UE
    n_ant_ue = 1
    scs = 30e3
    # Retrieves default scenario properties
    row_column_users = scenario_prop()
    # Selected scenario and folder
    parameters = DeepMIMOv3.default_params()
    parameters['dataset_folder'] = './scenarios'
    parameters['scenario'] = scenario.split("_v")[0]

    n_ant_bs = row_column_users[scenario]['n_ant_bs']
    n_subcarriers = row_column_users[scenario]['n_subcarriers']
    parameters['active_BS'] = np.array([bs_idx])
    
    if isinstance(row_column_users[scenario]['n_rows'], int):
        parameters['user_rows'] = np.arange(row_column_users[scenario]['n_rows'])
    else:
        parameters['user_rows'] = np.arange(row_column_users[scenario]['n_rows'][0],
                                            row_column_users[scenario]['n_rows'][1])

    parameters['bs_antenna']['shape'] = np.array([n_ant_bs, 1]) # Horizontal, Vertical 
    parameters['bs_antenna']['rotation'] = np.array([0,0,-135]) # (x,y,z)
    parameters['ue_antenna']['shape'] = np.array([n_ant_ue, 1])
    parameters['enable_BS2BS'] = False
    parameters['OFDM']['subcarriers'] = n_subcarriers
    parameters['OFDM']['selected_subcarriers'] = np.arange(n_subcarriers)
    
    parameters['OFDM']['bandwidth'] = scs * n_subcarriers / 1e9
    parameters['num_paths'] = 20

    return parameters

def deepmimo_data_cleaning(deepmimo_data):
    # Remove users without a path between the TX and RX
    idxs = np.where(deepmimo_data['user']['LoS'] != -1)[0]
    cleaned_deepmimo_data = deepmimo_data['user']['channel'][idxs]
    return np.array(cleaned_deepmimo_data) * 1e6

def deepmimo_data_gen(scenario_names=None, bs_idxs=[1,2,3]):
    deepmimo_data = []
    # For each BS, generates the data
    for scenario_name in scenario_names:
        for bs_idx in bs_idxs:
            parameters = get_parameters(scenario_name, bs_idx)
            deepMIMO_dataset = DeepMIMOv3.generate_data(parameters)[0]
            cleaned_deepmimo_data = deepmimo_data_cleaning(deepMIMO_dataset).squeeze()
            deepmimo_data.append({"scenario": f"{scenario_name} - {bs_idx}", "channels": cleaned_deepmimo_data})
        # Those properties will be useful during the sampling
    return deepmimo_data

def sample(deepmimo_data, N_samples, n_users):
    samples = []

    for i in tqdm(range(N_samples), desc="Sampling"):
        scenario_idx = np.random.randint(0, len(deepmimo_data))
        selected_scenario = deepmimo_data[scenario_idx]["scenario"]

        ue_idxs = np.random.randint(0, len(deepmimo_data[scenario_idx]), n_users)
        selected_users = deepmimo_data[scenario_idx]["channels"][ue_idxs]

        samples.append({"scenario": selected_scenario, "channels": selected_users})

    return samples