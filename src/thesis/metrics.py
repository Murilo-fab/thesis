import pandas as pd
import matplotlib.pyplot as plt
import os
import math
import seaborn as sns
from thesis.utils import get_parameters
import numpy as np
import DeepMIMOv3
import textwrap


# --- Define Theme Colors ---
BG_COLOR = '#FFFFFF'     # White background
BORDER_COLOR = '#404040' # Dark gray for spines, ticks, and text
GRID_COLOR = '#E8E8E8'   # Subtle light gray for the grid

# --- Apply Global Settings via rcParams ---
plt.rcParams.update({
    # Font settings
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],

    # Background Colors
    'figure.facecolor': BG_COLOR,
    'axes.facecolor': BG_COLOR,
    
    # Spines & Ticks
    'axes.edgecolor': BORDER_COLOR,
    'axes.linewidth': 1.2,
    'xtick.color': BORDER_COLOR,
    'ytick.color': BORDER_COLOR,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.major.size': 6,
    'ytick.major.size': 6,
    
    # Grid Formatting
    'axes.grid': True,
    'grid.color': GRID_COLOR,
    'grid.linestyle': '--',
    'grid.linewidth': 0.8,
    
    # Labels
    'axes.labelcolor': BORDER_COLOR,
    'text.color': BORDER_COLOR,
    'axes.labelsize': 13,
    
    # Legend
    'legend.facecolor': BG_COLOR,
    'legend.edgecolor': BORDER_COLOR,
    'legend.framealpha': 0.25,
    'legend.fontsize': 8,

    # Lines
    'lines.linestyle': '-',
    'lines.linewidth': 1.5,
    'lines.markersize': 5,
    'lines.markerfacecolor': '#F8F9FA',
    'lines.markeredgewidth': 1.5,
})

def plot_data_efficiency_results(results_path,
                                 selected_models,
                                 selected_train_ratio,
                                 style_dict):
    """
    """
    # Output file name
    base = os.path.splitext(results_path)[0]
    output_file_name = base + ".pdf"

    # Load and filter CSV data
    df = pd.read_csv(results_path)
    filtered_df = df[(df['Model_Name'].isin(selected_models))&
                     (df['Train_Ratio'].isin(selected_train_ratio))].copy()
    filtered_df['Train_Ratio_PCT'] = filtered_df['Train_Ratio'] * 100

    # Plot figure
    plt.figure(figsize=(6.5, 4))

    for model, (hex_color, marker, z) in style_dict.items():
        if model in filtered_df['Model_Name'].values:
            model_data = filtered_df[filtered_df['Model_Name'] == model]
                
            plt.plot(
                model_data['Train_Ratio_PCT'], 
                model_data['F1_Score'], 
                color=hex_color,
                marker=marker, 
                markeredgecolor=hex_color,
                label=model,
                zorder=z
            )

    # Axis Formatting
    plt.xscale('log')
    plt.xticks([ 1, 10, 100], ['1%', '10%', '100%'])

    plt.xlabel('Number of training samples (%)', fontsize=18)
    plt.ylabel('Weighted F1-Score', fontsize=18)

    # Legend Formatting
    plt.legend(loc='lower right', fontsize=11)

    plt.tight_layout()
    plt.savefig(output_file_name)
    plt.show()


def plot_snr_robustness(results_path, 
                        selected_models, 
                        target_train_ratio, 
                        selected_snr,
                        style_dict):
    """
    """
    # Output file name
    base = os.path.splitext(results_path)[0]
    output_file_name = base + ".pdf"

    # Load and filter CSV data
    df = pd.read_csv(results_path)
    
    filtered_df = df[
        (df['Model_Name'].isin(selected_models)) & 
        (df['Train_Ratio'] == target_train_ratio) &
        (df['SNR_dB'].isin(selected_snr))
    ].copy()

    # Plot figure
    plt.figure(figsize=(6.5, 4))

    for model, (hex_color, marker, z) in style_dict.items():
        if model in filtered_df['Model_Name'].values:
            model_data = filtered_df[filtered_df['Model_Name'] == model]
            
            # Sort by SNR
            model_data = model_data.sort_values(by='SNR_dB')
                
            plt.plot(
                model_data['SNR_dB'], 
                model_data['F1_Score'], 
                color=hex_color,
                marker=marker, 
                markeredgecolor=hex_color,   
                label=model,
                zorder=z
            )

    # Axis Formatting
    plt.xlabel('SNR (dB)', fontsize=18)
    plt.ylabel('Weighted F1 score', fontsize=18)
    
    plt.xticks(selected_snr)

    # Legend Formatting
    plt.legend(loc='upper left', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_file_name)
    plt.show()

# Plotting tsne cell helper function
def plot_tsne_cell(ax, df, x_col, y_col, hue_col, palette_map, show_legend=False):
    df_plot = df.copy()
    df_plot[hue_col] = df_plot[hue_col].astype(int)

    sns.scatterplot(
        data=df, 
        x=x_col, 
        y=y_col, 
        hue=hue_col, 
        ax=ax, 
        s=20,
        alpha=1.0,
        edgecolor='none',
        linewidth=0,
        palette=palette_map,
        legend=show_legend  
    )
    
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor('black')
        spine.set_linewidth(1)

    if show_legend:
        ax.legend(loc='lower right', fontsize=10, markerscale=3)

def plot_tsne_grid(user_map_path,
                   tsne_results_path,
                   model_names,
                   complexity_selection=None,
                   max_cols=3):
    """
    """
    # Output file name
    base = os.path.splitext(tsne_results_path)[0]
    output_file_name = base + ".pdf"

    # Load and filter CSV data
    df_user_map = pd.read_csv(user_map_path)
    df_tsne = pd.read_csv(tsne_results_path)

    if complexity_selection:
        if 'n_beams' in df_tsne.columns:
            df_tsne = df_tsne[df_tsne['n_beams']==complexity_selection]

    map_labels = set(df_user_map['label'].unique())
    tsne_labels = set(df_tsne['label'].unique())
    all_unique_labels = sorted(list(map_labels.union(tsne_labels)))
    
    all_unique_labels = [int(lbl) for lbl in all_unique_labels]
    n_colors = len(all_unique_labels)
    
    if n_colors <= 2:
        raw_colors = sns.color_palette("tab10", n_colors)
    else:
        raw_colors = sns.color_palette("tab20", n_colors)
        
    palette_map = dict(zip(all_unique_labels, raw_colors))
    
    # Calculate dynamic layout
    total_plots = len(model_names) + 1  
    cols = min(total_plots, max_cols)   
    rows = math.ceil(total_plots / cols) 
    
    # Setup figure
    fig, axes = plt.subplots(
        nrows=rows, ncols=cols, 
        figsize=(4.0 * cols, 4.0 * rows),
        gridspec_kw={'wspace': 0, 'hspace': 0.5}
    )
    
    if isinstance(axes, plt.Axes):
        axes = [axes]
    else:
        axes = axes.flatten()

    # Plot data and add titles
    
    # Slot 0 (top left): Scenario layout
    plot_tsne_cell(axes[0], df_user_map, 'x', 'y', 'label', palette_map, show_legend=False)
    axes[0].set_title("Scenario layout", fontsize=12, pad=10)
    axes[0].set_aspect('equal', adjustable='datalim')
    
    # t-SNE embeddings
    for i, model in enumerate(model_names):
        ax_idx = i + 1 
        subset = df_tsne[df_tsne['model_name'] == model]

        wrapped_title = "\n".join(textwrap.wrap(model, width=25))   
        plot_tsne_cell(axes[ax_idx], subset, 'tsne_1', 'tsne_2', 'label', palette_map, show_legend=False)
        axes[ax_idx].set_title(wrapped_title, fontsize=14, pad=10)

    # Clean up empty subplots 
    for j in range(total_plots, len(axes)):
        axes[j].axis('off')

    plt.savefig(output_file_name)
    plt.show()

def plot_coverage_map(city_name):
    # Initialize and set parameters
    parameters = get_parameters(city_name)

    # Generate the dataset
    dataset = DeepMIMOv3.generate_data(parameters)

    n_plots = len (dataset)
    n_cols = 2
    n_rows = math.ceil(n_plots/n_cols)

    # Create the figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 6 * n_rows))
    
    # Flatten axes for easy iteration; handle the single-plot case
    if n_plots > 1:
        axes_flat = axes.flatten()
    else:
        axes_flat = [axes]

    for i in range(n_plots):
        ax = axes_flat[i]

        # Extract data for the first BS
        bs_loc = dataset[i]['location']
        user_locs = dataset[i]['user']['location']
        channel = dataset[i]['user']['channel']

        # Calculate received power
        axes_to_sum = tuple(range(1, channel.ndim)) # Dynamically adapts to array dimensions
        power_linear = np.sum(np.abs(channel)**2, axis=axes_to_sum)

        # Convert the linear power to Decibels (dB) for standard visualization
        power_db = 10 * np.log10(power_linear + 1e-12)

        scatter = ax.scatter(user_locs[:, 0], user_locs[:, 1], 
                             c=power_db, cmap='viridis', marker='s', s=60, edgecolors= 'none',
                             alpha=1.0, zorder=2)
        
        # Plot the BS
        plt.scatter(bs_loc[0], bs_loc[1], 
                    c='black', s=150, marker='^', edgecolor='white', linewidths=1.5, 
                    label=f'Base Station (BS) {i+1}', zorder=5)

        # Formatting the plot
        plt.xlabel('X Coordinate (m)', fontsize=12)
        plt.ylabel('Y Coordinate (m)', fontsize=12)
        plt.title(f"Coverage Map: {parameters['scenario']}", fontsize=14)
        plt.grid(True, linestyle='--', color='lightgray', alpha=0.7, zorder=1)
        plt.legend(loc='upper right', framealpha=1.0) 
        plt.axis('equal')

        # Add a colorbar to act as the map's legend for signal strength
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Received Power (dB)', fontsize=12)

    for j in range(i + 1, len(axes_flat)):
        fig.delaxes(axes_flat[j])
    
    plt.savefig(f"../results/{city_name}_coverage_map.pdf")
    plt.show()

def plot_beam_vs_data_heatmaps(results_path,
                               selected_models,
                               max_cols=3):
    """
    """
    # Output file name
    base = os.path.splitext(results_path)[0]
    output_file_name = base + ".pdf"

    # Load and filter CSV data
    df = pd.read_csv(results_path)
    valid_models = [m for m in selected_models if m in df['Model_Name'].unique()]
    filtered_df = df[df['Model_Name'].isin(valid_models)].copy()

    filtered_df['Train_Ratio_Pct'] = filtered_df['Train_Ratio'].apply(lambda x: f"{x * 100:g}")

    # Determine global limits and grid shape
    global_vmin = filtered_df['F1_Score'].min()
    global_vmax = filtered_df['F1_Score'].max()
    global_y_order = [f"{x * 100:g}" for x in sorted(filtered_df['Train_Ratio'].unique())]
    global_x_order = sorted(filtered_df['Num_Beams'].unique())

    # Calculate dynamic layout
    num_models = len(valid_models)
    if num_models == 0:
        print("Error: None of the selected models were found in the CSV.")
        return
    cols = min(num_models, max_cols)   
    rows = math.ceil(num_models / cols) 

    # Plot figure
    fig, axes = plt.subplots(
        nrows=rows, ncols=cols, 
        figsize=(3.5 * cols + 1.5, 4.5 * rows),
        sharey=True, sharex=True
    )
    if isinstance(axes, plt.Axes): axes = [axes]
    else: axes = axes.flatten()

    # 6. Plot each model
    for i in range(len(axes)):
        ax = axes[i]
        
        # Explicitly turn off the default matplotlib grid for this axis
        ax.grid(False) 

        if i < num_models:
            model = valid_models[i]
            subset = filtered_df[filtered_df['Model_Name'] == model]
            pivot_df = subset.pivot(index='Train_Ratio_Pct', columns='Num_Beams', values='F1_Score')
            pivot_df = pivot_df.reindex(index=global_y_order, columns=global_x_order)
            
            sns.heatmap(
                pivot_df, ax=ax, annot=True, fmt=".2f", cmap="coolwarm", 
                vmin=global_vmin, vmax=global_vmax, 
                cbar=(i == num_models - 1), 
                cbar_kws={'label': ''} if i == num_models - 1 else None 
            )
            
            ax.set_title(model, fontsize=12, pad=10, fontweight='bold')
            ax.set_xlabel('')
            if i % cols == 0:
                ax.set_ylabel('Percentage of Training Samples (%)', fontsize=14)
            else: ax.set_ylabel('')
        else:
            ax.axis('off')

    fig.supxlabel('Task Complexity (Number of Beams)', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_file_name)
    plt.show()

def plot_beam_vs_snr_heatmaps(results_path,
                              selected_models,
                              fixed_train_ratio=1.0,
                              selected_snr = [-5, 0, 5, 10, 15, 20, 25],
                              max_cols=3):
    """
    """
    # Output file name
    base = os.path.splitext(results_path)[0]
    output_file_name = base + ".pdf"

    # Load and filter CSV data
    df = pd.read_csv(results_path)
    filtered_df = df[(df['Model_Name'].isin(selected_models))&
                     (df['Train_Ratio']==fixed_train_ratio)&
                     (df['SNR_dB'].isin(selected_snr))].copy()

    # Determine global limits and grid shape ---
    global_vmin = filtered_df['F1_Score'].min()
    global_vmax = filtered_df['F1_Score'].max()
    global_y_order = sorted(filtered_df['SNR_dB'].unique())
    global_x_order = sorted(filtered_df['Num_Beams'].unique())

    # Calculate dynamic layout 
    num_models = len(selected_models)
    if num_models == 0:
        print("Error: None of the selected models were found in the CSV.")
        return
    cols = min(num_models, max_cols)   
    rows = math.ceil(num_models / cols) 

    # Plot figure
    fig, axes = plt.subplots(
        nrows=rows, ncols=cols, 
        figsize=(3.5 * cols + 1.5, 4.5 * rows),
        sharey=True, sharex=True
    )
    if isinstance(axes, plt.Axes): axes = [axes]
    else: axes = axes.flatten()

    # Plot each model
    for i in range(len(axes)):
        ax = axes[i]
        
        ax.grid(False) 

        if i < num_models:
            model = selected_models[i]
            subset = filtered_df[filtered_df['Model_Name'] == model]
            pivot_df = subset.pivot(index='SNR_dB', columns='Num_Beams', values='F1_Score')
            pivot_df = pivot_df.reindex(index=global_y_order, columns=global_x_order)
            
            sns.heatmap(
                pivot_df, ax=ax, annot=True, fmt=".2f", cmap="coolwarm", 
                vmin=global_vmin, vmax=global_vmax, 
                cbar=(i == num_models - 1), 
                cbar_kws={'label': ''} if i == num_models - 1 else None 
            )
            
            ax.set_title(model, fontsize=12, pad=10, fontweight='bold')
            ax.set_xlabel('')
            if i % cols == 0:
                ax.set_ylabel('SNR (dB)', fontsize=14)
            else: ax.set_ylabel('')
        else:
            ax.axis('off')

    fig.supxlabel('Task Complexity (Number of Beams)', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_file_name)
    plt.show()

def show_resources_table(metrics_file,
                         performance_file,
                         complexity_selection=None):
    # Load the datasets
    try:
        df_metrics = pd.read_csv(metrics_file)
        df_perf = pd.read_csv(performance_file)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    table_metrics = df_metrics.copy()

    if complexity_selection:
        if 'Num_Beams' in table_metrics.columns and 'Num_Beams' in df_perf.columns:
            table_metrics = table_metrics[table_metrics['Num_Beams']==complexity_selection]
            df_perf = df_perf[df_perf['Num_Beams']==complexity_selection]
        if 'Num_Users' in table_metrics.columns and 'Num_Users' in df_perf.columns:
            table_metrics = table_metrics[table_metrics['Num_Users']==complexity_selection]
            df_perf = df_perf[df_perf['Num_Users']==complexity_selection]

    table_metrics['Total_Time_ms'] = table_metrics['Encoder_ms'] + table_metrics['Head_ms']

    table_training_time = df_perf.pivot(
        index='Model_Name', 
        columns='Train_Ratio', 
        values='Training_Time'
    )

    model_order = table_metrics['Model_Name'].tolist()
    available_models = [m for m in model_order if m in table_training_time.index]
    table_training_time = table_training_time.reindex(available_models)

    # Display results
    print("--- TABLE 1: MODEL EFFICIENCY METRICS ---")
    print(table_metrics.to_string(index=False))
    
    print("\n" + "="*50 + "\n")
    
    print("--- TABLE 2: TRAINING TIME BY MODEL AND RATIO ---")
    print(table_training_time)


def plot_data_efficiency_subplots(results_path,
                                  selected_models,
                                  selected_train_ratio,
                                  fixed_snr,
                                  style_dict,
                                  user_col='Num_Users',
                                  max_cols=2):
    """
    Plots data efficiency (Sum-rate vs Train Ratio) in a dynamic grid 
    based on the number of users/task complexity.
    """
    # Output file name
    base = os.path.splitext(results_path)[0]
    output_file_name = base + ".pdf"

    # Load and filter CSV data
    df = pd.read_csv(results_path)
    filtered_df = df[(df['Model_Name'].isin(selected_models)) &
                     (df['SNR_dB']==fixed_snr) &
                     (df['Train_Ratio'].isin(selected_train_ratio))].copy()
    filtered_df['Train_Ratio_PCT'] = filtered_df['Train_Ratio'] * 100

    # --- Grid Calculation ---
    # Find all unique user counts in the filtered data
    unique_users = sorted(filtered_df[user_col].unique())
    n_plots = len(unique_users)
    cols = min(n_plots, max_cols)
    rows = math.ceil(n_plots / cols)

    # Plot figure using subplots
    fig, axes = plt.subplots(
        nrows=rows, ncols=cols, 
        figsize=(4.5 * cols, 3.5 * rows), # Scales figure size automatically
        constrained_layout=True           # Prevents overlapping labels
    )

    # Handle 1D array or single axis cases safely
    if n_plots == 1:
        axes_flat = [axes]
    else:
        axes_flat = axes.flatten()

    # --- Plotting Loop ---
    for i, num_users in enumerate(unique_users):
        ax = axes_flat[i]
        
        # Filter data for this specific subplot (e.g., just the 8-user data)
        subplot_data = filtered_df[filtered_df[user_col] == num_users]

        optimal_series = df[(df['Model_Name'] == 'Optimal') & (df[user_col] == num_users) & (df['SNR_dB']==fixed_snr)]['Sum_Rate']
        equal_power_series = df[(df['Model_Name'] == 'Equal-Power') & (df[user_col] == num_users) & (df['SNR_dB']==fixed_snr)]['Sum_Rate']

        if not optimal_series.empty:
            upper_bound = optimal_series.iloc[0] # Grabs the actual numerical value
            ax.axhline(y=upper_bound, color='black', linestyle='--', linewidth=1.5, 
                       zorder=1, label='Optimal' if i == 0 else "")
            
        if not equal_power_series.empty:
            lower_bound = equal_power_series.iloc[0] # Grabs the actual numerical value
            ax.axhline(y=lower_bound, color='black', linestyle=':', linewidth=1.5, 
                       zorder=1, label='Equal-Power' if i == 0 else "")

        for model, (hex_color, marker, z) in style_dict.items():
            if model in subplot_data['Model_Name'].values:
                model_data = subplot_data[subplot_data['Model_Name'] == model]
                
                # Sort by Train Ratio to ensure lines connect logically from left to right
                model_data = model_data.sort_values('Train_Ratio_PCT')
                    
                ax.plot(
                    model_data['Train_Ratio_PCT'], 
                    model_data['Sum_Rate'], 
                    color=hex_color,
                    marker=marker, 
                    markeredgecolor=hex_color,
                    label=model if i == 0 else "", # Only label the very first plot to avoid duplicate legends
                    zorder=z
                )

        # Axis Formatting for each subplot
        ax.set_title(f"{num_users} Users", fontsize=16, fontweight='bold')
        ax.set_xlabel('Training samples (%)', fontsize=14)
        
        # Only add the Y-axis label to the leftmost plots to keep the grid clean
        if i % cols == 0:
            ax.set_ylabel('Sum-Rate (bps/Hz)', fontsize=14)
            
        ax.grid(True, linestyle='--', alpha=0.5)

    # Clean up any unused subplots (if you have an odd number of user configurations)
    for j in range(n_plots, len(axes_flat)):
        axes_flat[j].axis('off')

    # Global Legend Formatting
    # Pull the handles from the first plot and place them globally at the top
    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles, labels, 
            fontsize = 16,
            loc='upper center',
            bbox_to_anchor=(0.5, 1.12), # Pushes legend just above the title
            ncol=len(labels), 
            frameon=False
        )

    plt.savefig(output_file_name, bbox_inches='tight') # bbox_inches catches the moved legend
    plt.show()

def plot_noise_robustness_subplots(results_path,
                                   selected_models,
                                   selected_snr,
                                   fixed_train_ratio,
                                   style_dict,
                                   user_col='Num_Users',
                                   max_cols=2):
    """
    Plots noise robustness (Sum-rate vs SNR) in a dynamic grid 
    based on the number of users/task complexity for a strictly fixed training ratio.
    """
    # Output file name appended with _noise_robustness
    base = os.path.splitext(results_path)[0]
    output_file_name = base + "_noise_robustness.pdf"

    # Load CSV data
    df = pd.read_csv(results_path)
    
    # Filter for ML models matching the fixed train ratio and selected SNRs
    filtered_df = df[(df['Model_Name'].isin(selected_models)) &
                     (df['Train_Ratio'] == fixed_train_ratio) &
                     (df['SNR_dB'].isin(selected_snr))].copy()

    # --- Grid Calculation ---
    unique_users = sorted(filtered_df[user_col].unique())
    n_plots = len(unique_users)
    
    if n_plots == 0:
        print("No data found for the given criteria.")
        return
        
    cols = min(n_plots, max_cols)
    rows = math.ceil(n_plots / cols)

    # Plot figure using subplots
    fig, axes = plt.subplots(
        nrows=rows, ncols=cols, 
        figsize=(4.5 * cols, 3.5 * rows), # Scales figure size automatically
        constrained_layout=True           # Prevents overlapping labels
    )

    # Handle 1D array or single axis cases safely
    if n_plots == 1:
        axes_flat = [axes]
    else:
        axes_flat = axes.flatten()

    # --- Plotting Loop ---
    for i, num_users in enumerate(unique_users):
        ax = axes_flat[i]
        
        # Filter ML data for this specific subplot
        subplot_data = filtered_df[filtered_df[user_col] == num_users]

        # --- Extract Baselines ---
        # Baselines usually do not have a 'Train_Ratio', so we pull them from the main df 
        # and sort them by SNR to ensure the line plots correctly from left to right.
        optimal_data = df[(df['Model_Name'] == 'Optimal') & 
                          (df[user_col] == num_users) & 
                          (df['SNR_dB'].isin(selected_snr))].sort_values('SNR_dB')
        
        equal_power_data = df[(df['Model_Name'] == 'Equal-Power') & 
                              (df[user_col] == num_users) & 
                              (df['SNR_dB'].isin(selected_snr))].sort_values('SNR_dB')

        # Plot Baselines as curves, NOT horizontal lines
        if not optimal_data.empty:
            ax.plot(optimal_data['SNR_dB'], optimal_data['Sum_Rate'], 
                    color='black', linestyle='--', linewidth=1.5, 
                    zorder=1, label='Optimal' if i == 0 else "")
            
        if not equal_power_data.empty:
            ax.plot(equal_power_data['SNR_dB'], equal_power_data['Sum_Rate'], 
                    color='black', linestyle=':', linewidth=1.5, 
                    zorder=1, label='Equal-Power' if i == 0 else "")

        # --- Plot ML Models ---
        for model, (hex_color, marker, z) in style_dict.items():
            if model in subplot_data['Model_Name'].values:
                model_data = subplot_data[subplot_data['Model_Name'] == model]
                
                # Sort by SNR instead of Train Ratio
                model_data = model_data.sort_values('SNR_dB')
                    
                ax.plot(
                    model_data['SNR_dB'], 
                    model_data['Sum_Rate'], 
                    color=hex_color,
                    marker=marker, 
                    markeredgecolor=hex_color,
                    label=model if i == 0 else "", # Avoid duplicate legends
                    zorder=z
                )

        # Axis Formatting for each subplot
        ax.set_title(f"{num_users} Users", fontsize=16, fontweight='bold')
        ax.set_xlabel('SNR (dB)', fontsize=14) 

        ax.set_xticks(selected_snr)
        ax.set_xticklabels([f"{val}" for val in selected_snr], fontsize=12)
        
        # Only add the Y-axis label to the leftmost plots
        if i % cols == 0:
            ax.set_ylabel('Sum-Rate (bps/Hz)', fontsize=14) 
            
        ax.grid(True, linestyle='--', alpha=0.5)

    # Clean up any unused subplots
    for j in range(n_plots, len(axes_flat)):
        axes_flat[j].axis('off')

    # Global Legend Formatting
    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles, labels, 
            fontsize=16,
            loc='upper center',
            bbox_to_anchor=(0.5, 1.12), # Pushes legend just above the title
            ncol=len(labels), 
            frameon=False
        )

    plt.savefig(output_file_name, bbox_inches='tight')
    plt.show()

def plot_tsne_grid_models_only(tsne_results_path,
                               model_names,
                               complexity_selection=None,
                               max_cols=3):
    """
    Plots a grid of t-SNE latent spaces for selected models, 
    omitting the physical scenario layout subplot.
    """
    # Output file name
    base = os.path.splitext(tsne_results_path)[0]
    output_file_name = base + "_models_only.pdf"

    # Load and filter CSV data
    df_tsne = pd.read_csv(tsne_results_path)

    if complexity_selection:
        if 'n_beams' in df_tsne.columns:
            df_tsne = df_tsne[df_tsne['n_beams']==complexity_selection]

    # Extract labels strictly from the t-SNE dataset
    tsne_labels = set(df_tsne['label'].unique())
    all_unique_labels = sorted(list(tsne_labels))
    
    all_unique_labels = [int(lbl) for lbl in all_unique_labels]
    n_colors = len(all_unique_labels)
    
    # Generate dynamic palette
    if n_colors <= 2:
        raw_colors = sns.color_palette("tab10", n_colors)
    else:
        raw_colors = sns.color_palette("tab20", n_colors)
        
    palette_map = dict(zip(all_unique_labels, raw_colors))
    
    # Calculate dynamic layout (No +1 for the scenario layout)
    total_plots = len(model_names)  
    
    # Handle edge case where no models are provided
    if total_plots == 0:
        print("No models provided to plot.")
        return
        
    cols = min(total_plots, max_cols)   
    rows = math.ceil(total_plots / cols) 
    
    # Setup figure
    fig, axes = plt.subplots(
        nrows=rows, ncols=cols, 
        figsize=(3.5 * cols, 3.5 * rows),
        gridspec_kw={'wspace': 0, 'hspace': 0.2}
    )
    
    # Normalize axes array for 1D or single plot cases
    if isinstance(axes, plt.Axes):
        axes = [axes]
    else:
        axes = axes.flatten()

    # Plot t-SNE embeddings
    for i, model in enumerate(model_names):
        ax_idx = i # Plots start directly at index 0
        subset = df_tsne[df_tsne['model_name'] == model]
        
        plot_tsne_cell(axes[ax_idx], subset, 'tsne_1', 'tsne_2', 'label', palette_map, show_legend=False)
        axes[ax_idx].set_title(model, fontsize=14, pad=10)

    # Clean up any empty subplots in the grid
    for j in range(total_plots, len(axes)):
        axes[j].axis('off')

    plt.savefig(output_file_name)
    plt.show()