from collections import defaultdict
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def plot_boxes(df, recall_label='B', precision_label='BPM', save_path=None):
    # Create separate boxplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    sns.boxplot(x='name', y=f'{recall_label}_recall', data=df, ax=ax1)
    ax1.set_title(f'Recall({recall_label}) by Model')
    ax1.set_xlabel('Model')
    ax1.set_ylabel(f'Recall')

    sns.boxplot(x='name', y=f'{precision_label}_precision', data=df, ax=ax2)
    ax2.set_title(f'Precision({precision_label}) by Model')
    ax2.set_xlabel('Model')
    ax2.set_ylabel(f'Precision')

    # Rotate x-axis labels if needed
    for ax in [ax1, ax2]:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=70)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')  # Save with high resolution and tight layout

    plt.tight_layout()
    plt.show()


def plot_one_box(df, is_precision=True, label='BPM', save_path=None):

    field = 'precision' if is_precision else 'recall'
    # Create separate boxplots
    fig, ax = plt.subplots(1, 1, figsize=(16, 6))

    sns.boxplot(x='name', y=f'{label}_{field}', data=df, ax=ax)
    ax.set_title(f'{field}({label}) by Model')
    ax.set_xlabel('Model')
    ax.set_ylabel(f'{field}')


    # Rotate x-axis labels if needed
    ax.set_xticklabels(ax.get_xticklabels(), rotation=70)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')  # Save with high resolution and tight layout

    plt.tight_layout()
    plt.show()


def plot_three_boxes(df, is_precision=True, save_path=None, source_ids=None):
    """
    Creates three vertically stacked boxplots for BPM, BP and B metrics.

    Parameters:
    df (pandas.DataFrame): Input DataFrame containing the metrics
    is_precision (bool): If True, plots precision metrics, otherwise plots recall
    save_path (str, optional): If provided, saves the plot to this path
    """

    if source_ids:
        df = df[df['source_id'].isin(source_ids)]

    plt.rcParams.update({
        'font.size': 22,  # Base font size
        'axes.titlesize': 26,  # Title font size
        'axes.labelsize': 24,  # Label font size
        'xtick.labelsize': 22,  # X-tick label size
        'ytick.labelsize': 22   # Y-tick label size
    })


    field = 'precision' if is_precision else 'recall'
    labels = ['BPM', 'BP', 'B']

    # Create figure with three subplots stacked vertically
    fig, axes = plt.subplots(3, 1, figsize=(16, 24), sharex=True)

    # Create boxplots for each label
    for ax, label in zip(axes, labels):
        sns.boxplot(x='name', y=f'{label}_{field}', data=df, ax=ax)
        ax.set_ylabel(f'{field}({label})')
        ax.set_ylim(0, 1)

        # Only show x-label and rotate x-ticks for bottom plot
        if ax == axes[0]:
            ax.set_title(f'{field} by Model')

        if ax != axes[-1]:
            ax.set_xlabel('')
            ax.set_xticklabels([])
        else:
            ax.set_xlabel('Model')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=70)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.tight_layout()
    plt.show()


def filter_and_reorder(df, col, order_list):
    """
    Reorders a DataFrame based on a specified column according to a given list of values.

    Parameters:
    df (pandas.DataFrame): Input DataFrame
    col (str): Column name to sort by
    order_list (list): List of values defining the desired order

    Returns:
    pandas.DataFrame: Reordered DataFrame
    """
    # Create a categorical type with the custom order
    cat_type = pd.CategoricalDtype(categories=order_list, ordered=True)

    # Convert the column to the categorical type
    df[col] = df[col].astype(cat_type)

    # Sort by the now-categorical column and reset to original type
    result = df.sort_values(col).copy()
    result[col] = result[col].astype(str)

    result = result[result[col].isin(order_list)]

    return result


def generate_order():

    berts = ['berel', 'dictabert', 'hebert', 'alephbert']
    delims = ['period', 'period_and_colon']
    ks = ['5', '15']

    order = ['gpt4o', 'claude_sonnet_v1', 'claude_sonnet_v2', 'DictaLM20 - Fewshot', 'SaT-12L']

    for b in berts:
        for d in delims:
            for k in ks:
                order.append(f'{b}_{d}_k{k}')

    # order += ['moishe-mlp']
    return order


def plot_scores_by_name(df, names=list(), prefixes=None, score_col="f1_pBPM_rB"):
    """
    Creates a line plot showing scores for each name across different text_ids.

    Parameters:
    df (pandas.DataFrame): DataFrame containing 'name', 'text_id', and 'score' columns
    """

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))

    all_text_ids = sorted(df['source_id'].unique())
    x_positions = range(len(all_text_ids))
    x_mapping = dict(zip(all_text_ids, x_positions))

    # Plot a line for each unique name
    if prefixes:
        for prefix in prefixes:
            names_to_plot = df[df['name'].str.startswith(prefix)]['name'].unique()
            names += names_to_plot.tolist()

    df = df[df['name'].isin(names)]

    # Plot a line for each matching name
    for name in names:
        # Get data for this name
        name_data = df[df['name'] == name]
        # Sort by text_id to ensure proper line connection
        name_data = name_data.sort_values('source_id')
        # Convert text_ids to evenly spaced positions
        x_values = [x_mapping[tid] for tid in name_data['source_id']]
        # Plot the line
        # plt.plot(name_data['source_id'], name_data[score_col], marker='o', label=name)
        jitter = 0.3
        x_values = np.array(x_values) + np.random.uniform(-jitter, jitter, len(x_values))
        ax.plot(x_values, name_data[score_col], marker='o', label=name, linestyle='None')


    # Customize the plot
    ax.set_xlabel('Text ID')
    ax.set_ylabel('Score')
    ax.set_title(f'{score_col} Scores by Name Across Text IDs')
    ax.legend()
    ax.set_xticks(x_positions)
    ax.set_xticklabels(all_text_ids, rotation=45)
    ax.grid(True, linestyle='--', alpha=0.7)

    # Adjust layout to prevent label cutoff
    fig.tight_layout()

    # Show the plot
    plt.show()


def analyze_chunk_lengths(df, name_tag='', n_cols=None, name_list=None, out_path=None):
    """
    Analyze chunk lengths for each name in the DataFrame and create histograms.

    Parameters:
    df (pandas.DataFrame): DataFrame with 'id', 'name', and 'chunked_text' columns
    """

    plt.rcParams.update({
        'font.size': 12,  # Base font size
        'axes.titlesize': 16,  # Title font size
        'axes.labelsize': 14,  # Label font size
        'xtick.labelsize': 12,  # X-tick label size
        'ytick.labelsize': 12   # Y-tick label size
    })
    if name_list:
        df = df[df['name'].isin(name_list)]

    # Dictionary to store chunk lengths for each name
    name_chunks = defaultdict(list)

    # Process each row
    for _, row in df.iterrows():
        # Split the text by // to get chunks
        chunks = row['chunked_text'].split(r'//')

        # Count words in each chunk (splitting by whitespace for Hebrew text)
        chunk_lengths = [len(chunk.strip().split()) for chunk in chunks]

        # Add lengths to the corresponding name's list
        name_chunks[row['name']].extend(chunk_lengths)

    # Create subplots based on number of unique names
    unique_names = list(name_chunks.keys())
    n_names = len(unique_names)

    # Calculate layout dimensions
    if not n_cols:
        n_cols = min(3, n_names)  # Maximum 3 columns

    n_rows = (n_names + n_cols - 1) // n_cols

    # Create figure and subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    fig.suptitle(f'Distribution of Chunk Lengths (# words) - {name_tag}', fontsize=16, y=0.98)

    # Flatten axes array for easier iteration if there are multiple rows
    if n_rows > 1:
        axes = axes.flatten()
    elif n_rows == 1 and n_cols == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes.flatten()

    # Create histograms
    for idx, name in enumerate(unique_names):
        if idx < len(axes):  # Ensure we have an axis available
            sns.histplot(data=name_chunks[name], bins=50, ax=axes[idx])
            axes[idx].set_xlim(0, 102)
            axes[idx].set_title(f'Name: {name}')
            axes[idx].set_ylabel('Count')

    # Remove empty subplots if any
    for idx in range(len(unique_names), len(axes)):
        fig.delaxes(axes[idx])

    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=300, bbox_inches='tight')

    return fig, name_chunks


if __name__ == '__main__':
    from pathlib import Path
    folder = Path(r"C:\Users\bensh\workspace\datasets\chunking_paper_ALP2025")

    df = pd.read_csv(folder / "ALP2025_analysis.csv")
    order = generate_order()
    df = filter_and_reorder(df, 'name', order_list=order)

    geonic_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    plot_three_boxes(df, save_path=folder / 'precision_3_layers__geonim.png', source_ids=geonic_ids)
