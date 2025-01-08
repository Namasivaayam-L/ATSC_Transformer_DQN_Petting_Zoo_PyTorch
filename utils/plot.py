import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os

def plot_rewards(csv_file_path):
    """Plots and saves reward data from a CSV file.

    Args:
        csv_file_path (str): Path to the CSV file containing reward data.  Must have 'ep', '1', '2', '5', '6' columns.
    """
    file_name = os.path.dirname(csv_file_path)
    plot_dir = os.path.join(file_name, 'plots')  #More readable path construction
    os.makedirs(plot_dir, exist_ok=True)
    plot_save_path = os.path.join(plot_dir, 'rewards_combined.png')

    data = pd.read_csv(csv_file_path)
    print(data.head(5))
    data = data.groupby('ep')[['1', '2', '5', '6']].mean().reset_index()
    data.columns = ['ep', '1', '2', '5', '6']

    ep_col = data['ep']
    reward_cols = data[['1', '2', '5', '6']].mean(axis=1)

    plt.figure(figsize=(10, 6))
    plt.plot(ep_col, reward_cols)
    plt.title('DQN-TRF Agent Rewards')
    plt.xlabel('No Of Episodes')
    plt.ylabel('Rewards')
    plt.savefig(plot_save_path)
    plt.close()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", required=True, help="Path to the CSV file")
    args = parser.parse_args()
    plot_rewards(args.file)
