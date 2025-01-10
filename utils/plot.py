import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os

import glob
from itertools import cycle
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import logging


sns.set(
    style="darkgrid",
    rc={
        "figure.figsize": (7.2, 4.45),
        "text.usetex": True,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "font.size": 15,
        "figure.autolayout": True,
        "axes.titlesize": 16,
        "axes.labelsize": 17,
        "lines.linewidth": 2,
        "lines.markersize": 6,
        "legend.fontsize": 15,
    },
)
colors = sns.color_palette("colorblind", 4)
# colors = sns.color_palette("Set1", 2)
# colors = ['#FF4500','#e31a1c','#329932', 'b', 'b', '#6a3d9a','#fb9a99']
dashes_styles = cycle(["-", "-.", "--", ":"])
sns.set_palette(colors)
colors = cycle(colors)

logging.basicConfig(level=logging.INFO)


def moving_average(interval, window_size):
    if window_size == 1:
        return interval
    window = np.ones(int(window_size)) / float(window_size)
    return np.convolve(interval, window, "same")


def plot_df(df, color, axis, xaxis, yaxis, ma=1, label=""):
    df = df.apply(pd.to_numeric, errors='coerce')
    mean = df.groupby(xaxis).mean()[yaxis]
    std = df.groupby(xaxis).std()[yaxis]
    if ma > 1:
        mean = moving_average(mean, ma)
        std = moving_average(std, ma)
    x = df.groupby(xaxis)[xaxis].mean().keys()
    axis.plot(x, mean, label=label, color=color, linestyle=next(dashes_styles))

def plot_results(files, legends=None, xaxis="step", ma=1, sep=",", xlabel="Time step (ms)"):
    labels = cycle(legends) if legends is not None else cycle([str(i) for i in range(len(files))])
    save_path = os.path.dirname(os.path.dirname(files[0]))
    print(f'{save_path}/plots/')
    os.makedirs(f'{save_path}/plots/',exist_ok=True)
    
    # File reading and grouping    
    for file in files:
        main_df = pd.DataFrame()
        for f in glob.glob(file + "*"):
            df = pd.read_csv(f, sep=sep)
            logging.info(f"Reading file: {f}")
            if main_df.empty:
                main_df = df
            else:
                main_df = pd.concat((main_df, df))
        
        for yax in ['system_mean_waiting_time', 'system_mean_speed']:            
            fig, axs = plt.subplots()
            logging.info(f"Plotting {yax}")
            plot_df(main_df, axis=axs, xaxis=xaxis, yaxis=yax, label=next(labels), color=next(colors), ma=ma)
            axs.set_title(str(yax+' vs '+xlabel))
            axs.set_ylabel(yax)
            axs.set_xlabel(xlabel)
            axs.set_ylim(bottom=0)
            try:
                plt.savefig(f'{save_path}/plots/{yax}_1.png')
                logging.info(f"Saved plot to {save_path}/plots/{yax}.png")
            except Exception as e:
                logging.error(f"Error saving plot: {e}")
            plt.clf()  # Clear the current figure to avoid overlapping plots
            plt.close(fig) 


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
