# python experiments/compare_plot.py system_mean_waiting_time outputs/all_veh_wt_obs/trf_dqn/2x2/#/emb_relu_bn/2000eps-p40/test/20ep/csv/19ep.csv outputs/all_veh_wt_obs/dqn/2x2/#/emb_relu_bn/2000eps-p40/test/20ep/csv/19ep.csv 
#   outputs/fixed_time/2x2/2000eps-40/csv/1999ep.csv outputs/random_time/2x2/2000eps-40/csv/1999ep.csv

import sys,os
import pandas as pd
import matplotlib.pyplot as plt

# from sumo_rl.exploration import plot_epsilon

# Get CSV file paths from command line arguments
col_val = sys.argv[1]
csv_file_1 = sys.argv[2]
csv_file_2 = sys.argv[3]
csv_file_3 = sys.argv[4]
csv_file_4 = sys.argv[5]
# Define styling for the different algorithms
style_dict = {
    'TRF-DQN': {'color': 'tab:blue', 'linestyle': '-', 'linewidth': 2},
    'DQN': {'color': 'tab:red', 'linestyle': '--', 'linewidth': 2},
    'Fixed-time': {'color': 'tab:green', 'linestyle': '-', 'marker': 'o', 'markersize': 4, 'markevery': 20, 'markeredgecolor': 'black', 'markeredgewidth': 0.5},
    'Random-time': {'color': 'tab:purple', 'linestyle': '-', 'marker': 'x', 'markersize': 4, 'markevery': 20, 'markeredgecolor': 'black', 'markeredgewidth': 0.5},
}


    
def gg_par_dir(path):
    parent = os.path.dirname(path)
    grandparent = os.path.dirname(parent)
    greatgrandparent = os.path.dirname(grandparent)
    plot_dir = os.path.dirname(os.path.dirname(os.path.dirname(path)))+'/plots/'
    os.makedirs(plot_dir, exist_ok=True)
    if 'trf' in plot_dir:    
        print(plot_dir+f'{rf}_{col_val}_line_dash_fill.png')
    return plot_dir+f'{rf}_{col_val}_line_dash_fill.png'


# Load and plot the data
# for rf in ['dwt','pressure','queue']: # commented out since you only use dwt
for rf in ['dwt']:
    csv_1 = csv_file_1.replace('#', rf)
    csv_2 = csv_file_2.replace('#', rf)
    # csv_3 = csv_file_3.replace('#', rf)
    # csv_4 = csv_file_4.replace('#', rf)

    df1 = pd.read_csv(csv_1)
    df2 = pd.read_csv(csv_2)
    df3 = pd.read_csv(csv_file_3)
    df4 = pd.read_csv(csv_file_4)

    x = df1['step'].values

    plt.figure(figsize=(10, 6))

    # Plot TRF-DQN
    plt.plot(x, df1[col_val].values, label='TRF-DQN', **style_dict['TRF-DQN'])

    # Plot DQN
    plt.plot(x, df2[col_val].values, label='DQN', **style_dict['DQN'])

    # Plot Fixed-time
    plt.plot(x, df3[col_val].values, label='Fixed-time', **style_dict['Fixed-time'])

    # Plot Random-time
    plt.plot(x, df4[col_val].values, label='Random-time', **style_dict['Random-time'])


    plt.title(f'{rf} {col_val} Comparison')  # More general title
    plt.xlabel('Time Step')
    plt.ylabel(col_val)
    plt.legend()
    plt.grid(True)
    
    plt.savefig(gg_par_dir(csv_1))
    plt.savefig(gg_par_dir(csv_2))