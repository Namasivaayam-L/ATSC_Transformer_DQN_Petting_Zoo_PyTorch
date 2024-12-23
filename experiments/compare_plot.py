import sys,os
import pandas as pd
import matplotlib.pyplot as plt

from sumo_rl.exploration import plot_epsilon

# Get CSV file paths from command line arguments
col_val = sys.argv[1]
csv_file_1 = sys.argv[2]
csv_file_2 = sys.argv[3]

for rf in ['dwt','pressure','queue']:
    csv_1 = csv_file_1.replace('#',rf)
    csv_2 = csv_file_2.replace('#',rf)
    
    df1 = pd.read_csv(csv_1)
    df2 = pd.read_csv(csv_2)

    x = df1['step'].values
    y1 = df1[col_val].values
    y2 = df2[col_val].values
    
    plt.figure(figsize=(10, 6))
    
    # plt.plot(x, y1, label='DQN', color='blue')
    # plt.plot(x, y2, label='TRF-DQN', color='red')

    plt.plot(x, y1, label='DQN', color='red', linestyle='--', linewidth=2)
    plt.plot(x, y2, label='TRF-DQN', color='blue', linestyle='-', linewidth=2)

    plt.fill_between(x, y1, y2, where=(y2 >= y1), interpolate=True, color='blue', alpha=0.3)
    plt.fill_between(x, y1, y2, where=(y2 < y1), interpolate=True, color='red', alpha=0.8)


    plt.title(f'{rf} {col_val} Comparison DQN vs TRF-DQN')
    plt.xlabel('Time Step')
    plt.ylabel(col_val)
    plt.legend()
    plt.grid(True)
    
    def gg_par_dir(path):
        parent = os.path.dirname(path)
        grandparent = os.path.dirname(parent)
        greatgrandparent = os.path.dirname(grandparent)
        plot_dir = os.path.dirname(os.path.dirname(os.path.dirname(path)))+'/plots/'
        os.makedirs(plot_dir, exist_ok=True)
        if 'trf' in plot_dir:    
            print(plot_dir+f'{rf}_{col_val}_line_dash_fill.png')
        return plot_dir+f'{rf}_{col_val}_line_dash_fill.png'
    
    plt.savefig(gg_par_dir(csv_1))
    plt.savefig(gg_par_dir(csv_2))