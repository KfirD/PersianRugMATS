import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import torch as t
import pandas as pd

from importlib import reload
from tqdm import tqdm
import itertools

# Custom Packages
import model
import data as scd
from utils import model_scan as ms
from utils import model_analytics as ma
from utils import synthetic_model as sm
from NonLinSAE import hadamard_model as hm
from utils import opt_linear
from utils import opt_nonlin_sym
from utils.plot import lrp
from NonLinSAE.make_paper_plots import process_experiments

import utils.opt_linear as ol


from dataclasses import dataclass
from typing import Callable, Union, List



def process_hadamard_experiments(dict_of_experiments, overwrite: bool = False):
    combined_df = pd.DataFrame()
    combined_dict = {}
    
    for (experiment_name, experiment_dict) in dict_of_experiments.items():
        df, mm_dict = ms.load_hadamard_model_measurements(experiment_name, overwrite=overwrite)
        df['experiment_name'] = experiment_name
        for (k,v) in experiment_dict.items():
            df[k] = v
        combined_df = pd.concat([combined_df, df], ignore_index=True)
        combined_dict.update(mm_dict)

    return combined_df, combined_dict

def linear_model_from_row(row):
    model = ol.LinearModel(int(row['n_dense']), int(row['n_sparse']), row['p_feat'])
    return model


if __name__ == "__main__":
    trained_experiments = {
        "non_average_run_09_26": "Trained Models (128,...,4096)", 
        "non_average_run_8192_09_26": "Trained Models (8192)"}
    
    hadamard_experiments = {
        'hadamard_128_num_tries_5_better_spec': {
            'experiment_label': r'$\mathrm{Hadamard} (n_s=128)$',
            'experiment_nickname': 'h-128'
            },
        'hadamard_1024_num_tries_5_better_spec': {
            'experiment_label': r'$\mathrm{Hadamard} (n_s=1024)$',
            'experiment_nickname': 'h-1024'
            },
        '8192_overnight': {
            'experiment_label': r'$\mathrm{Hadamard} (n_s=8192)$',
            'experiment_nickname': 'h-8192'
            }
    }

    styles = {
        'h-8192': {'label': r'$\mathrm{Persian\ Rug}\ (n_s=8192)$',
                   'color': 'black', 
                #    'linestyle': '-', 
                #    'linewidth': 12, 
                #    'alpha': 0.5,
                   'marker': 'x',
                   's': 150,
                #    'markersize': 12
                   },      
        't-128': {'label': r'$\mathrm{Trained\ Model}\ (n_s=128)$', 
                  'color': 'red', 
                #   'linestyle': '-', 
                #   'linewidth': 4, 
                  'alpha': 0.5,
                #   'marker': 'x',
                   's': 100,
                #   'markersize': 12
                  },
        't-1024': {'label': r'$\mathrm{Trained\ Model}\ (n_s=1024)$', 
                  'color': 'green', 
                #   'linestyle': '-', 
                #   'linewidth': 4, 
                  'alpha': 0.5,
                #   'marker': 'D', 
                   's': 100,
                #   'markersize': 12
                  },
        't-8192': {'label': r'$\mathrm{Trained\ Model}\ (n_s=8192)$', 
                  'color': 'orange', 
                #   'linestyle': '-',
                #   'linewidth': 4, 
                  'alpha': 0.5,
                #   'marker': '^', 
                   's': 100
                #   'markersize': 12
                },
        'opt_linear': {'label': r'$\mathrm{Optimal\ Linear\ Model}$', 
                  'color': 'blue', 
                #   'linestyle': '-',
                #   'linewidth': 4, 
                #   'alpha': 0.7,
                'marker': 'x', 
                's': 100    
                #   'markersize': 12
                }
    }

    n_sparses = [8192]
    n_sparses_to_keep = [8192]
    n_sparse_for_hstar = 8192
    target_p_feats = [0.001, 0.01, 0.05, 0.1, 0.4, 0.6]
    target_ratios = [0.01, 0.1, 0.3, 0.5, 0.8]
    

    df, mm_dict = process_experiments(trained_experiments, overwrite=False)
    df['experiment_label'] = df['n_sparse'].apply(lambda x: 'Trained Model ' + r'$(n_s =' + f'{x}' + r')$')
    df['experiment_nickname'] = df['n_sparse'].apply(lambda x: 't-' +f'{x}')
    h_df, h_dict = process_hadamard_experiments(hadamard_experiments, overwrite=False)

    # grid_width = 20
    # ratios = np.linspace(0.002001, 0.8, grid_width),
    # p_feats = np.linspace(0.002001, 0.8, grid_width)

    grid_width = 20
    ratios = np.linspace(0.002001, 0.8, grid_width)
    p_feats = np.linspace(0.002001, 0.8, grid_width)

  
    combinations = list(itertools.product(ratios, p_feats, n_sparses))
    lin_df = pd.DataFrame(combinations, columns=['ratio', 'p_feat', 'n_sparse'])
    lin_df['n_dense'] = lin_df[['ratio','n_sparse']].apply(lambda x: max(1,int(x['n_sparse']*x['ratio'])), axis=1)


    # # Create DataFrame
    lin_df['experiment_nickname'] = 'opt_linear'
    lin_df['experiment_name'] = 'optimal_linear'
    lin_df['final_loss'] = lin_df.apply(lambda x: linear_model_from_row(x).final_loss() , axis=1)

    lrp.plot_multiple_column_vs_r_and_p(h_df, 'final_loss', 'n_sparse', label='final loss', filename_prefix='hadamard')


    h_df_star = h_df[h_df['n_sparse']==n_sparse_for_hstar]
    combined_df = pd.concat([h_df_star, df, lin_df], axis=0)


    for target_p_feat in target_p_feats:
        combined_target_p_feat = ma.get_closest_models_df(combined_df, ['n_sparse', 'experiment_name'], p_feat = target_p_feat)
        lrp.plot_columns_simple(combined_target_p_feat, xcol='ratio', ycol='final_loss', huecol='experiment_nickname', styles=styles, xlabel=r'$n_d/n_s$', ylabel='Loss', title=r'Loss at $p='+f'{target_p_feat}$', filename=f'all_line_plot_target_p_feat_{target_p_feat}')
    
    combined_df = pd.concat([h_df_star, df], axis=0)

    for target_ratio in target_ratios:
        combined_target_ratio = ma.get_closest_models_df(combined_df, groupby_cols=['n_sparse', 'experiment_name'], ratio=target_ratio)
        lrp.plot_columns_simple(combined_target_ratio, xcol='p_feat', ycol='final_loss', huecol='experiment_nickname', styles=styles, xlabel=r'$p$', ylabel='Loss', title=r'Loss at $r='+f'{target_ratio}$', filename=f'all_line_plot_target_ratio_{target_ratio}')



    df = df[df['n_sparse'].isin(n_sparses_to_keep)]
    combined_df = pd.concat([h_df_star, df, lin_df], axis=0)
    
    for target_p_feat in target_p_feats:
        combined_target_p_feat = ma.get_closest_models_df(combined_df, ['n_sparse', 'experiment_name'], p_feat = target_p_feat)
        lrp.plot_columns_simple(combined_target_p_feat, xcol='ratio', ycol='final_loss', huecol='experiment_nickname', styles=styles, xlabel=r'$n_d/n_s$', ylabel='Loss', title=r'Loss at $p='+f'{target_p_feat}$', filename=f'line_plot_target_p_feat_{target_p_feat}', paper_directory = None)

    combined_df = pd.concat([h_df_star, df, lin_df], axis=0)
    
    for target_ratio in target_ratios:
        combined_target_ratio = ma.get_closest_models_df(combined_df, groupby_cols=['n_sparse', 'experiment_name'], ratio=target_ratio)
        lrp.plot_columns_simple(combined_target_ratio, xcol='p_feat', ycol='final_loss', huecol='experiment_nickname', styles=styles, xlabel=r'$p$', ylabel='Loss', title=r'Loss at $r='+f'{target_ratio}$', filename=f'line_plot_target_ratio_{target_ratio}', paper_directory = None)


    




