# External Packages
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import torch as t
import pandas as pd

import itertools

from importlib import reload
from tqdm import tqdm

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
from dataclasses import dataclass
from typing import Callable, Union, List
import re

@dataclass
class NewMeasurement:
    col_name: str
    label: str
    func: Callable[[model.Model], Union[float,int]]

def add_log_after_dollar(s:str) -> str:
    return re.sub(r'(\$)', r'\1\\log_{10} ', s, count=1)


def process_experiments(experiments: dict, overwrite: bool = False):
    combined_df = pd.DataFrame()
    combined_dict = {}
    
    for (experiment_name, experiment_label) in experiments.items():
        if not overwrite:
            try:
                df, mm_dict = ms.load_df_from_file(experiment_name=experiment_name)
            except:
                list_of_model_measurements = ms.load_model_measurements_from_models(experiment_name)
                df, mm_dict = ms.model_measurements_to_dataframe(list_of_model_measurements, experiment_name=experiment_name)
                df['experiment_label'] = experiment_label
        else:
            list_of_model_measurements = ms.load_model_measurements_from_models(experiment_name)
            df, mm_dict = ms.model_measurements_to_dataframe(list_of_model_measurements, experiment_name=experiment_name)
            df['experiment_label'] = experiment_label

        df['experiment_name'] = experiment_name        
        combined_df = pd.concat([combined_df, df], ignore_index=True)
        combined_dict.update(mm_dict)
    
    return combined_df, combined_dict

def make_paper_plots(df: pd.DataFrame,
                    mm_dict: dict, 
                    # paper_directory: str = '/Users/alexinf/Dropbox/apps/Overleaf/Sparsity and Computation in NNs/NonLinSAE/images/',
                    paper_directory = None,
                    orientation = (2,3),
                    show_plot: bool = False):
    cols_to_plot = {
        'chi_varvar': r'$\Delta\mathrm{var}(\nu)$',
        'chi_pval': r'$p_{\mathrm{KS}}$',
        'diag_var': r'$\Delta \mathrm{diag}(W)$',
        'bias_var': r'$\Delta b_i$',
        'chi_meanvar': r'$\Delta \mathrm{var}_X(\nu)$',
        'final_loss': r'$\mathrm{Loss}$'
        }
    for (col_name, col_label) in cols_to_plot.items():
        lrp.plot_multiple_column_vs_r_and_p(df, col_name, groupby_col='n_sparse', label=col_label, paper_directory=paper_directory, orientation=orientation, show_plot=show_plot)
        lrp.plot_multiple_column_vs_r_and_p(df, col_name, groupby_col='n_sparse', label=col_label, paper_directory=paper_directory, orientation=orientation, show_plot=show_plot, logscale=True)

    funcs_to_plot = {
        "Max Lyapunov": {
            "label": r"$\Lambda$",
            "func": lambda x: mm_dict[x].Lyapunov[1].item()
        }
        # "Delta Enu": {
        #     "label": r'$\Delta \nu$',
        #     "func": lambda x: mm_dict[x].eps_mat
        # }
    }

    for (f, f_dict) in funcs_to_plot.items():
        assert f not in df.keys()
        col_name = f
        col_label = f_dict["label"]
        df[col_name] = df['model_id'].apply(lambda x: f_dict['func'](x))
        print(np.array(df[col_name])[0])
        lrp.plot_multiple_column_vs_r_and_p(df, col_name, groupby_col='n_sparse', label=col_label, paper_directory=paper_directory, orientation=orientation, show_plot=show_plot)
        lrp.plot_multiple_column_vs_r_and_p(df, col_name, groupby_col='n_sparse', label=col_label, paper_directory=paper_directory, orientation=orientation, show_plot=show_plot, logscale=True)

if __name__ == "__main__":
    experiments = {"non_average_run_09_26": "Trained Models (128,...,4096)", 
                   "non_average_run_8192_09_26": "Trained Models (8192)"}

    df, mm_dict = process_experiments(experiments, overwrite=False)
    n_sparses_to_keep = [128, 1024, 8192]
    df = df[df['n_sparse'].isin(n_sparses_to_keep)]
    make_paper_plots(df, mm_dict, orientation=None)

