import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
# plt.rcParams['text.usetex'] = True
# plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath,amssymb}'
import numpy as np
from scipy.interpolate import griddata
from typing import Callable, TypeVar, List, Optional, Union
from matplotlib.gridspec import GridSpec
from matplotlib.cm import ScalarMappable
import seaborn as sns
import pandas as pd
from pathlib import Path
import itertools
import NonLinSAE.utils.plot.plotdefaults
plt.rc('font', **{'size': 30})



def plot_column_vs_r_and_p(df, column_name, title = None, colorbar_label = "Loss Values", ax = None, norm=None, interpolation = "nearest"):
    pivot = df.pivot(index='ratio', columns='p_feat', values=column_name)

    # Convert lists to numpy arrays
    ratios = np.array(df["ratio"])
    p_feats = np.array(df["p_feat"])
    column_vals = np.array(df[column_name])
    #losses = np.array([min(loss, .05) for loss in losses])

    # # Define a finer grid for Ainterpolation
    # a_fine = np.linspace(min(ratios), max(ratios), 100)
    # b_fine = np.linspace(min(p_feats), max(p_feats), 100)
    # A_fine, B_fine = np.meshgrid(a_fine, b_fine)

    # # Interpolate the c values over the finer grid using 'linear' method
    # C_fine = griddata((ratios, p_feats), column_vals, (A_fine, B_fine), method=interpolation)
    # pivot = df.pivot(index='ratio', columns='p_feat', values=column_name)



    if ax == None:
        # Plotting the smooth heatmap
        plt.figure(figsize=(6, 5))
        # plt.pcolormesh(B_fine, A_fine, C_fine, cmap='viridis', norm=norm)
        plt.pcolormesh(pivot.columns, pivot.index, pivot.values, cmap='viridis', norm=norm)
        plt.colorbar(label=colorbar_label)
        plt.ylabel('$n_d/n_s$')
        plt.xlabel('$p$')
        if title:
            plt.title(title)
        plt.show()
    else:
        # Plotting the smooth heatmap
        # im = ax.contourf(pivot.columns, pivot.index, pivot.values, cmap='viridis', norm=norm)
        ax.pcolormesh(pivot.columns, pivot.index, pivot.values, cmap='viridis', norm=norm)
        ax.set_ylabel('$n_d/n_s$')
        ax.set_xlabel('$p$')
        if title:
            ax.set_title(title)
    return


def plot_multiple_column_vs_r_and_p(df: pd.DataFrame, 
                                    col_name: str, 
                                    groupby_col: str, 
                                    label: str,
                                    paper_directory: Optional[str] = None,
                                    show_plot: bool=False,
                                    orientation: Optional[tuple] = None,
                                    logscale: bool = False,
                                    filename_prefix: Optional[str] = None):
    
    try: 
        groupby_vals = sorted(df[groupby_col].unique())
    except:
        raise ValueError('Column not sortable.')

    if orientation:
        assert len(groupby_vals) == orientation[0] * orientation[1]
        fig, axes = plt.subplots(nrows=orientation[0], ncols = orientation[1], figsize=(6*orientation[1], 5*orientation[0]))
        axes = axes.reshape(-1)
    else:
        fig, axes = plt.subplots(nrows=1, ncols = len(groupby_vals), figsize=(6*len(groupby_vals), 5))
    
    vmin = df[col_name].min()
    vmax = df[col_name].max()
    if logscale:
        norm = LogNorm(vmin=vmin, vmax=vmax)
    else:
        norm = plt.Normalize(vmin=vmin, vmax=vmax)

    sm = ScalarMappable(cmap=plt.get_cmap('viridis'), norm=norm)
    sm.set_array([])

    for (i, groupby_val) in enumerate(groupby_vals):
        if groupby_col == "n_sparse":
            plot_column_vs_r_and_p(df[df[groupby_col] == groupby_val], col_name, title=f'$n_s= {groupby_val}$', colorbar_label=label, norm=norm, ax = axes[i])
        else:
            plot_column_vs_r_and_p(df[df[groupby_col] == groupby_val], col_name, title=f'{groupby_val}', colorbar_label=label, norm=norm, ax = axes[i])

    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.86, 0.1, 0.01, 0.78])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label(label)

    if filename_prefix: 
        filename = f'{filename_prefix}_RPS_{col_name}.png'
    else:
        filename = f'RPS_{col_name}.png'


    plt.savefig(f'figures/{filename}', bbox_inches='tight', dpi=300)
    if paper_directory:
        plt.savefig(Path(paper_directory) / f'{filename}', bbox_inches='tight', dpi=300)
    if show_plot:
        plt.show()
    else:
        plt.close()
    return


def plot_columns(df: pd.DataFrame,
                 xcol: str,
                 ycol: str, 
                 huecols: Union[List[str], str], 
                 title = None, 
                 scatter: bool = False,
                 alpha: float = 0.6):

    if isinstance(huecols, list):
        df['hue_combined'] = df[huecols].astype(str).agg(' - '.join, axis=1)
        huecol = 'hue_combined'
    else:
        huecol = huecols

    # Create the plot
    plt.figure(figsize=(5*3, 5*3)) 
    if scatter:
        sns.scatterplot(data=df, x=xcol, y=ycol, hue=huecol, size=huecol, alpha=alpha)
    else:
        sns.lineplot(data=df, x=xcol, y=ycol, hue=huecol, size=huecol, style=huecol, alpha=alpha)


    # Customize the plot
    if title == None:
        plt.title(f'Relationship between {xcol} and {ycol} for different values of {huecols}')
    else:
        plt.title(title)
    plt.xlabel(xcol)
    plt.ylabel(ycol)

    # Show the legend
    if isinstance(huecols, list):
        plt.legend(title=' - '.join(huecols))
    else:
        plt.legend(title=huecol)

    # Adjust layout to prevent cutting off the legend
    plt.tight_layout()

    # Show the plot
    plt.show()
    return

def plot_columns_simple(df, xcol:str, ycol:str, huecol:str, styles:dict, xlabel: str, ylabel: str, title, show=False, filename = None, paper_directory = None):
   
    # plt.figure(figsize=(15, 15)) 
    plt.figure(figsize=(14, 12)) 
    df_sorted = df.sort_values(by=xcol)

    # Define line styles and widths
    for hc, style in styles.items():
        if hc not in list(df['experiment_nickname'].unique()):
            continue
        data = df_sorted[df_sorted[huecol] == hc]
        if hc == 'opt_linear':
            plt.plot(np.array(data[xcol]), np.array(data[ycol]), color=style['color'], label=style['label'])
            continue
        plt.scatter(np.array(data[xcol]), np.array(data[ycol]), **style)


    plt.xlabel(xlabel, fontsize=40)
    plt.ylabel(ylabel, fontsize=40)
    plt.tick_params(labelsize=30)
    plt.title(title)
    plt.legend(title='Model')
    # plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    if filename:        
        plt.savefig(f'figures/{filename}.png', bbox_inches='tight', dpi=300)
        if paper_directory:
            plt.savefig(Path(paper_directory) / f'{filename}.png', bbox_inches='tight', dpi=300)

    if show:
        plt.show()
    return 



# def plot_columns_2(df, xcol, ycol, hue_cols, title=None, scatter=False, figsize=(10, 6), 
#                  use_markers=True, use_linestyles=True, alpha=0.7, offset=False):
#     plot_df = df.copy()
    
#     if isinstance(hue_cols, list):
#         plot_df['hue_combined'] = plot_df[hue_cols].astype(str).agg(' - '.join, axis=1)
#         hue_col = 'hue_combined'
#     else:
#         hue_col = hue_cols

#     plt.figure(figsize=figsize)
    
#     # Define line styles and markers
#     linestyles = ['-', '--', '-.', ':'] if use_linestyles else ['-']
#     markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*'] if use_markers else [None]
    
#     # Get unique hue values
#     hue_values = plot_df[hue_col].unique()
    
#     # Create style cycler
#     style_cycler = itertools.cycle(itertools.product(linestyles, markers))
    
#     for i, hue_value in enumerate(hue_values):
#         subset = plot_df[plot_df[hue_col] == hue_value]
#         linestyle, marker = next(style_cycler)
        
#         x = subset[xcol]
#         y = subset[ycol]
        
#         if offset:
#             y = y + i * 0.01 * y.mean()  # Add a small offset
        
#         if scatter:
#             plt.scatter(x, y, label=hue_value, marker=marker, alpha=alpha)
#         else:
#             plt.plot(x, y, label=hue_value, linestyle=linestyle, marker=marker, alpha=alpha)

#     plt.title(title or f'Relationship between {xcol} and {ycol}')
#     plt.xlabel(xcol)
#     plt.ylabel(ycol)
    
#     legend_title = ' - '.join(hue_cols) if isinstance(hue_cols, list) else hue_col
#     plt.legend(title=legend_title)
    
#     plt.tight_layout()
#     plt.show()

# Example usage
# df = your_dataframe
# plot_columns(df, 'your_xcol', 'your_ycol', ['experiment_name', 'n_sparse'], 
#              title='Your Custom Title', use_markers=True, use_linestyles=True, 
#              alpha=0.7, offset=False)


# def plot_columns_mp(df, xcol, ycol, huecol, title=None):
#     if not all(col in df.columns for col in [xcol, ycol, huecol]):
#         raise ValueError("One or more specified columns not found in the dataframe")

#     # Create the plot
#     fig, ax = plt.subplots(figsize=(10, 6))

#     # Get unique values in the huecol
#     hue_values = df[huecol].unique()

#     # Create a color map
#     # cmap = plt.get_cmap('tab10')
#     cmap = plt.get_cmap()
#     colors = cmap(np.linspace(0, 1, len(hue_values)))

#     # Plot a line for each unique value in huecol
#     for i, hue_value in enumerate(hue_values):
#         subset = df[df[huecol] == hue_value]
#         subset = subset.sort_values(by=xcol)  # Sort by x values
#         ax.plot(subset[xcol], subset[ycol], label=str(hue_value), color=colors[i])

#     # Customize the plot
#     if title is None:
#         ax.set_title(f'Relationship between {xcol} and {ycol} for different {huecol}')
#     else:
#         ax.set_title(title)
#     ax.set_xlabel(xcol)
#     ax.set_ylabel(ycol)

#     # Show the legend
#     ax.legend(title=huecol, bbox_to_anchor=(1.05, 1), loc='upper left')

#     # Adjust layout to prevent cutting off the legend
#     plt.tight_layout()

#     # Show the plot
#     plt.show()

def plot_matrix_vector(A, B):
    # Combine A and B for unified color scaling
    width_unit = A.shape[0]
    assert A.shape[0] == A.shape[1]

    gap_size = int(width_unit * .05 + 1)
    tot_width = width_unit + gap_size * 2 + 1
    
    combined = np.hstack((A, B.reshape(-1, 1)))
    
    # Create a diverging colormap
    cmap = plt.cm.bwr  # Red-White-Blue colormap, reversed
    
    # Determine the absolute maximum value for symmetric color scaling
    max_abs_val = max(abs(combined.min()), abs(combined.max()))
    
    # Create the figure
    fig = plt.figure(figsize=(12 * (tot_width / width_unit), 12))
    
    # Create GridSpec
    gs = GridSpec(width_unit, tot_width, figure=fig, wspace=0)
    
    # Create subplots
    ax1 = fig.add_subplot(gs[:, :width_unit])  # Matrix A
    ax2 = fig.add_subplot(gs[:, -1 - gap_size])  # Vector B
    cax = fig.add_subplot(gs[:, -1])  # Colorbar
    
    # Plot matrix A
    im1 = ax1.imshow(A, cmap=cmap, vmin=-max_abs_val, vmax=max_abs_val, aspect='auto')
    ax1.set_title('Matrix A')
    ax1.set_xlabel('Column')
    ax1.set_ylabel('Row')
    ax1.set_xticks([])
    ax1.set_yticks([])
    
    # Plot vector B
    im2 = ax2.imshow(B.reshape(-1, 1), cmap=cmap, vmin=-max_abs_val, vmax=max_abs_val, aspect='auto')
    ax2.set_title('Vector B')
    ax2.set_xlabel('Column')
    ax2.set_yticks([])  # Remove y-axis ticks for B
    ax2.set_xticks([])  # Show only one x-tick for the vector
    
    # Remove spines between A and B
    # ax1.spines['right'].set_visible(False)
    # ax2.spines['left'].set_visible(False)
    
    # Add colorbar
    plt.colorbar(im1, cax=cax, orientation='vertical', label='Value')
    
    # Adjust layout
    # plt.tight_layout()
    plt.show()