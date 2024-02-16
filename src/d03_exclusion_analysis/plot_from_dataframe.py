import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def setup_fig_subplots(ydata_labels, max_cols=4):
    subplot_cols = min(len(ydata_labels), max_cols)
    subplot_rows = int(np.ceil(len(ydata_labels) / subplot_cols))
    fig, axes = plt.subplots(subplot_rows, subplot_cols, figsize=(4.5 * subplot_cols, 4.5 * subplot_rows),
                             constrained_layout=True)
    return fig, axes


def plot_timepoints(df, graphs_dir, ydata_labels, splitby, display):
    xdata_label = 'frame'
    graph_kind = 'line'
    splitby_vals = np.unique(df[splitby])

    for val in splitby_vals:
        fig, axs = setup_fig_subplots(ydata_labels)
        axs_r = axs.ravel()
        for i, ylabel in enumerate(ydata_labels):

            # Selects relevant data and excludes the first timepoint when the data represents a change from the previous timepoint
            df_selection = df.loc[(df[splitby] == val) & (df[ylabel] != 0)]
            sns.lineplot(data=df_selection, x=xdata_label, y=ylabel, hue='UID', ax=axs_r[i], legend=False)

        fig_title = f'{splitby} {val} over time'
        graph_title = f'{splitby}_{val}_tp.png'
        plt.suptitle(fig_title)

        graphs_path = os.path.join(graphs_dir, graph_title)
        plt.savefig(graphs_path)

        if display==True:
            plt.show()
        plt.close()


def plot_summary_metrics(df, graphs_dir, xdata_label, display):

    grouped_df = df.groupby(['img name']).apply(df_groupby_calculations).reset_index()
    datacol = grouped_df.columns.tolist()
    ydata_labels = datacol[datacol.index('initial cell area'):]

    fig, axs = setup_fig_subplots(ydata_labels)
    axs_r = axs.ravel()

    for i, ylabel in enumerate(ydata_labels):

        sns.pointplot(grouped_df, x=xdata_label, y=ylabel, estimator='median', errorbar='ci',
                      color='k', capsize=0.2, join=False, errwidth=0.6, ax=axs_r[i])
        sns.swarmplot(grouped_df, x=xdata_label, y=ylabel,
                      hue='average change in CAAX protein-negative cell area / hr',
                      palette='vlag', legend=False, ax=axs_r[i])
        axs_r[i].set_ylabel(ylabel)

    graph_path = os.path.join(graphs_dir, f'summary_metrics_by{xdata_label}.png')
    fig.savefig(graph_path)
    if display == True:
        plt.show()
    plt.close()


def df_groupby_calculations(x):
    d = {}
    d['DIV'] = x['DIV'].iloc[0]
    d['max frame'] = x['frame'].max()
    d['initial cell area'] = x['cell area'].iloc[0]
    d['final cell area'] = x['cell area'].iloc[-1]
    d['average change in cell area / hr'] = (d['final cell area'] - d['initial cell area']) / d['max timepoint']
    d['initial CAAX protein-negative cell area'] = x['CAAX protein-negative cell area'].iloc[0]
    d['final CAAX protein-negative cell area'] = x['CAAX protein-negative cell area'].iloc[-1]
    d['average change in CAAX protein-negative cell area / hr']\
        = (d['final CAAX protein-negative cell area'] - d['initial CAAX protein-negative cell area']) / d['max timepoint']
    d['initial CAAX protein-negative area / total area'] = x['CAAX protein-negative area / total area'].iloc[0]
    return pd.Series(d)


def plot_from_dataframe(df, graphs_dir, display=False):
    # get names of ydata columns to plot (all columns starting with 'CAAX protein-positive area'))
    datacol = df.columns.tolist()
    ydata_labels = datacol[datacol.index('CAAX protein-positive area'):]
    plot_timepoints(df, graphs_dir, ydata_labels, splitby='UID', display=display)
    #plot_timepoints(df, graphs_dir, ydata_labels, splitby='unique cell index', display=display)
    plot_summary_metrics(df, graphs_dir, xdata_label='DIV', display=display)


def main():
    processing_var = ''
    #graphs_dir = '/Users/kwu2/Library/CloudStorage/GoogleDrive-kwu2@stanford.edu/My Drive/Lab/CryoEM_live imaging/Experiments/CE006/Light_imaging_CE006/Live imaging_episcope/Processed/Timelapses/graphs'
    #df_name = 'overall_dataframe.csv'
    dataframe_path = "/Users/kwu2/Library/CloudStorage/GoogleDrive-kwu2@stanford.edu/My Drive/Lab/CryoEM_live_imaging/Experiments/CE012/CZI_to_process/MG488_mRcaax594_d3/img_processing/007_exclusion_analysis/region_areas_df.csv"
    graphs_dir = "/Users/kwu2/Library/CloudStorage/GoogleDrive-kwu2@stanford.edu/My Drive/Lab/CryoEM_live_imaging/Experiments/CE012/CZI_to_process/MG488_mRcaax594_d3/img_processing/007_exclusion_analysis"
    df = pd.read_csv(dataframe_path)
    plot_from_dataframe(df, graphs_dir, display=False)


if __name__ == '__main__':
    main()