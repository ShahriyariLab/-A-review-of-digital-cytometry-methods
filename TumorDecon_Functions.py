
import pandas as pd
from scipy import stats
import numpy as np
import collections
from scipy.stats.stats import pearsonr
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
from PySingscore.singscore import singscore
from sklearn.svm import NuSVR

import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')



def sing_score(rna_df, gene_sets):
    sing_scores = pd.DataFrame()
    for cell in gene_sets.keys():
        sing_score = singscore.score(up_gene=gene_sets[cell], sample=rna_df)
        sing_score.rename(columns={'total_score': cell}, inplace=True)
        sing_scores = pd.concat([sing_scores, sing_score], axis=1)
    return sing_scores




def corr_table(methods, results, cell_types, true_freqs):
    p_corr_per_cell = pd.DataFrame(index=cell_types, columns=methods)
    p_corr_per_sample = pd.DataFrame(index=true_freqs.index, columns=methods)
    s_corr_per_cell = pd.DataFrame(index=cell_types, columns=methods)
    s_corr_per_sample = pd.DataFrame(index=true_freqs.index, columns=methods)
    for method in methods:
        for cell in cell_types:
            p_corr_per_cell.loc[cell, method] = pearsonr(results[method][cell], true_freqs[cell])[0]
            s_corr_per_cell.loc[cell, method] = spearmanr(results[method][cell], true_freqs[cell])[0]
        for i in range(len(true_freqs)):
            p_corr_per_sample.loc[true_freqs.index[i], method] = pearsonr(results[method][cell_types].iloc[i], 
                                                                          true_freqs[cell_types].iloc[i])[0]
            s_corr_per_sample.loc[true_freqs.index[i], method] = spearmanr(results[method][cell_types].iloc[i], 
                                                                           true_freqs[cell_types].iloc[i])[0]
    return np.abs(p_corr_per_cell), np.abs(p_corr_per_sample), np.abs(s_corr_per_cell), np.abs(s_corr_per_sample)




def corr_mean_std(corr_per_cell, corr_per_sample):
    corr = pd.DataFrame(data=np.zeros((corr_per_cell.shape[1], 4)), index=corr_per_cell.columns,
                       columns=['Mean_corr_per_sample', 'Std_corr_per_sample', 'Mean_corr_per_cell', 'Std_corr_per_cell'])
    corr['Mean_corr_per_sample'] = np.mean(corr_per_sample, axis=0)
    corr['Std_corr_per_sample'] = np.std(corr_per_sample, axis=0)
    corr['Mean_corr_per_cell'] = np.mean(corr_per_cell, axis=0)
    corr['Std_corr_per_cell'] = np.std(corr_per_cell, axis=0)
    return corr




def flatten_corr_per_cell(corr_per_cell):
    corr_per_cell2 = pd.DataFrame(data=np.zeros((corr_per_cell.shape[0]*corr_per_cell.shape[1], 3)), 
                                  columns=['Method', 'Cell_type', 'Correlation'])
    methods = []
    for method in corr_per_cell.columns:
        methods.extend([method]*corr_per_cell.shape[0])
    corr_per_cell2['Method'] = methods
    corr_per_cell2['Cell_type'] = list(corr_per_cell.index)*corr_per_cell.shape[1]
    corr = []
    for method in corr_per_cell.columns:
        corr.extend(list(corr_per_cell[method]))
    corr_per_cell2['Correlation'] = corr

    return corr_per_cell2




def predicted_truth_bycell(method, method_freqs, exp_freqs, cell_types):
    df = pd.concat([method_freqs[cell_types], exp_freqs[cell_types]])
    df['Method'] = [method]*exp_freqs.shape[0] + ['Ground truth']*exp_freqs.shape[0]
    df = pd.DataFrame(data=np.zeros((exp_freqs.shape[0]*len(cell_types), 3)), 
                              columns=[method, 'Ground truth', 'Cell type'])
    method_fractions, true_fractions, cell_list = [], [], []
    for cell in cell_types:
        method_fractions.extend(list(method_freqs[cell]))
        true_fractions.extend(list(exp_freqs[cell]))
        cell_list.extend([cell]*exp_freqs.shape[0])
    df[method] = method_fractions
    df['Ground truth'] = true_fractions
    df['Cell type'] = cell_list
    
    return df
    
    
    
    
def stack_barchart(methods, results, true_freqs, cell_types, colors, fig_size, fig_name):
    sample_labels = np.arange(1,len(true_freqs)+1)
    
    fig, axs = plt.subplots(1, len(methods)+1, sharey=True, figsize=fig_size)
    # Remove vertical space between axes
    fig.subplots_adjust(wspace=0.1)
    
    for i in range(len(methods)):
        sns.set_style("white")
        results[methods[i]][cell_types].plot.barh(ax = axs[i], stacked=True, color=colors, width=1)
        axs[i].get_legend().remove()
        axs[i].set_yticklabels(sample_labels)
        axs[i].set_title(methods[i])
        axs[i].set_ylabel('Sample')
        if methods[i] in ['ssGSEA', 'singscore']:
            axs[i].set_xlabel('Score')
        else:
            axs[i].set_xlabel('Frequency')
    
    sns.set_style("white")
    true_freqs[cell_types].plot.barh(ax = axs[-1], stacked=True, color=colors, width=1)
    axs[-1].legend(bbox_to_anchor=(1.04,0.5), loc="center left")
    axs[-1].set_yticklabels(sample_labels)
    axs[-1].set_xlabel('Frequency')
    axs[-1].set_title('Ground_truth')
    
    
    
    
def corr_boxplot(methods, p_corr_per_sample, s_corr_per_sample, p_corr_per_cell, s_corr_per_cell, fig_size, this_color, filename):
    fig, axs = plt.subplots(1, 4, sharey=False, figsize=(22,4))
    # Remove vertical space between axes
    fig.subplots_adjust(wspace=0.2)
    
    for i, df in enumerate([p_corr_per_sample, s_corr_per_sample]):
        sns.boxplot(ax=axs[i], data=df, color='white')
        sns.swarmplot(ax=axs[i], data=df, color=".45")
        axs[i].set_xticklabels(methods, rotation=17)
        axs[i].set_ylim(-0.05, 1.05)
        if i == 0:
            axs[i].set_ylabel('Correlation with true fractions')
            axs[i].set_title('Pearson sample-level')
        else: 
            axs[i].set_title('Spearman sample-level')
            
    p_corr_per_cell2 = flatten_corr_per_cell(p_corr_per_cell)
    s_corr_per_cell2 = flatten_corr_per_cell(s_corr_per_cell)
    
    for i, df in enumerate([(p_corr_per_cell, p_corr_per_cell2), (s_corr_per_cell, s_corr_per_cell2)]):
        sns.boxplot(ax=axs[i+2], data=df[0], color='white')
        sns.swarmplot(ax=axs[i+2], x='Method', y='Correlation', hue='Cell_type', data=df[1], palette=this_color)
        if i == 0:
            axs[i+2].get_legend().remove()
        else:
            axs[i+2].legend(bbox_to_anchor=(1.04,0.5), loc="center left")
        axs[i+2].set_xticklabels(methods, rotation=17)
        axs[i+2].set_ylim(-0.05, 1.05)
        axs[i+2].set_xlabel('')
        axs[i+2].set_ylabel('')
        if i == 0:
            axs[i+2].set_title('Pearson cell-level')
        else:
            axs[i+2].set_title('Spearman cell-level')
    
    fig.subplots_adjust(bottom=0.2)
    
    
    
def corr_lineplot(p_per_sample_df, p_per_cell_df, s_per_sample_df, s_per_cell_df, fig_size):
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, sharey=False, figsize=fig_size)
    
    p_per_sample_df.plot.line(ax=ax1, marker='o', markersize=5)
    ax1.get_legend().remove()
    ax1.set_ylabel('Mean Pearson correlation per sample')
    ax1.set_ylim(0,1)
    ax1.set_xlabel('Signal to Noise ratio')
    ax1.set_xticklabels(['100:'+str(i*10) for i in range(6)])
    
    p_per_cell_df.plot.line(ax=ax3, marker='o', markersize=5)
    ax3.get_legend().remove()
    ax3.set_ylabel('Mean Pearson correlation per cell')
    ax3.set_ylim(0,1)
    ax3.set_xlabel('Signal to Noise ratio')
    ax3.set_xticklabels(['100:'+str(i*10) for i in range(6)])
    
    s_per_sample_df.plot.line(ax=ax2, marker='o', markersize=5)
    ax2.get_legend().remove()
    ax2.set_ylabel('Mean Spearman correlation per sample')
    ax2.set_ylim(0,1)
    ax2.set_xlabel('Signal to Noise ratio')
    ax2.set_xticklabels(['100:'+str(i*10) for i in range(6)])
    
    s_per_cell_df.plot.line(ax=ax4, marker='o', markersize=5)
    ax4.get_legend().remove()
    ax4.set_ylabel('Mean Spearman correlation per cell')
    ax4.set_ylim(0,1)
    ax4.set_xlabel('Signal to Noise ratio')
    ax4.set_xticklabels(['100:'+str(i*10) for i in range(6)])
    
    handles, labels = ax4.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.07), ncol=5, fontsize='large')
    fig.subplots_adjust(bottom=0.15)
    
