import matplotlib.pyplot as plt
import plotly_express as px
import pandas as pd
import numpy as np

def draw_bar_plot_heads_counts(unique, counts):
    f,ax = plt.subplots(figsize=(12, 7))
    plt.bar(unique, counts)
    plt.xlabel('number of heads')
    plt.ylabel('frequency')
    plt.title('Frequency of Number of Heads')
    plt.show()
    
def draw_step_plot_head_probabilities(unique, probabilities, what_we_get=None, probability_of_what_we_get=None):
    f,ax = plt.subplots(figsize=(12, 7))
    plt.step(unique, probabilities)
    if what_we_get is not None and probability_of_what_we_get is not None:
        plt.axvline(what_we_get, c='red')
        plt.text(
            what_we_get + 0.3, 0.12, 
            s='this is {}, with prob: {:.2%}'.format(what_we_get, probability_of_what_we_get), 
            color='red'
        )

    plt.xlabel('number of heads')
    plt.ylabel('probability')
    plt.title('Number of Heads Distribution')
    plt.show()
    
    
def draw_ctp_happening_binned(ctp_current, bins, ctp_bin_indexes, ctp_bin_counts, ctp_current_bin):
    current_ctp_count = ctp_bin_counts[ctp_bin_indexes == ctp_current_bin][0]
    prob_ctp_current = current_ctp_count / ctp_bin_counts.sum()
    
    f, ax = plt.subplots(figsize=(20, 7))

    plt.subplot(121)
    plt.title('Frequency of CTP Happening (Binned)')
    plt.bar(ctp_bin_indexes, ctp_bin_counts, width=1, alpha=0.5)
    plt.axvline(ctp_current_bin, color='red')
    plt.annotate(
        'current ctp: {:.2f}'.format(ctp_current),
        (ctp_current_bin, current_ctp_count),
        xytext=(ctp_current_bin + 0.5, current_ctp_count + 20),
        arrowprops=dict(
            arrowstyle='->'

        )
    )
    plt.xticks(ctp_bin_indexes, ['{:.2f}'.format(x) for x in bins[ctp_bin_indexes]])
    plt.xlabel('CTP')
    plt.ylabel('frequency')


    plt.subplot(122)
    plt.title('Probability of CTP Happening (Binned)')
    plt.bar(ctp_bin_indexes, ctp_bin_counts/ctp_bin_counts.sum(), width=1, alpha=0.8)
    plt.axvline(ctp_current_bin, color='red')
    plt.annotate(
        'current ctp prob: {:.2f}'.format(prob_ctp_current),
        (ctp_current_bin, prob_ctp_current),
        xytext=(ctp_current_bin + 0.2, prob_ctp_current + 0.1),
        arrowprops=dict(
            arrowstyle='->',
            connectionstyle='angle3'

        )
    )
    plt.xticks(ctp_bin_indexes, ['{:.2f}'.format(x) for x in bins[ctp_bin_indexes]])
    plt.xlabel('CTP')
    plt.ylabel('p(CTP)')

    plt.show()


def make_df_for_simulations(simulate_ctp_fn, n_simulations=[20, 50, 100, 250, 500, 1000, 2000, 3000]):
    sim_res = []
    for e, sim in enumerate(n_simulations):
        ctps = simulate_ctp_fn(1000, 200, n_simulations=sim)
        bins = np.linspace(0.7, 1, 20)
        ctp_bins = np.digitize(ctps, bins)
        ctp_bin_indexes, ctp_bin_counts = np.unique(ctp_bins, return_counts=True)
        ctp_bin_labels = bins[ctp_bin_indexes]
        
        sim_res.append(
            np.vstack([
                ctp_bin_labels,
                ctp_bin_counts,
                [e for x in range(len(ctp_bin_counts))],
                [sim for x in range(len(ctp_bin_counts))]
            ]).T
        )

    return pd.DataFrame(
        np.vstack(sim_res), 
        columns=['ctp_bin_labels', 'ctp_bin_counts', 'sim_index', 'simulations']
    )

def draw_interactive_bar_plot_for_simulations(df):
    return px.bar(
        df, 
        x='ctp_bin_labels', 
        y='ctp_bin_counts', 
        animation_frame='sim_index', 
        range_y=[0,1300]
    )

def draw_interactive_prob_bar_plot_for_simulations(df):
    _df = df.copy()
    _df['ctp_bin_prob'] = _df['ctp_bin_counts'] / _df['simulations']

    return px.bar(
        _df, 
        x='ctp_bin_labels', 
        y='ctp_bin_prob', 
        animation_frame='sim_index', 
        range_y=[0,0.4]
    )
