import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LogNorm

def plot_waveform_predictions(model_output, truth, batch_idx=0, log=False, downsample=1):
    # plot with imshow:
    # - truth 
    # - model_output['pred_pe_weighted']
    # - the residual between the two
    pe_pred = model_output['aggregated_pe'][batch_idx].detach().cpu()
    pe_truth = truth[batch_idx].detach().cpu()

    if downsample > 1:
        pe_pred = pe_pred.view(pe_pred.shape[0], -1, downsample).sum(-1)
        pe_truth = pe_truth.view(pe_truth.shape[0], -1, downsample).sum(-1)

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    axs[0].imshow(pe_truth, aspect='auto', interpolation='none', norm=LogNorm() if log else None)
    axs[0].set_title('Truth')
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('PMT ID')
    axs[1].imshow(pe_pred, aspect='auto', interpolation='none', norm=LogNorm() if log else None)
    axs[1].set_title('Prediction')
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('PMT ID')
    axs[2].imshow((pe_pred-pe_truth).abs(), aspect='auto', interpolation='none', norm=LogNorm() if log else None)
    axs[2].set_title('Residual')
    axs[2].set_xlabel('Time')
    axs[2].set_ylabel('PMT ID')
    return fig
