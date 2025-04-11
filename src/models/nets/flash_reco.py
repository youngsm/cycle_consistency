import torch
import torch.nn as nn

class FlashRecoModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.pe_num_ticks: int
        self.sigma: float

    def forward(self, w):
        pass

    def forward_with_aggregation(self, w):
        preds = self.forward(w)

        preds['aggregated_pe'] = aggregate_flashes(
            preds['pred_pe'],
            preds['pred_t'],
            preds['pred_c'],
            self.pe_num_ticks,
            self.sigma
        )

        return preds

def aggregate_flashes(flashes_pe, flashes_time, flashes_confidence, num_ticks, sigma):
    """
    Differentiable implementation using fully vectorized operations without loops
    Supports batched inputs
    """
    batch_size, num_flashes, num_pmts = flashes_pe.shape
    device = flashes_pe.device

    time_bins = torch.clamp(
        flashes_time * num_ticks, min=0, max=num_ticks - 1
    ).squeeze(-1)  # (batch, num_flashes)

    time_indices = torch.arange(num_ticks, device=device)
    time_bins_expanded = time_bins.unsqueeze(-1)  # (batch, num_flashes, 1)
    time_indices_expanded = time_indices.reshape(1, 1, num_ticks)  # (1, 1, num_ticks)

    # convert individual times to time bins.
    time_weights = torch.exp(
        -0.5 * ((time_indices_expanded - time_bins_expanded) / sigma) ** 2
    )  # (batch, num_flashes, num_ticks)
    time_weights = time_weights / (
        time_weights.sum(dim=2, keepdim=True) + 1e-10
    )  # normalize along time

    scaled_pe = flashes_pe * flashes_confidence.squeeze(-1).unsqueeze(-1)

    scaled_pe_expanded = scaled_pe.unsqueeze(-1)  # (batch, num_flashes, num_pmts, 1)
    time_weights_expanded = time_weights.unsqueeze(2)  # (batch, num_flashes, 1, num_ticks)

    contribution = (
        scaled_pe_expanded * time_weights_expanded
    )  # (batch, num_flashes, num_pmts, num_ticks)

    waveform = contribution.sum(dim=1)  # (batch, num_pmts, num_ticks)

    return waveform