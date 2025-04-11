import torch
import torch.nn as nn
import numpy as np

class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int = None,
        out_features: int = None,
        linear_layer: str = 'linear',
        act_layer: nn.Module = nn.GELU,
        drop: float = 0.0,
        outermost_linear: bool = False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        if linear_layer == 'sine':
            linear_layer = SineLayer
        else:
            linear_layer = nn.Linear

        kwargs = {"is_first": True, "omega_0": 30} if linear_layer == SineLayer else {}
        self.fc1 = linear_layer(in_features, hidden_features, **kwargs)
        self.act = act_layer()

        if not outermost_linear:
            kwargs = {"is_first": False, "omega_0": 30} if linear_layer == SineLayer else {}
            self.fc2 = linear_layer(hidden_features, out_features, **kwargs)
        else:
            self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop) if drop > 0.0 else nn.Identity()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class SineLayer(nn.Module):
    """
    A linear layer with the sinusoidal activation function, a typical layer in siren
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        is_first: bool = False,
        omega_0: float = 30
    ):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            else:
                self.linear.weight.uniform_(
                    -np.sqrt(6 / self.in_features) / self.omega_0,
                    np.sqrt(6 / self.in_features) / self.omega_0,
                )

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))


if __name__ == "__main__":
    _ = Mlp(128)
    _ = Mlp(128, linear_layer='sine')
