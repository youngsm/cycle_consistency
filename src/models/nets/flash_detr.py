import torch
import torch.nn as nn
from ..components.transformer import make_transformer
from ..components.mlp import Mlp
from .flash_reco import FlashRecoModel

class FlashDETR(FlashRecoModel):
    def __init__(
        self,
        num_pmts: int = 180,
        num_flash_queries: int = 10,
        pe_num_ticks: int = 1000,
        sigma: float = 0.1,
        tokenizer_linear: str = "sine",
        regression_linear: str = "sine",
        encoder_transformer_kwargs: dict = {},
        decoder_transformer_kwargs: dict = {},
        tokenizer_hidden_features: int = 1024,
    ):
        super().__init__()

        self.num_pmts = num_pmts
        self.num_flash_queries = num_flash_queries
        self.pe_num_ticks = pe_num_ticks
        self.sigma = sigma

        self.wvfm_encoder = make_transformer(**encoder_transformer_kwargs)
        self.flash_decoder = make_transformer(**decoder_transformer_kwargs)
        self.embed_dim = self.wvfm_encoder.embed_dim
        self.tokenizer = Mlp(
            in_features=self.pe_num_ticks,  # ticks
            hidden_features=tokenizer_hidden_features,
            out_features=self.embed_dim,
            linear_layer=tokenizer_linear,
        )
        self.pos_tokens = self.make_learned_tokens((num_pmts, self.embed_dim))
        self.flash_q = self.make_learned_tokens((num_flash_queries, self.embed_dim))
        self.linear_pe = nn.Linear(self.embed_dim, self.num_pmts)
        self.linear_t = nn.Linear(self.embed_dim, 1)
        self.linear_c = nn.Linear(self.embed_dim, 1)

    @staticmethod
    def make_learned_tokens(shape, init_std=0.02):
        return nn.Parameter(torch.randn(*shape) * init_std)

    def forward(self, w):
        # waveforms w: (batch_size, num_pmts, num_ticks)

        # tokenize waveforms
        tok = self.tokenizer(w)

        # encode waveforms w/ transformer
        enc_out = self.wvfm_encoder(tok, self.pos_tokens.unsqueeze(0)).last_hidden_state

        # decode flashes w/ transformer
        dec_in = torch.cat([self.flash_q.unsqueeze(0).repeat(enc_out.shape[0], 1, 1), enc_out], dim=1)
        dec_pos = torch.cat([torch.zeros_like(self.flash_q), self.pos_tokens], dim=0)
        dec_out = self.flash_decoder(
            dec_in,
            dec_pos.unsqueeze(0),
        ).last_hidden_state[:, : self.num_flash_queries, :]

        # decode pe
        pred_pe = self.linear_pe(dec_out).clamp(-10, 10).exp()
        pred_t = self.linear_t(dec_out).sigmoid()
        pred_c = self.linear_c(dec_out).sigmoid()
        pred_pe_weighted = pred_pe * pred_c

        return {
            "pred_pe": pred_pe,
            "pred_t": pred_t,
            "pred_c": pred_c,
            "pred_pe_weighted": pred_pe_weighted,
        }
