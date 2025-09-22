#!/usr/bin/env python3
import torch
from ax.fb.utils.storage.manifold import AEManifoldUseCase
from ax.fb.utils.storage.manifold_torch import AEManifoldTorchClient

from fblearner.flow.projects.ae.benchmarks.pfn.thirdparty.PFNs.pfns import (
    bar_distribution,
    encoders,
)
from fblearner.flow.projects.ae.benchmarks.pfn.thirdparty.PFNs.pfns.positional_encodings import (
    NoPositionalEncoding,
)
from fblearner.flow.projects.ae.benchmarks.pfn.thirdparty.PFNs.pfns.transformer import (
    TransformerModel,
)

# works with https://github.com/automl/PFNs/commit/fd212b187a22d5a07959484d03fcd82e060e9b21

STATE_DICT_KEY = {
    "hebo": "tree/trbo_dev/f380d01c-9861-11ef-9e69-84160c3811b6",
    "bnn": "tree/trbo_dev/0e6ad5d4-9862-11ef-b27f-84160c3811b6",
}


def load_bnn_pfn():
    # Load BNN PFN
    num_features = 18
    emsize = 512
    encoder_generator = encoders.get_normalized_uniform_encoder(
        encoders.get_variable_num_features_encoder(encoders.Linear)
    )
    encoder = encoder_generator(num_features, emsize)
    seq_len = 60

    bnn_model = TransformerModel(
        encoder,
        ninp=emsize,
        nhead=4,
        nhid=1024,
        nlayers=6,
        style_encoder=None,
        y_encoder=encoders.Linear(1, emsize),
        pos_encoder=NoPositionalEncoding(emsize, seq_len * 2),
        decoder_dict={"standard": (None, 1000)},
    )
    # check if it's full or not
    bnn_model.criterion = bar_distribution.FullSupportBarDistribution(
        torch.arange(1001).float()
    )
    bnn_model.decoder = bnn_model.decoder_dict.standard

    client = AEManifoldTorchClient(AEManifoldUseCase.TRBO_DEV)
    bnn_state_dict = client.torch_load(STATE_DICT_KEY["bnn"])
    bnn_model.load_state_dict(bnn_state_dict)

    bnn_model.borders = bnn_model.criterion.borders
    return bnn_model


def load_hebo_pfn():
    # Load HEBO+ PFN
    num_features = 18
    emsize = 512
    encoder_generator = encoders.get_normalized_uniform_encoder(
        encoders.get_variable_num_features_encoder(encoders.Linear)
    )
    encoder = encoder_generator(num_features, emsize)
    seq_len = 60

    hebo_model = TransformerModel(
        encoder,
        ninp=emsize,
        nhead=4,
        nhid=1024,
        nlayers=12,
        style_encoder=None,
        y_encoder=encoders.Linear(1, emsize),
        pos_encoder=NoPositionalEncoding(emsize, seq_len * 2),
        decoder_dict={"standard": (None, 1000)},
    )
    # check if it's full or not
    hebo_model.criterion = bar_distribution.FullSupportBarDistribution(
        torch.arange(1001).float()
    )

    client = AEManifoldTorchClient(AEManifoldUseCase.TRBO_DEV)
    hebo_state_dict = client.torch_load(STATE_DICT_KEY["hebo"])
    hebo_model.load_state_dict(hebo_state_dict)

    hebo_model.borders = hebo_model.criterion.borders
    return hebo_model
