import math
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Module

from torch.nn.modules.transformer import (
    _get_activation_fn,
    Dropout,
    LayerNorm,
    Linear,
    Module,
    MultiheadAttention,
    Optional,
    Tensor,
)
from .positional_encodings import RotaryPositionalEmbeddings
from .utils import bool_mask_to_att_mask, SeqBN


class HierarchicalTaskEncoder(Module):
    r"""TransformerEncoder is a stack of N encoder layers

    Args:
        encoder_layer_creator: a function generating objects of TransformerEncoderLayer class without args (required).
        nlayers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).
    """

    __constants__ = ["norm"]

    def __init__(
        self,
        global_with_target_points=False,
        local_with_target_points=False,
        **kwargs):
        super().__init__()
        
        global_kwargs = kwargs.copy()
        global_kwargs["global_with_target_points"] = global_with_target_points
        global_kwargs.pop("nlayers")
        
        single_kwargs = kwargs.copy()
        single_kwargs["local_with_target_points"] = local_with_target_points
        single_kwargs.pop("nlayers")
        
        layers = []
        for i in range(kwargs.get("nlayers", 24)):
            if (i+1) % 2 == 1:
                layers.append(GlobalTaskEncoderLayer(**global_kwargs))
            else:
                layers.append(SingleTaskEncoderLayer(**single_kwargs))
                
        self.layers = nn.ModuleList(layers)
        self.meta_embedding = nn.Embedding(1, kwargs.get("d_model", 512))
        
        self.norm = kwargs.get("norm", None)
        self.meta_tokens = kwargs.get("meta_tokens", 1)

    def forward(
        self,
        src: Tensor,
        task_ids: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """        
        # all task ids should be the same across batches
        assert (task_ids.min(dim=1).values == task_ids.max(dim=1).values).all()
        squeezed_task_id = task_ids[:, 0] # N x 1
        unique_task_ids = torch.unique(squeezed_task_id)
        
        # add meta-token task id for each task in front of the sequence
        meta_task_ids_summary = unique_task_ids.repeat_interleave(self.meta_tokens)
        meta_task_ids = torch.cat([meta_task_ids_summary.unsqueeze(-1), squeezed_task_id], dim=0)
        
        # add meta-token embeddings
        meta_output = self.meta_embedding(torch.zeros(1, device=src.device).long())
        meta_output = meta_output.repeat(len(meta_task_ids_summary), src.shape[1], 1)
        meta_output = torch.cat([meta_output, src], dim=0)
        
        # move eval position to account for meta-tokens
        src_mask = src_mask + len(unique_task_ids) * self.meta_tokens
        for mod in self.layers:
            meta_output = mod(
                meta_output, meta_task_ids, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask
            )

        if self.norm is not None:
            meta_output = self.norm(meta_output)

        return meta_output[self.meta_tokens * len(unique_task_ids):]


class SingleTaskEncoderLayer(Module):
            
    def __init__(
        self,
        meta_tokens,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        layer_norm_eps=1e-5,
        local_with_target_points=False,
        batch_first=False,
        pre_norm=False,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
    
        self.pre_norm = pre_norm
        
        self.attn = MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first, **factory_kwargs
        )
        self.norm = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout = Dropout(dropout)
        
        self.ff_norm = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.linear1 = Linear(d_model, dim_feedforward, **factory_kwargs)
        self.linear2 = Linear(dim_feedforward, d_model, **factory_kwargs)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        
        self.activation = _get_activation_fn(activation)
        
        # each source task attends to itself and the target task
        self.local_with_target_points = local_with_target_points
        if local_with_target_points:
            self.source_embedding = nn.Embedding(2, d_model) # embed target vs source
        
    def single_task_forward(
        self,
        src: Tensor,
        single_eval_position: Optional[Tensor] = None,
    ) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        src_ = src
        
        if self.pre_norm:
            src_ = self.norm(src)
        else:
            src_ = src
            
        # have each task attend within itself
        src_train = src_[:single_eval_position]
        src_eval = src_[single_eval_position:]
        src_left = self.attn(src_train, src_train, src_train)[0]
        src_right = self.attn(src_eval, src_train, src_train)[0]
        src2 = torch.cat([src_left, src_right], dim=0)
        
        src = src + self.dropout(src2)
        if not self.pre_norm:
            src = self.norm(src)            

        # feedforward
        if self.pre_norm:
            src_ff = self.ff_norm(src)
        else:
            src_ff = src
        src_ff = self.linear2(self.dropout1(self.activation(self.linear1(src_ff))))
        src = src + self.dropout2(src_ff)
        if not self.pre_norm:
            src = self.ff_norm(src)    
            
        return src
    
    def forward(
        self,
        src: Tensor,
        task_ids: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        assert isinstance(src_mask, int)
        single_eval_position = src_mask
        
        single_tasks_encoding = torch.zeros_like(src)
        train_task_id = task_ids[:single_eval_position]
        train_x = src[:single_eval_position]
        
        # attend to source and target task points
        if self.local_with_target_points:
            target_task_mask = (task_ids == 0).squeeze()
            target_embedding = self.source_embedding(torch.zeros(1, device=src.device).long())
            src[target_task_mask] = src[target_task_mask] + target_embedding
            source_embedding = self.source_embedding(torch.ones(1, device=src.device).long())
            src[~target_task_mask] = src[~target_task_mask] + source_embedding
        
        # Optimize this when scaling
        for task in torch.unique(task_ids):
            source_task_mask = (task_ids == task).squeeze()
            if self.local_with_target_points:
                # each task attends to itself and the target task
                task_mask = target_task_mask | source_task_mask
            else:
                task_mask = source_task_mask
            task_src = src[task_mask]
            
            train_task_mask = (train_task_id == task).squeeze()
            task_train_length = len(train_x[train_task_mask])
            
            task_single_tasks_encoding = self.single_task_forward(task_src, task_train_length)
            single_tasks_encoding[task_mask] = task_single_tasks_encoding
        
        return single_tasks_encoding


class GlobalTaskEncoderLayer(Module):
            
    def __init__(
        self,
        meta_tokens,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        layer_norm_eps=1e-5,
        batch_first=False,
        pre_norm=False,
        device=None,
        dtype=None,
        task_position_embedding=False,
        recompute_attn=False,
        save_trainingset_representations=False,
        global_with_target_points=False,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.pre_norm = pre_norm
        
        meta_d_model = d_model * meta_tokens
        self.attn = MultiheadAttention(
            meta_d_model, nhead, dropout=dropout, batch_first=batch_first, **factory_kwargs
        )
        self.norm = LayerNorm(meta_d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout = Dropout(dropout)
        
        self.ff_norm = LayerNorm(meta_d_model, eps=layer_norm_eps, **factory_kwargs)
        self.linear1 = Linear(meta_d_model, dim_feedforward, **factory_kwargs)
        self.linear2 = Linear(dim_feedforward, meta_d_model, **factory_kwargs)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        
        self.activation = _get_activation_fn(activation)
        self.meta_tokens = meta_tokens
        
        self.global_with_target_points = global_with_target_points
        
    def forward(
        self,
        src: Tensor,
        task_ids: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        assert isinstance(src_mask, int)
        assert src_key_padding_mask is None
        
        meta_end = self.meta_tokens * len(torch.unique(task_ids))
        
        # get summary tokens with T x B x (M x E), add summary embedding
        summary_src = src[:meta_end].reshape(len(torch.unique(task_ids)), src.shape[1], -1)

        if self.global_with_target_points:
            # make global layer also attend to target points
            task_src = src[(task_ids == 0).squeeze()][self.meta_tokens:].repeat(1, 1, self.meta_tokens)
            global_src_ = torch.cat([summary_src, task_src], dim=0)
        else:
            # only attend to summary tokens
            global_src_ = summary_src
        
        if self.pre_norm:
            global_src_ = self.norm(global_src_)
            summary_src_ = self.norm(summary_src)
        else:
            global_src_ = global_src_
            
        summary_src_ = self.attn(summary_src, global_src_, global_src_)[0]
        
        summary_src_ = summary_src + self.dropout(summary_src_)
        if not self.pre_norm:
            summary_src_ = self.norm(summary_src_)

        # feedforward
        if self.pre_norm:
            summary_src_ff = self.ff_norm(summary_src_)
        else:
            summary_src_ff = summary_src_
        summary_src_ff = self.linear2(self.dropout1(self.activation(self.linear1(summary_src_ff))))
        summary_src_ = summary_src_ + self.dropout2(summary_src_ff)
        if not self.pre_norm:
            summary_src_ = self.ff_norm(summary_src_)
            
        # reshape back
        summary_src = summary_src_.reshape(-1, src.shape[1], src.shape[2])
            
        original_tokens = src[meta_end:]
        combined_src = torch.cat([summary_src, original_tokens], dim=0)
        return combined_src