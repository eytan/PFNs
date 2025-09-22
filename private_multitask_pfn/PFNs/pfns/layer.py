from functools import partial

import torch
from torch import nn
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

from torch.utils.checkpoint import checkpoint


class AxialAttentionModule(Module):
    r"""
    AxialAttention is a module that applies self-attention to rows and columns of a 2D input tensor.
    """

    __constants__ = ["batch_first"]

    def __init__(
        self,
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
        recompute_attn=False,
        save_trainingset_representations=False,
        sequential_axial=False,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        
        self.seq_axial = sequential_axial
        self.pre_norm = pre_norm
        
        self.self_attn_row = MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first, **factory_kwargs
        )
        self.row_norm = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.row_dropout = Dropout(dropout)
        self.self_attn_col = MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first, **factory_kwargs
        )
        self.col_norm = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.col_dropout = Dropout(dropout)
        
    def __setstate__(self, state):
        if "activation" not in state:
            state["activation"] = F.relu
        super().__setstate__(state)
        self.__dict__.setdefault("save_trainingset_representations", False)

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
        N, D, B, E = src.shape
        
        assert isinstance(src_mask, int)
        assert src_key_padding_mask is None
        single_eval_position = src_mask
        
        def self_attn_forward(attn_module, src, row_first):                
            if row_first:
                src = src.reshape(N, D*B, E)
                attn_mask = torch.zeros(N, N, device=src.device).bool()
                attn_mask[single_eval_position:, single_eval_position:] = True
            else:
                # column first
                src = src.permute(1, 0, 2, 3).reshape(D, N*B, E)    
                attn_mask = torch.zeros(D, D, device=src.device).bool()
                            
            src_full = attn_module(
                src,
                src,
                src,
                attn_mask=attn_mask,
            )[0]
            
            if row_first:
                return src_full.reshape(N, D, B, E)
            else:
                src_full = src_full.reshape(D, N, B, E).permute(1, 0, 2, 3)
            return src_full.reshape(N, D, B, E)
        
        # attend to rows
        if self.pre_norm:
            src_row = self.row_norm(src)
        else:
            src_row = src
        src_row = self_attn_forward(self.self_attn_row, src_row, row_first=True)
        src_row = self.row_dropout(src_row)
        
        # pass row as input to col
        if self.seq_axial:
            src = src + src_row
            if not self.pre_norm:
                src = self.row_norm(src)
            
        # attend to cols
        if self.pre_norm:
            src_col = self.col_norm(src)
        else:
            src_col = src
        src_col = self_attn_forward(self.self_attn_col, src_col, row_first=False)
        src_col = self.col_dropout(src_col)
        
        if self.seq_axial:
            src = src + src_col
            if not self.pre_norm:
                src = self.col_norm(src)
        else:
            # combine row and col
            src = src + src_row + src_col
            if not self.pre_norm:
                src = self.col_norm(src)

        return src


class BasicTaskAttentionLayer(Module):
    def __init__(
        self,
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
        recompute_attn=False,
        save_trainingset_representations=False,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        
        self.pre_norm = pre_norm
        
        self.attn = MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first, **factory_kwargs
        )
        self.norm = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout = Dropout(dropout)

    def __setstate__(self, state):
        if "activation" not in state:
            state["activation"] = F.relu
        super().__setstate__(state)
        self.__dict__.setdefault("save_trainingset_representations", False)

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
        single_eval_position = src_mask
        
        if self.pre_norm:
            src_ = self.norm(src)
        else:
            src_ = src
            
        src_train = src_[:single_eval_position]
        src_eval = src_[single_eval_position:]
        
        src_left = self.attn(
            src_train,
            src_train,
            src_train,
        )[0]
        src_right = self.attn(
            src_eval, src_train, src_train
        )[0]
        src2 = torch.cat([src_left, src_right], dim=0)
        
        src = src + self.dropout(src2)
        if not self.pre_norm:
            src = self.norm(src)
            
        return src
    
    

class TaskAttention(Module):
    def __init__(
        self,
        num_tasks,
        shared,
        optimized,
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
        recompute_attn=False,
        save_trainingset_representations=False,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        
        self.pre_norm = pre_norm
        
        self.attn = MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first, **factory_kwargs
        )
        self.norm = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout = Dropout(dropout)
        
        self.optimized = optimized
        self.num_tasks = num_tasks
        
        kwargs = {
            "d_model": d_model,
            "nhead": nhead,
            "dim_feedforward": dim_feedforward,
            "dropout": dropout,
            "activation": activation,
            "layer_norm_eps": layer_norm_eps,
            "batch_first": batch_first,
            "pre_norm": pre_norm,
            "device": device,
            "dtype": dtype,
            "recompute_attn": recompute_attn,
            "save_trainingset_representations": save_trainingset_representations,
        }
            
        if not self.optimized:
            if shared:
                self.task_attn_module = BasicTaskAttentionLayer(**kwargs)
                self.task_attn_modules = [self.task_attn_module for _ in range(num_tasks)]
            else:
                self.task_attn_modules = nn.ModuleList([BasicTaskAttentionLayer(**kwargs) for _ in range(num_tasks)])
        else:
            self.task_attn = MultiheadAttention(
                d_model, nhead, dropout=dropout, batch_first=batch_first, **factory_kwargs
            )
            self.task_norm = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
            self.task_dropout = Dropout(dropout)

    def __setstate__(self, state):
        if "activation" not in state:
            state["activation"] = F.relu
        super().__setstate__(state)
        self.__dict__.setdefault("save_trainingset_representations", False)

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
        src_ = src
        if not self.optimized:
            combined_task_src = torch.zeros_like(src_)
            for task_id in range(self.num_tasks):
                task_src = src_[task_ids == task_id]
                task_id_src = task_ids[task_ids == task_id]
                combined_task_src[task_ids == task_id] = self.task_attn_modules[task_id](task_src, task_id_src, src_mask, src_key_padding_mask)
                
            src = combined_task_src
        else:
            single_eval_position = src_mask
            
            if self.pre_norm:
                src_ = self.task_norm(src_)
            
            # Train attends to train with same task
            src_train = src_[:single_eval_position]
            task_id_train = task_ids[:single_eval_position]
            # true when tasks are different
            train_src_mask = task_id_train[:, None] != task_id_train[None, :]         
            src_train, _ = self.task_attn(src_train, src_train, src_train, attn_mask=train_src_mask)
            
            # Eval attends to train with same task
            src_eval = src_[single_eval_position:]
            task_id_eval = task_ids[single_eval_position:]
            eval_src_mask = task_id_eval[:, None] != task_id_train[None, :]
            src_eval, _ = self.task_attn(src_eval, src_train, src_train, attn_mask=eval_src_mask)
            
            src2 = torch.cat([src_train, src_eval], dim=0)
            src = src + self.dropout(src2)
            
            if not self.pre_norm:
                src = self.norm(src)
            
        return src
            

class TransformerAttentionLayer(Module):
            
    def __init__(
        self,
        num_tasks,
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
        recompute_attn=False,
        save_trainingset_representations=False,
        attn_type="standard",
        task_embed_type="task_attn"
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        
        kwargs = {
            "d_model": d_model,
            "nhead": nhead,
            "dim_feedforward": dim_feedforward,
            "dropout": dropout,
            "activation": activation,
            "layer_norm_eps": layer_norm_eps,
            "batch_first": batch_first,
            "pre_norm": pre_norm,
            "device": device,
            "dtype": dtype,
            "recompute_attn": recompute_attn,
            "save_trainingset_representations": save_trainingset_representations,
        }
            
        self.num_tasks = num_tasks
        self.pre_norm = pre_norm
        
        if attn_type == "standard":
            if task_embed_type == "task_attn":
                self.task_attn = TaskAttention(num_tasks=num_tasks, shared=False, optimized=False, **kwargs)
            elif task_embed_type == "task_attn_shared":
                self.task_attn = TaskAttention(num_tasks=num_tasks, shared=True, optimized=False, **kwargs)
            elif task_embed_type == "task_attn_opt":
                self.task_attn = TaskAttention(num_tasks=num_tasks, shared=True, optimized=True, **kwargs)
            else:
                self.task_attn = None
                
            self.all_attn = BasicTaskAttentionLayer(**kwargs)
        elif "axial" in attn_type:
            sequential_axial = "parallel" not in attn_type
            args = *args, sequential_axial
            
            if task_embed_type == "task_attn":
                self.task_attn = nn.ModuleList([AxialAttentionModule(**kwargs) for _ in range(num_tasks)])
            elif task_embed_type == "task_attn_shared":
                self.task_attn_one = AxialAttentionModule(**kwargs)
                self.task_attn = [self.task_attn_one for _ in range(num_tasks)]
            else:
                self.task_attn = None
                
            self.all_attn = AxialAttentionModule(**kwargs)
        else:
            raise ValueError(f"Unknown attn_type: {attn_type}")
        
        self.ff_norm = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.linear1 = Linear(d_model, dim_feedforward, **factory_kwargs)
        self.linear2 = Linear(dim_feedforward, d_model, **factory_kwargs)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        
        self.recompute_attn = recompute_attn
        self.save_trainingset_representations = save_trainingset_representations
        self.saved_src_to_attend_to = None
        
        self.activation = _get_activation_fn(activation)
        
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
        
        src_ = src
            
        assert isinstance(src_mask, int)
        assert src_key_padding_mask is None
        
        # attend to all, contains skip connection
        all_src = self.all_attn(src_, task_ids, src_mask, src_key_padding_mask)
        
        # attend to tasks
        if self.task_attn is not None:
            assert torch.all(task_ids[:, 0] == task_ids.max(1)[0]) and torch.all(task_ids[:, 0] == task_ids.min(1)[0])
            task_ids = task_ids[:, 0].squeeze()
            
            task_src = self.task_attn(src_, task_ids, src_mask, src_key_padding_mask)
            
            # combine task and all
            src = all_src + task_src
        else:
            src = all_src

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
