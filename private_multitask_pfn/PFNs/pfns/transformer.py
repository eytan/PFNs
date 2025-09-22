import math
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Module, TransformerEncoder

# from .layer import _get_activation_fn, TransformerEncoderLayer, TransformerAxialEncoderLayer, TransformerTaskEncoderLayer, TransformerAttentionLayer
from .layer import TransformerAttentionLayer
from .hierarchical import HierarchicalTaskEncoder
from .utils import bool_mask_to_att_mask, SeqBN


class TransformerModel(nn.Module):
    def __init__(
        self,
        encoder,
        ninp,
        nhead,
        nhid,
        nlayers,
        dropout=0.0,
        style_encoder=None,
        y_encoder=None,
        pos_encoder=None,
        decoder_dict=None,
        input_normalization=False,
        init_method=None,
        pre_norm=False,
        activation="gelu",
        recompute_attn=False,
        num_global_att_tokens=0,
        full_attention=False,
        all_layers_same_init=False,
        efficient_eval_masking=True,
        decoder_once_dict=None,
        return_all_outputs=False,
        save_trainingset_representations=False,
        num_features=None,
        num_tasks=None,
        attn_type="axial",
        task_encoder=None,
        task_embed_type="linear",
        hierarchical=False,
        **model_extra_args,
    ):
        super().__init__()
        self.model_type = "Transformer"
        if hierarchical:
            global_with_target_points = model_extra_args.get("global_with_target_points", False)
            local_with_target_points = model_extra_args.get("local_with_target_points", False)
            meta_tokens = model_extra_args.get("meta_tokens", 1)
            self.transformer_encoder = HierarchicalTaskEncoder(
                global_with_target_points=global_with_target_points,
                local_with_target_points=local_with_target_points,                
                nlayers=nlayers,
                meta_tokens=meta_tokens,
                d_model=ninp,
                nhead=nhead,
                dim_feedforward=nhid,
                dropout=dropout,
                activation=activation,
                pre_norm=pre_norm,
            )
        else:
            encoder_layer_creator = lambda: TransformerAttentionLayer(
                num_tasks,
                ninp,
                nhead,
                nhid,
                dropout,
                activation=activation,
                pre_norm=pre_norm,
                recompute_attn=recompute_attn,
                save_trainingset_representations=save_trainingset_representations,
                attn_type=attn_type,
                task_embed_type=task_embed_type,
            )
            self.transformer_encoder = TransformerTaskEncoder(encoder_layer_creator, nlayers)
        self.ninp = ninp
        self.encoder = encoder
        self.task_encoder = task_encoder
        self.task_embed_type = task_embed_type
        self.y_encoder = y_encoder
        self.pos_encoder = pos_encoder
        self.return_all_outputs = return_all_outputs
        self.num_features = num_features
        self.num_tasks = num_tasks
        self.attn_type = attn_type

        def make_decoder_dict(decoder_description_dict):
            if decoder_description_dict is None or len(decoder_description_dict) == 0:
                return None
            initialized_decoder_dict = {}
            for decoder_key in decoder_description_dict:
                decoder_model, decoder_n_out = decoder_description_dict[decoder_key]
                if decoder_model is None:
                    initialized_decoder_dict[decoder_key] = nn.Sequential(
                        nn.Linear(ninp, nhid), nn.GELU(), nn.Linear(nhid, decoder_n_out)
                    )
                else:
                    initialized_decoder_dict[decoder_key] = decoder_model
                # print(
                #     "Initialized decoder for",
                #     decoder_key,
                #     "with",
                #     decoder_description_dict[decoder_key],
                #     " and nout",
                #     decoder_n_out,
                # )
            return torch.nn.ModuleDict(initialized_decoder_dict)

        self.decoder_dict = make_decoder_dict(decoder_dict)
        self.decoder_dict_once = make_decoder_dict(decoder_once_dict)

        # N(0,1) is the initialization as the default of nn.Embedding
        self.decoder_dict_once_embeddings = (
            torch.nn.Parameter(torch.randn((len(self.decoder_dict_once), 1, ninp)))
            if self.decoder_dict_once is not None
            else None
        )
        # nn.Embedding(len(self.decoder_dict.keys()), nhid)
        self.input_ln = SeqBN(ninp) if input_normalization else None
        self.style_encoder = style_encoder
        self.init_method = init_method
        if num_global_att_tokens is not None:
            assert not full_attention
        self.global_att_embeddings = (
            nn.Embedding(num_global_att_tokens, ninp) if num_global_att_tokens else None
        )
        self.full_attention = full_attention
        self.efficient_eval_masking = efficient_eval_masking

        self.nhid = nhid

        self.init_weights()
        self.encode_x_y = None
        
        self.target_aware = model_extra_args.get("target_aware", False)
        self.target_encoder = nn.Embedding(2, ninp) if self.target_aware else None
        # self.from_tabpfn()
        
    def from_tabpfn(self):
        
        import einops

        from tabpfn.base import (
            create_inference_engine,
            determine_precision,
            initialize_tabpfn_model,
        )
        from tabpfn.model.loading import get_encoder, _preprocess_config, get_y_encoder
        
        import os
        model_path = "/home/lily_l/private_multitask_pfn/TabPFN/tabpfn-v2-regressor.ckpt"
        if not os.path.exists(model_path):
            model_path = "/home/yl9959/mtpfn/TabPFN/tabpfn-v2-regressor.ckpt"
            

        checkpoint = torch.load(model_path, map_location="cpu", weights_only=None)

        state_dict = checkpoint["state_dict"]
        config = _preprocess_config(checkpoint["config"])

        encoder=get_encoder(
            num_features=config.features_per_group,
            embedding_size=config.emsize,
            remove_empty_features=config.remove_empty_features,
            remove_duplicate_features=config.remove_duplicate_features,
            nan_handling_enabled=config.nan_handling_enabled,
            normalize_on_train_only=config.normalize_on_train_only,
            normalize_to_ranking=config.normalize_to_ranking,
            normalize_x=config.normalize_x,
            remove_outliers=config.remove_outliers,
            normalize_by_used_features=config.normalize_by_used_features,
            encoder_use_bias=config.encoder_use_bias,
        )
        encoder.load_state_dict({"5.layer.weight": state_dict["encoder.5.layer.weight"]})
        y_encoder=get_y_encoder(
            num_inputs=1,
            embedding_size=config.emsize,
            nan_handling_y_encoder=config.nan_handling_y_encoder,
            max_num_classes=config.max_num_classes,
        )
        y_encoder.load_state_dict({
            "1.layer.weight": state_dict['y_encoder.1.layer.weight'], 
            "1.layer.bias": state_dict['y_encoder.1.layer.bias']
        })
        self.encoder = encoder.to("cuda")
        self.y_encoder = y_encoder.to("cuda")
        
        def encoder_forward(x, y, single_eval_pos_):
            
            if isinstance(x, dict):
                assert "main" in set(x.keys()), f"Main must be in input keys: {x.keys()}."
            else:
                x = {"main": x}
                y = {"main": y}
            seq_len, batch_size, num_features = x["main"].shape


            for k in x:
                num_features_ = x[k].shape[2]

                # pad to multiple of features_per_group
                missing_to_next = (
                    config.features_per_group - (num_features_ % config.features_per_group)
                ) % config.features_per_group

                if missing_to_next > 0:
                    x[k] = torch.cat(
                        (
                            x[k],
                            torch.zeros(
                                seq_len,
                                batch_size,
                                missing_to_next,
                                device=x[k].device,
                                dtype=x[k].dtype,
                            ),
                        ),
                        dim=-1,
                    )

            # Splits up the input into subgroups
            for k in x:
                x[k] = einops.rearrange(
                    x[k],
                    "s b (f n) -> b s f n",
                    n=config.features_per_group,
                )  # s b f -> b s #groups #features_per_group
                

            for k in y:
                if y[k].ndim == 1:
                    y[k] = y[k].unsqueeze(-1)
                if y[k].ndim == 2:
                    y[k] = y[k].unsqueeze(-1)  # s b -> s b 1

                y[k] = y[k].transpose(0, 1)  # s b 1 -> b s 1
                # print(y[k].shape, x["main"].shape)

                if y[k].shape[1] < x["main"].shape[1]:
                    assert (
                        y[k].shape[1] == single_eval_pos_
                        or y[k].shape[1] == x["main"].shape[1]
                    )
                    assert k != "main" or y[k].shape[1] == single_eval_pos_, (
                        "For main y, y must not be given for target"
                        " time steps (Otherwise the solution is leaked)."
                    )
                    if y[k].shape[1] == single_eval_pos_:
                        y[k] = torch.cat(
                            (
                                y[k],
                                torch.nan
                                * torch.zeros(
                                    y[k].shape[0],
                                    x["main"].shape[1] - y[k].shape[1],
                                    y[k].shape[2],
                                    device=y[k].device,
                                    dtype=y[k].dtype,
                                ),
                            ),
                            dim=1,
                        )

                y[k] = y[k].transpose(0, 1)  # b s 1 -> s b 1

            # making sure no label leakage ever happens
            y["main"][single_eval_pos_:] = torch.nan

            embedded_y = y_encoder(
                y,
                single_eval_pos=single_eval_pos_,
            ).transpose(0, 1)



            for k in x:
                x[k] = einops.rearrange(x[k], "b s f n -> s (b f) n")
                        
            # embedded_x = encoder(x, single_eval_pos=single_eval_pos_)

            embedded_x = einops.rearrange(
                encoder(
                    x,
                    single_eval_pos=len(x["main"]),
                ),
                "s (b f) e -> b s f e",
                b=embedded_y.shape[0],
            )  # b s f 1 -> b s f e
            
            # mean over f dimension
            embedded_x = embedded_x.mean(dim=2)
            
            return embedded_x.transpose(0, 1) + embedded_y.transpose(0, 1)
            
        self.encode_x_y = encoder_forward


    def __setstate__(self, state):
        super().__setstate__(state)
        self.__dict__.setdefault("efficient_eval_masking", False)
        if not hasattr(self, "decoder_dict_once"):
            self.__dict__.setdefault("decoder_dict_once", None)
        if hasattr(self, "decoder") and not hasattr(self, "decoder_dict"):
            self.add_module("decoder_dict", nn.ModuleDict({"standard": self.decoder}))
        self.__dict__.setdefault("return_all_outputs", False)

        def add_approximate_false(module):
            if isinstance(module, nn.GELU):
                module.__dict__.setdefault("approximate", "none")

        self.apply(add_approximate_false)

    @staticmethod
    def generate_square_subsequent_mask(sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        return bool_mask_to_att_mask(mask)

    @staticmethod
    def generate_D_q_matrix(sz, query_size):
        train_size = sz - query_size
        mask = torch.zeros(sz, sz) == 0
        mask[:, train_size:].zero_()
        mask |= torch.eye(sz) == 1
        return bool_mask_to_att_mask(mask)

    @staticmethod
    def generate_global_att_query_matrix(
        num_global_att_tokens, seq_len, num_query_tokens
    ):
        train_size = seq_len + num_global_att_tokens - num_query_tokens
        sz = seq_len + num_global_att_tokens
        mask = torch.zeros(num_query_tokens, sz) == 0
        mask[:, train_size:].zero_()
        mask[:, train_size:] |= torch.eye(num_query_tokens) == 1
        return bool_mask_to_att_mask(mask)

    @staticmethod
    def generate_global_att_trainset_matrix(
        num_global_att_tokens, seq_len, num_query_tokens
    ):
        train_size = seq_len + num_global_att_tokens - num_query_tokens
        trainset_size = seq_len - num_query_tokens
        mask = torch.zeros(trainset_size, num_global_att_tokens) == 0
        # mask[:,num_global_att_tokens:].zero_()
        # mask[:,num_global_att_tokens:] |= torch.eye(trainset_size) == 1
        return bool_mask_to_att_mask(mask)

    @staticmethod
    def generate_global_att_globaltokens_matrix(
        num_global_att_tokens, seq_len, num_query_tokens
    ):
        mask = (
            torch.zeros(
                num_global_att_tokens,
                num_global_att_tokens + seq_len - num_query_tokens,
            )
            == 0
        )
        return bool_mask_to_att_mask(mask)


    def init_weights(self):
        try:
            # lily
            # nn.init.normal_(self.encoder[1].base_encoder.weight, mean=0, std=1.0 / math.sqrt(self.ninp))
            # if self.y_encoder is not None:
            #     nn.init.normal_(self.y_encoder.weight, mean=0, std=1.0 / math.sqrt(self.ninp))
            
            # lily
            for n, p in self.transformer_encoder.named_parameters():
                if "bias" in n:
                    nn.init.zeros_(p)
                elif "attn" in n:
                    nn.init.xavier_uniform_(p)
                elif "norm" in n:
                    nn.init.ones_(p)
                elif "linear1" in n:
                    nn.init.kaiming_normal_(p, nonlinearity='relu')
                elif "linear2" in n:
                    nn.init.xavier_uniform_(p)
                    
            # # decoder dict
            # for k, v in self.decoder_dict.items():
            #     for n, p in v.named_parameters():
            #         if "bias" in n:
            #             nn.init.zeros_(p)
            #         elif "weight" in n:
            #             nn.init.xavier_uniform_(p)
            #             import pdb; pdb.set_trace()
            #             print(p)
        except:
            self.init_weights_old()


    def init_weights_old(self):
        initrange = 1.0
        # if isinstance(self.encoder,EmbeddingEncoder):
        #    self.encoder.weight.data.uniform_(-initrange, initrange)
        # self.decoder.bias.data.zero_()
        # self.decoder.weight.data.uniform_(-initrange, initrange)
        if self.init_method is not None:
            self.apply(self.init_method)
        for layer in self.transformer_encoder.layers:
            nn.init.zeros_(layer.linear2.weight)
            nn.init.zeros_(layer.linear2.bias)
            if isinstance(layer, TransformerAttentionLayer):
                pass
                # nn.init.zeros_(layer.all_attn.out_proj.weight)
                # nn.init.zeros_(layer.all_attn.out_proj.bias)
                
                # TODO: update initialization
                
                
            # if isinstance(layer, TransformerEncoderLayer):
            #     attns = (
            #         layer.self_attn
            #         if isinstance(layer.self_attn, nn.ModuleList)
            #         else [layer.self_attn]
            #     )
            #     for attn in attns:
            #         nn.init.zeros_(attn.out_proj.weight)
            #         nn.init.zeros_(attn.out_proj.bias)
            # elif isinstance(layer, TransformerAxialEncoderLayer):
            #     nn.init.zeros_(layer.self_attn_col.out_proj.weight)
            #     nn.init.zeros_(layer.self_attn_col.out_proj.bias)

    def forward(self, *args, **kwargs):
        """
        This will perform a forward-pass (possibly recording gradients) of the model.
        We have multiple interfaces we support with this model:

        model(train_x, train_y, test_x, src_mask=None, style=None, only_return_standard_out=True)
        model((x,y), src_mask=None, single_eval_pos=None, only_return_standard_out=True)
        model((style,x,y), src_mask=None, single_eval_pos=None, only_return_standard_out=True)
        """
        if len(args) == 5:
            # case model(train_x, train_task_id, train_y, test_x, test_task_id=None, src_mask=None, style=None, only_return_standard_out=True)
            assert all(
                kwarg in {"src_mask", "style", "only_return_standard_out"}
                for kwarg in kwargs.keys()
            ), f"Unrecognized keyword argument in kwargs: {set(kwargs.keys()) - {'src_mask', 'style', 'only_return_standard_out'}}"
            x = args[0]
            task_id = args[1]
            y = args[2]
            # combine train and test x along sequence dimension
            if args[3] is not None: # test_x
                x = torch.cat((x, args[3]), dim=0)
                
                if args[4] is None: # test_task_id
                    # assume test_task_id is all zeros
                    test_task_id = torch.zeros(args[3].shape[:-1]).unsqueeze(-1).to(task_id)
                else:
                    test_task_id = args[4]
                task_id = torch.cat((task_id, test_task_id), dim=0)
            style = kwargs.pop("style", None)
            # forward(style, x, y),  "single_eval_pos" corresponds to where test x starts
            return self._forward(
                (style, x, y, task_id), single_eval_pos=len(args[0]), **kwargs
            )
        # during training (x, y, task_id) is passed with single_eval_pos
        elif len(args) == 1 and isinstance(args, tuple):
            # case model((x,y,task_id), src_mask=None, single_eval_pos=None, only_return_standard_out=True)
            # case model((style,x,y,task_id), src_mask=None, single_eval_pos=None, only_return_standard_out=True)
            assert all(
                kwarg in {"src_mask", "single_eval_pos", "only_return_standard_out"}
                for kwarg in kwargs.keys()
            ), f"Unrecognized keyword argument in kwargs: {set(kwargs.keys()) - {'src_mask', 'single_eval_pos', 'only_return_standard_out'}}"
            return self._forward(*args, **kwargs)

    def _forward(
        self, src, src_mask=None, single_eval_pos=None, only_return_standard_out=True
    ):
        assert isinstance(
            src, tuple
        ), "inputs (src) have to be given as (x,y,task_id) or (style,x,y,task_id) tuple"

        if len(src) == 3:  # (x,y,task_id) and no style
            src = (None,) + src

        style_src, x_src, y_src, task_id = src

        if single_eval_pos is None:
            single_eval_pos = x_src.shape[0]
                
        if self.encode_x_y is not None:
            src = self.encode_x_y(x_src, y_src, single_eval_pos)
        else:
            # x_src is N x B x D
            if self.task_embed_type == "feature" and task_id is not None:
                onehot_task_ids = torch.nn.functional.one_hot(
                    task_id.squeeze(-1), num_classes=self.num_tasks
                ).to(x_src)
                x_src = torch.cat([onehot_task_ids, x_src], dim=-1)
            
            
            
            x_src = self.encoder(x_src)
            if "axial" in self.attn_type:
                N, D, B, E = x_src.shape
            else:
                N, B, E = x_src.shape
                
            if task_id is not None and self.task_encoder is not None:
                assert task_id.shape == (N, B, 1)
                
                if self.task_embed_type == "linear":
                    task_id = self.task_encoder(task_id.to(x_src.dtype))
                elif self.task_embed_type == "onehot_linear":
                    task_id = torch.nn.functional.one_hot(
                        task_id.squeeze(-1), num_classes=self.num_tasks
                    ).to(x_src.dtype)
                    task_id = self.task_encoder(task_id)
                elif self.task_embed_type == "self_attn":
                    task_id = self.task_encoder(task_id.to(x_src.dtype), x_src)
                else:
                    raise ValueError(
                        f"task_embed_type {self.task_embed_type} not recognized"
                    )
                
                if len(x_src.shape) == 4:
                    task_id = task_id.unsqueeze(1).repeat(1, D, 1, 1)
                else:
                    assert len(x_src.shape) == 3
            
                x_src = x_src + task_id
                
            # distinguish between target and source inputs
            if self.target_aware:
                target_mask = task_id.squeeze(-1) == 0
                target_embedding = self.target_encoder(torch.zeros(1, device=x_src.device).long())
                x_src[target_mask] = x_src[target_mask] + target_embedding
                source_embedding = self.target_encoder(torch.ones(1, device=x_src.device).long())
                x_src[~target_mask] = x_src[~target_mask] + source_embedding


            if self.decoder_dict_once is not None:
                x_src = torch.cat(
                    [x_src, self.decoder_dict_once_embeddings.repeat(1, B, 1)],
                    dim=0,
                )

            # y_src = y_src.unsqueeze(1) if len(y_src.shape) < len(x_src.shape) else y_src
            y_src = self.y_encoder(y_src.unsqueeze(-1) if len(y_src.shape) < len(x_src.shape) else y_src) if y_src is not None else None

        if self.style_encoder:
            assert (
                style_src is not None
            ), "style_src must be given if style_encoder is used"
            style_src = self.style_encoder(style_src).unsqueeze(0)
        else:
            style_src = torch.tensor([], device=x_src.device)
        global_src = (
            torch.tensor([], device=x_src.device)
            if self.global_att_embeddings is None
            else self.global_att_embeddings.weight.unsqueeze(1).repeat(
                1, x_src.shape[1], 1
            )
        )

        if src_mask is not None:
            assert self.global_att_embeddings is None or isinstance(src_mask, tuple)

        if src_mask is None:
            if self.global_att_embeddings is None:
                full_len = len(x_src) + len(style_src)
                if self.full_attention:
                    # all tokens attend to all tokens
                    src_mask = bool_mask_to_att_mask(
                        torch.ones((full_len, full_len), dtype=torch.bool)
                    ).to(x_src.device)
                elif self.efficient_eval_masking:
                    # shortcut? used in layer.py line 112
                    src_mask = single_eval_pos + len(style_src)
                else:
                    # attend to self + train set
                    src_mask = self.generate_D_q_matrix(
                        full_len, len(x_src) - single_eval_pos
                    ).to(x_src.device)
            else:
                src_mask_args = (
                    self.global_att_embeddings.num_embeddings,
                    len(x_src) + len(style_src),
                    len(x_src) + len(style_src) - single_eval_pos,
                )
                src_mask = (
                    self.generate_global_att_globaltokens_matrix(*src_mask_args).to(
                        x_src.device
                    ),
                    self.generate_global_att_trainset_matrix(*src_mask_args).to(
                        x_src.device
                    ),
                    self.generate_global_att_query_matrix(*src_mask_args).to(
                        x_src.device
                    ),
                )

        if self.encode_x_y is None:
            train_embeddings = x_src[:single_eval_pos]
            # combine embeddings for train
            if y_src is not None:
                if "axial" in self.attn_type:
                    y_src = y_src.repeat(1, D, 1, 1)
                train_embeddings = train_embeddings + y_src[:single_eval_pos]
            # src is concatenated (train_x, train_y) and test_x
            src = torch.cat([global_src, style_src, train_embeddings, x_src[single_eval_pos:]], 0)

        if self.input_ln is not None:
            src = self.input_ln(src)

        if self.pos_encoder is not None:
            src = self.pos_encoder(src)

        # if self.task_embed_type == "task_attn" or self.task_embed_type == "task_attn_shared":
        #     output = self.transformer_encoder(src, task_id, src_mask)
        # else:
        #     output = self.transformer_encoder(src, src_mask)
        output = self.transformer_encoder(src, task_id, src_mask)

        num_prefix_positions = len(style_src) + (
            self.global_att_embeddings.num_embeddings
            if self.global_att_embeddings
            else 0
        )
        if self.return_all_outputs:
            out_range_start = num_prefix_positions
        else:
            out_range_start = single_eval_pos + num_prefix_positions

        # In the line below, we use the indexing feature, that we have `x[i:None] == x[i:]`
        out_range_end = (
            -len(self.decoder_dict_once_embeddings)
            if self.decoder_dict_once is not None
            else None
        )

        # take care the output once are counted from the end
        output_once = (
            {
                k: v(output[-(i + 1)])
                for i, (k, v) in enumerate(self.decoder_dict_once.items())
            }
            if self.decoder_dict_once is not None
            else {}
        )

        output_dict = {}
        for k, v in self.decoder_dict.items():
            if "axial" in self.attn_type:
                # mean-pool across column dimension
                output = output.mean(dim=1)
            # decode only test points
            output_dict[k] = v(output[out_range_start:out_range_end])
        output = output_dict

        if only_return_standard_out:
            return output["standard"]

        if output_once:
            return output, output_once
        return output

    @torch.no_grad()
    def init_from_small_model(self, small_model):
        assert (
            isinstance(self.decoder, nn.Linear)
            and isinstance(self.encoder, (nn.Linear, nn.Sequential))
            and isinstance(self.y_encoder, (nn.Linear, nn.Sequential))
        )

        def set_encoder_weights(my_encoder, small_model_encoder):
            my_encoder_linear, small_encoder_linear = (
                (my_encoder, small_model_encoder)
                if isinstance(my_encoder, nn.Linear)
                else (my_encoder[-1], small_model_encoder[-1])
            )
            small_in_dim = small_encoder_linear.out_features
            my_encoder_linear.weight.zero_()
            my_encoder_linear.bias.zero_()
            my_encoder_linear.weight[:small_in_dim] = small_encoder_linear.weight
            my_encoder_linear.bias[:small_in_dim] = small_encoder_linear.bias

        set_encoder_weights(self.encoder, small_model.encoder)
        set_encoder_weights(self.y_encoder, small_model.y_encoder)

        small_in_dim = small_model.decoder.in_features

        self.decoder.weight[:, :small_in_dim] = small_model.decoder.weight
        self.decoder.bias = small_model.decoder.bias

        for my_layer, small_layer in zip(
            self.transformer_encoder.layers, small_model.transformer_encoder.layers
        ):
            small_hid_dim = small_layer.linear1.out_features
            my_in_dim = my_layer.linear1.in_features

            # packed along q,k,v order in first dim
            my_in_proj_w = my_layer.self_attn.in_proj_weight
            small_in_proj_w = small_layer.self_attn.in_proj_weight

            my_in_proj_w.view(3, my_in_dim, my_in_dim)[
                :, :small_in_dim, :small_in_dim
            ] = small_in_proj_w.view(3, small_in_dim, small_in_dim)
            my_layer.self_attn.in_proj_bias.view(3, my_in_dim)[:, :small_in_dim] = (
                small_layer.self_attn.in_proj_bias.view(3, small_in_dim)
            )

            my_layer.self_attn.out_proj.weight[:small_in_dim, :small_in_dim] = (
                small_layer.self_attn.out_proj.weight
            )
            my_layer.self_attn.out_proj.bias[:small_in_dim] = (
                small_layer.self_attn.out_proj.bias
            )

            my_layer.linear1.weight[:small_hid_dim, :small_in_dim] = (
                small_layer.linear1.weight
            )
            my_layer.linear1.bias[:small_hid_dim] = small_layer.linear1.bias

            my_layer.linear2.weight[:small_in_dim, :small_hid_dim] = (
                small_layer.linear2.weight
            )
            my_layer.linear2.bias[:small_in_dim] = small_layer.linear2.bias

            my_layer.norm1.weight[:small_in_dim] = (
                math.sqrt(small_in_dim / my_in_dim) * small_layer.norm1.weight
            )
            my_layer.norm2.weight[:small_in_dim] = (
                math.sqrt(small_in_dim / my_in_dim) * small_layer.norm2.weight
            )

            my_layer.norm1.bias[:small_in_dim] = small_layer.norm1.bias
            my_layer.norm2.bias[:small_in_dim] = small_layer.norm2.bias


class TransformerEncoderDiffInit(Module):
    r"""TransformerEncoder is a stack of N encoder layers

    Args:
        encoder_layer_creator: a function generating objects of TransformerEncoderLayer class without args (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).
    """

    __constants__ = ["norm"]

    def __init__(self, encoder_layer_creator, num_layers, norm=None):
        super().__init__()
        self.layers = nn.ModuleList(
            [encoder_layer_creator() for _ in range(num_layers)]
        )
        self.num_layers = num_layers
        self.norm = norm

    def forward(
        self,
        src: Tensor,
        mask: Optional[Tensor] = None,
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
        output = src

        for mod in self.layers:
            output = mod(
                output, src_mask=mask, src_key_padding_mask=src_key_padding_mask
            )

        if self.norm is not None:
            output = self.norm(output)

        return output



class TransformerTaskEncoder(Module):
    r"""TransformerEncoder is a stack of N encoder layers

    Args:
        encoder_layer_creator: a function generating objects of TransformerEncoderLayer class without args (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).
    """

    __constants__ = ["norm"]

    def __init__(self, encoder_layer_creator, num_layers, norm=None):
        super().__init__()
        self.layers = nn.ModuleList(
            [encoder_layer_creator() for _ in range(num_layers)]
        )
        self.num_layers = num_layers
        self.norm = norm

    def forward(
        self,
        src: Tensor,
        task_ids: Tensor,
        mask: Optional[Tensor] = None,
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
        output = src

        for mod in self.layers:
            output = mod(
                output, task_ids, src_mask=mask, src_key_padding_mask=src_key_padding_mask
            )

        if self.norm is not None:
            output = self.norm(output)

        return output
