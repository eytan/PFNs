import math

import torch
import torch.nn as nn

from .utils import normalize_data


class StyleEncoder(nn.Module):
    def __init__(self, num_hyperparameters, em_size):
        super().__init__()
        self.em_size = em_size
        self.embedding = nn.Linear(num_hyperparameters, self.em_size)

    def forward(self, hyperparameters):  # B x num_hps
        return self.embedding(hyperparameters)


class StyleEmbEncoder(nn.Module):
    def __init__(self, num_hyperparameters, em_size, num_embeddings=100):
        super().__init__()
        assert num_hyperparameters == 1
        self.em_size = em_size
        self.embedding = nn.Embedding(num_embeddings, self.em_size)

    def forward(self, hyperparameters):  # B x num_hps
        return self.embedding(hyperparameters.squeeze(1))


class _PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.0):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        self.device_test_tensor = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):  # T x B x num_features
        assert self.d_model % x.shape[-1] * 2 == 0
        d_per_feature = self.d_model // x.shape[-1]
        pe = torch.zeros(*x.shape, d_per_feature, device=self.device_test_tensor.device)
        # position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        interval_size = 10
        div_term = (
            (1.0 / interval_size)
            * 2
            * math.pi
            * torch.exp(
                torch.arange(
                    0, d_per_feature, 2, device=self.device_test_tensor.device
                ).float()
                * math.log(math.sqrt(2))
            )
        )
        # print(div_term/2/math.pi)
        pe[..., 0::2] = torch.sin(x.unsqueeze(-1) * div_term)
        pe[..., 1::2] = torch.cos(x.unsqueeze(-1) * div_term)
        return self.dropout(pe).view(x.shape[0], x.shape[1], self.d_model)


Positional = lambda _, emsize: _PositionalEncoding(d_model=emsize)


class EmbeddingEncoder(nn.Module):
    def __init__(self, num_features, em_size, num_embs=100):
        super().__init__()
        self.num_embs = num_embs
        self.embeddings = nn.Embedding(num_embs * num_features, em_size, max_norm=True)
        self.init_weights(0.1)
        self.min_max = (-2, +2)

    @property
    def width(self):
        return self.min_max[1] - self.min_max[0]

    def init_weights(self, initrange):
        self.embeddings.weight.data.uniform_(-initrange, initrange)

    def discretize(self, x):
        split_size = self.width / self.num_embs
        return (x - self.min_max[0] // split_size).int().clamp(0, self.num_embs - 1)

    def forward(self, x):  # T x B x num_features
        x_idxs = self.discretize(x)
        x_idxs += (
            torch.arange(x.shape[-1], device=x.device).view(1, 1, -1) * self.num_embs
        )
        # print(x_idxs,self.embeddings.weight.shape)
        return self.embeddings(x_idxs).mean(-2)


class Normalize(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, x):
        return (x - self.mean) / self.std


class NormalizeMultitask(nn.Module):
    def __init__(self, num_tasks, mean, std):
        super().__init__()
        self.num_tasks = num_tasks
        self.mean = mean
        self.std = std

    def forward(self, x):
        features = x[..., self.num_tasks :]
        features = (features - self.mean) / self.std
        x[..., self.num_tasks :] = features
        return x


class SqueezeBetween0and1(nn.Module):  # take care of test set here
    def forward(self, x):
        width = x.max(0).values - x.min(0).values
        result = (x - x.min(0).values) / width
        result[(width == 0)[None].repeat(len(x), *[1] * (len(x.shape) - 1))] = 0.5
        return result


def get_normalized_uniform_encoder(encoder_creator):
    """
    This can be used to wrap an encoder that is fed uniform samples in [0,1] and normalizes these to 0 mean and 1 std.
    For example, it can be used as `encoder_creator = get_normalized_uniform_encoder(encoders.Linear)`, now this can
    be initialized with `encoder_creator(feature_dim, in_dim)`.
    :param encoder:
    :return:
    """
    return lambda in_dim, out_dim: nn.Sequential(
        Normalize(0.5, math.sqrt(1 / 12)), encoder_creator(in_dim, out_dim)
    )


def get_normalized_uniform_multitask_encoder(encoder_creator):
    """
    This can be used to wrap an encoder that is fed uniform samples in [0,1] and normalizes these to 0 mean and 1 std.
    For example, it can be used as `encoder_creator = get_normalized_uniform_encoder(encoders.Linear)`, now this can
    be initialized with `encoder_creator(feature_dim, in_dim)`.
    :param encoder:
    :return:
    """
    return lambda feature_dim, task_dim, out_dim: nn.Sequential(
        NormalizeMultitask(task_dim, 0.5, math.sqrt(1 / 12)),
        encoder_creator(feature_dim, task_dim, out_dim),
    )
    
    
class AxialNormalizedUniformEncoder(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.normalize = Normalize(0.5, math.sqrt(1 / 12))
        self.unsqueeze = lambda x: x.unsqueeze(-1)  # Custom unsqueeze
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        # Start with N x B x D
        x = self.normalize(x).unsqueeze(-1)   # Normalize and add a dimension at the last axis
        x = self.linear(x)      # Apply the linear transformation 
        return x.permute(0, 2, 1, 3)    # Permute the axes to get N x D x B x E

    
def get_axial_normalized_uniform_encoder():
    return lambda in_dim, out_dim: AxialNormalizedUniformEncoder(in_dim, out_dim)

def get_normalized_encoder(encoder_creator, data_std):
    return lambda in_dim, out_dim: nn.Sequential(
        Normalize(0.0, data_std), encoder_creator(in_dim, out_dim)
    )


def get_log_dims(x, eps=1e-10):
    logged_x = ((x + eps).log() - math.log(eps)) / (math.log(1.0 + eps) - math.log(eps))
    return logged_x


def add_log_neglog_dims(x, eps=1e-10):
    logged_x = get_log_dims(x, eps) / 2.0
    neglogged_x = 1 - get_log_dims(1 - x, eps) / 2.0
    logged_x[x > 0.5] = neglogged_x[x > 0.5]
    return torch.stack([x, logged_x], -1).view(*x.shape[:-1], -1)


class AddLogNegLogDims(nn.Module):
    def __init__(self, eps=1e-10):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        return add_log_neglog_dims(x, self.eps)


def get_logdim_encoder(encoder_creator, eps=1e-10):
    return lambda in_dim, out_dim: nn.Sequential(
        AddLogNegLogDims(eps), encoder_creator(in_dim * 2, out_dim)
    )


class ZNormalize(nn.Module):
    def forward(self, x):
        std = x.std(-1, keepdim=True)
        std[std == 0.0] = 1.0
        return (x - x.mean(-1, keepdim=True)) / std


class ZNormalizePerDataset(nn.Module):
    def forward(self, x):
        std = x.std(0, keepdim=True)
        std[std == 0.0] = 1.0
        return (x - x.mean(0, keepdim=True)) / std


class AppendEmbeddingEncoder(nn.Module):
    def __init__(self, base_encoder, num_features, emsize):
        super().__init__()
        self.num_features = num_features
        self.base_encoder = base_encoder
        self.emb = nn.Parameter(torch.zeros(emsize))

    def forward(self, x):
        if (x[-1] == 1.0).all():
            append_embedding = True
        else:
            assert (x[-1] == 0.0).all(), (
                "You need to specify as last position whether to append embedding. "
                "If you don't want this behavior, please use the wrapped encoder instead."
            )
            append_embedding = False
        x = x[:-1]
        encoded_x = self.base_encoder(x)
        if append_embedding:
            encoded_x = torch.cat(
                [encoded_x, self.emb[None, None, :].repeat(1, encoded_x.shape[1], 1)], 0
            )
        return encoded_x


def get_append_embedding_encoder(encoder_creator):
    return lambda num_features, emsize: AppendEmbeddingEncoder(
        encoder_creator(num_features, emsize), num_features, emsize
    )


class VariableNumFeaturesMultitaskEncoder(nn.Module):
    def __init__(self, base_encoder, num_features, num_tasks):
        super().__init__()
        self.base_encoder = base_encoder
        self.num_features = num_features
        self.num_tasks = num_tasks

    def forward(self, x):
        x_num_features = x.shape[-1] - self.num_tasks
        if x_num_features > self.num_features:
            raise ValueError(
                f"The model was trained using {self.num_features} features, but got {x_num_features}."
            )
        rescale_factor = self.num_features / x_num_features
        x[..., -x_num_features:] = x[..., -x_num_features:] * rescale_factor
        # pad with zeros
        x = torch.cat(
            (
                x,
                torch.zeros(
                    *x.shape[:-1], self.num_features - x_num_features, device=x.device
                ),
            ),
            -1,
        )
        return self.base_encoder(x)


def get_variable_num_features_multitask_encoder(encoder_creator):
    return lambda num_features, num_tasks, emsize: VariableNumFeaturesMultitaskEncoder(
        encoder_creator(num_features + num_tasks, emsize), num_features, num_tasks
    )
    
    
def get_axial_num_features_multitask_encoder(encoder_creator):
    return lambda num_features, num_columns, num_tasks, emsize: encoder_creator(num_features, num_columns, emsize)


class VariableNumFeaturesEncoder(nn.Module):
    def __init__(self, base_encoder, num_features):
        super().__init__()
        self.base_encoder = base_encoder
        self.num_features = num_features

    def forward(self, x):
        x = x * (self.num_features / x.shape[-1])
        x = torch.cat(
            (
                x,
                torch.zeros(
                    *x.shape[:-1], self.num_features - x.shape[-1], device=x.device
                ),
            ),
            -1,
        )
        return self.base_encoder(x)


def get_variable_num_features_encoder(encoder_creator):
    return lambda num_features, emsize: VariableNumFeaturesEncoder(
        encoder_creator(num_features, emsize), num_features
    )


class NoMeanEncoder(nn.Module):
    """
    This can be useful for any prior that is translation invariant in x or y.
    A standard GP for example is translation invariant in x.
    That is, GP(x_test+const,x_train+const,y_train) = GP(x_test,x_train,y_train).
    """

    def __init__(self, base_encoder):
        super().__init__()
        self.base_encoder = base_encoder

    def forward(self, x):
        return self.base_encoder(x - x.mean(0, keepdim=True))


def get_no_mean_encoder(encoder_creator):
    return lambda num_features, emsize: NoMeanEncoder(
        encoder_creator(num_features, emsize)
    )


MLP = lambda num_features, emsize: nn.Sequential(
    nn.Linear(num_features, emsize * 2), nn.ReLU(), nn.Linear(emsize * 2, emsize)
)


class NanHandlingEncoder(nn.Module):
    def __init__(self, num_features, emsize, keep_nans=True):
        super().__init__()
        self.num_features = 2 * num_features if keep_nans else num_features
        self.emsize = emsize
        self.keep_nans = keep_nans
        self.layer = nn.Linear(self.num_features, self.emsize)

    def forward(self, x):
        if self.keep_nans:
            x = torch.cat(
                [
                    torch.nan_to_num(x, nan=0.0),
                    normalize_data(
                        torch.isnan(x) * -1
                        + torch.logical_and(torch.isinf(x), torch.sign(x) == 1) * 1
                        + torch.logical_and(torch.isinf(x), torch.sign(x) == -1) * 2
                    ),
                ],
                -1,
            )
        else:
            x = torch.nan_to_num(x, nan=0.0)
        return self.layer(x)


class Linear(nn.Linear):
    def __init__(self, num_features, emsize, replace_nan_by_zero=False):
        super().__init__(num_features, emsize)
        self.num_features = num_features
        self.emsize = emsize
        self.replace_nan_by_zero = replace_nan_by_zero

    def forward(self, x):
        if self.replace_nan_by_zero:
            x = torch.nan_to_num(x, nan=0.0)
        return super().forward(x)

    def __setstate__(self, state):
        super().__setstate__(state)
        self.__dict__.setdefault("replace_nan_by_zero", True)
        
        
# class AxialLinear(nn.Linear):
#     def __init__(self, emsize, replace_nan_by_zero=False):
#         super().__init__(emsize)
#         self.emsize = emsize
#         self.replace_nan_by_zero = replace_nan_by_zero

#     def forward(self, x):
#         if self.replace_nan_by_zero:
#             x = torch.nan_to_num(x, nan=0.0)
#         return super().forward(x)

#     def __setstate__(self, state):
#         super().__setstate__(state)
#         self.__dict__.setdefault("replace_nan_by_zero", True)



class Conv(nn.Module):
    def __init__(self, input_size, emsize):
        super().__init__()
        self.convs = torch.nn.ModuleList(
            [nn.Conv2d(64 if i else 1, 64, 3) for i in range(5)]
        )
        self.linear = nn.Linear(64, emsize)

    def forward(self, x):
        size = math.isqrt(x.shape[-1])
        assert size * size == x.shape[-1]
        x = x.reshape(*x.shape[:-1], 1, size, size)
        for conv in self.convs:
            if x.shape[-1] < 4:
                break
            x = conv(x)
            x.relu_()
        x = nn.AdaptiveAvgPool2d((1, 1))(x).squeeze(-1).squeeze(-1)
        return self.linear(x)


class CanEmb(nn.Embedding):
    def __init__(
        self, num_features, num_embeddings: int, embedding_dim: int, *args, **kwargs
    ):
        assert embedding_dim % num_features == 0
        embedding_dim = embedding_dim // num_features
        super().__init__(num_embeddings, embedding_dim, *args, **kwargs)

    def forward(self, x):
        lx = x.long()
        assert (lx == x).all(), "CanEmb only works with tensors of whole numbers"
        x = super().forward(lx)
        return x.view(*x.shape[:-2], -1)


def get_Canonical(num_classes):
    return lambda num_features, emsize: CanEmb(num_features, num_classes, emsize)


def get_Embedding(num_embs_per_feature=100):
    return lambda num_features, emsize: EmbeddingEncoder(
        num_features, emsize, num_embs=num_embs_per_feature
    )


class SelfAttentionTaskOld(nn.Module):
    
    def __init__(self, num_tasks, emsize):
        super(SelfAttentionTask, self).__init__()
        self.num_tasks = num_tasks
        
        self.task_embedding = nn.Embedding(num_tasks, emsize)
        # Multi-head self-attention
        self.attention = nn.MultiheadAttention(embed_dim=emsize, num_heads=1, batch_first=False)
        # Fully connected layers for downstream task
        self.fc = nn.Sequential(
            nn.Linear(emsize, 128),
            nn.ReLU(),
            nn.Linear(128, emsize)  # Example: regression output
        )
        
    def forward(self, task_ids, context):
        # remove the last dimension
        x = x.squeeze(-1)
        
        task_embeddings = self.task_embedding(x.long())
        attn_output, _ = self.attention(task_embeddings, task_embeddings, task_embeddings)
        
        output = self.fc(attn_output)
        return output



class SelfAttentionTask(nn.Module):
    def __init__(self, num_tasks, emsize, num_heads=4):
        super(SelfAttentionTask, self).__init__()
        self.num_tasks = num_tasks
        self.embedding_dim = emsize
        
        # Multi-head self-attention layer
        self.attention = nn.MultiheadAttention(embed_dim=emsize, num_heads=num_heads)
        
        # Linear transformation to produce task embeddings
        self.context_transform = nn.Linear(emsize, emsize)
        
        # Task ID embedding layer
        self.task_id_embedding = nn.Embedding(num_tasks, emsize)

    def _ssforward(self, task_ids, context):
        """
        Args:
            task_ids: Tensor of shape (N, B, 1), representing task IDs.
            context: Tensor of shape (N, D, B, E), representing the context for tasks.

        Returns:
            task_embeddings: Tensor of shape (N, B, E), task embeddings.
        """
        # average across D dimension
        context = context.mean(dim=1)
        
        # Embed task IDs
        task_id_embeds = self.task_id_embedding(task_ids.squeeze(-1).long())  # Shape: (N, B, E)
        context_with_task_ids = context + task_id_embeds
        
        # Self-attention: Query, Key, and Value are the context points with task embeddings
        attn_output, _ = self.attention(context_with_task_ids, context_with_task_ids, context_with_task_ids)
        # task_embedding is average per task
        
        # Apply linear transformation to the aggregated context
        task_embeddings = self.context_transform(attn_output)  # Shape: (N, B, E)
        
        return task_embeddings
    
    def nobatchforward(self, task_ids, context):
        """
        Args:
            task_ids: Tensor of shape (N, B, 1), representing task IDs.
            context: Tensor of shape (N, D, B, E), representing the context for tasks.

        Returns:
            task_embeddings: Tensor of shape (N, B, E), task embeddings.
        """
        N, D, B, E = context.shape
        task_embeddings = torch.zeros(N, B, E, device=context.device)
        task_embedding_lookup = torch.zeros(self.num_tasks, E, device=context.device)
        
        # average across D dimension
        context = context.mean(dim=1)
        # Self-attention: Query, Key, and Value are the context points with task embeddings
        attn_output, _ = self.attention(context, context, context)
        # print(attn_output.shape, "ATTN") # (N, B, E)
        
        for task in task_ids.long().unique():
            task_mask = task == task_ids
            attn_tasks = attn_output[task_mask.squeeze(-1)]
            # print(attn_tasks.shape)
            task_embedding_lookup[task] = attn_tasks.sum(dim=0) / task_mask.sum()
        
        # expanded_task_ids = task_ids.expand(-1, -1, E)
        # import pdb; pdb.set_trace()
        task_embeddings = task_embedding_lookup[task_ids.long().squeeze(-1)]
        # print(task_embeddings.shape, "TASK")
        
        return task_embeddings
    
    
    def forward(self, task_ids, context):
        """
        Args:
            task_ids: Tensor of shape (N, B, 1), representing task IDs.
            context: Tensor of shape (N, D, B, E), representing the context for tasks.

        Returns:
            task_embeddings: Tensor of shape (N, B, E), task embeddings.
        """
        N, D, B, E = context.shape
        task_embeddings = torch.zeros(N, B, E, device=context.device)
        task_embedding_lookup = torch.zeros(self.num_tasks, B, E, device=context.device)
        
        # average across D dimension
        context = context.mean(dim=1)
        # Self-attention: Query, Key, and Value are the context points with task embeddings
        attn_output, _ = self.attention(context, context, context)

        # # Flatten task_ids for advanced indexing
        # task_ids_flat = task_ids.reshape(-1)  # Shape: (N * B,)
        
        # # Create task-specific mask matrix for each batch and task (N, B)
        # task_mask = task_ids_flat.unsqueeze(1) == torch.arange(self.num_tasks, device=context.device).view(1, -1)  # Shape: (N * B, T)

        # # Repeat attn_output to match task_ids (N, B, E)
        # attn_output_flat = attn_output.view(N * B, E)  # Shape: (N * B, E)

        # # Gather attention outputs based on task_mask (task_mask is (N * B, T))
        # # Sum the attention outputs for each task across the batch
        # # The matmul gives us the task-wise embeddings as (T, N * B, E)
        # task_embedding_lookup = torch.matmul(task_mask.float().T, attn_output_flat)  # Shape: (T, N * B, E)

        
        # # Take the mean across N dimension (the number of samples for each task) 
        # task_embedding_lookup = task_embedding_lookup.mean(dim=2)  # Shape: (T, B, E)

        # # Normalize task embeddings by the number of occurrences per task
        # task_counts = task_mask.sum(dim=0).view(self.num_tasks)  # Shape: (T,)
        # task_embedding_lookup = task_embedding_lookup / task_counts.view(-1, 1).float()  # Shape: (T, B, E)

        # # Lookup task embeddings using task_ids for each batch
        # task_embeddings = task_embedding_lookup[task_ids_flat].view(N, B, E)

        # return task_embeddings
        
        
        
        # Iterate over batches
        for batch in range(B):
            # Get unique task_ids in this batch
            batch_task_ids = task_ids[:, batch].long().unique()  # Shape: (T,)
            
            # For each task in the batch, compute its embedding
            for task in batch_task_ids:
                # Create mask for this task in the batch
                task_mask = (task_ids[:, batch] == task).squeeze(-1)  # Shape: (N,)
                # print(task_mask.shape, attn_output.shape, batch)
                
                # Get attention output for this task (flattening across N dimension)
                attn_tasks = attn_output[task_mask, batch]  # Shape: (D, E)
                
                # Aggregate the task-specific embeddings by averaging
                task_embedding = attn_tasks.sum(dim=0) / task_mask.sum().float()  # Shape: (E)
                
                # Store this task embedding in the task_embeddings tensor
                task_embedding_lookup[task, batch] = task_embedding
                
            task_embeddings[:, batch] = task_embedding_lookup[task_ids[:, batch].long().squeeze(-1), batch]
            
        return task_embeddings


# class SelfAttentionTask(nn.Module):
    
#     def __init__(self, num_tasks, emsize):
#         super(SelfAttentionTask, self).__init__()
        
#         self.attention = nn.MultiheadAttention(embed_dim=emsize, num_heads=4)
#         self.context_transform = nn.Linear(emsize, emsize)
        
#     def forward(self, task_ids, context):
#         N, D, B, E = context.shape
#         context = context.view(N * D, B, E)
#         attn_output = self.attention(context, context, context)[0]
#         attn_output = attn_output.view(N, D, B, E)
        
#         task_embeddings = torch.zeros(N, D, B, E, device=context.device)
        
#         for task_id in torch.unique(task_ids):
#             task_mask = task_id == task_ids
#             task_embedding = self.task_embedding(task_id)
#             task_embedding = task_embedding.repeat(N, D, 1, 1)
#             task_embedding = task_embedding * task_mask.unsqueeze(-1)
#             task_embeddings += task_embedding
        
        
        


def get_self_attention_task_encoder():
    return lambda num_tasks, emsize: SelfAttentionTask(num_tasks, emsize)