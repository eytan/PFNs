import torch
from gen_task_batch import task_invariant_batch, task_invariant_eval_batch
import json
import os
from utils import load_model


device = "cuda" if torch.cuda.is_available() else "cpu"
# standard attention
# ckpt_dir = "/home/yl9959/mtpfn/ckpt/24-12-18_15-48-06__prior_toy_task_invariant__features_3__tasks_2__epochs_10__seqlen_200__attn_axial"
ckpt_dir = "/home/yl9959/mtpfn/ckpt/24-12-18_17-39-47__prior_toy_task_invariant__features_3__tasks_2__epochs_10__seqlen_200__attn_axial" # use batch and seq len task encoding

model = load_model(ckpt_dir).to(device)

torch.manual_seed(0)

num_features = 3
num_train = 50
num_test = 20


batch_size = 1
seq_len = num_train + num_test


# x = torch.randint(5, (batch_size, seq_len, num_features), device=device)
x = torch.rand(batch_size, seq_len, num_features, device=device)
y = torch.zeros(batch_size, seq_len, 1, device=device)

task_id = torch.randint(2, (batch_size, seq_len, 1), device=device)
opposite_task_id = (task_id + 1) % 2

correct_y = torch.zeros(batch_size, seq_len, 1, device=device)
incorrect_y = torch.zeros(batch_size, seq_len, 1, device=device)

# ID Task 1: y = x_1 + x_2^2 + ... + x_d^d
for feature in range(num_features):
    task_mask = task_id == 0
    correct_y += 0 * task_mask
    incorrect_y += 0 * ~task_mask
    
for feature in range(num_features):
    task_mask = task_id == 1
    correct_y += torch.pow(x[:, :, feature:feature+1], num_features - feature) * task_mask
    incorrect_y += torch.pow(x[:, :, feature:feature+1], num_features - feature) * ~task_mask

train_x = x.transpose(0, 1)[:num_train]
train_correct_y = correct_y.transpose(0, 1)[:num_train]
train_incorrect_y = incorrect_y.transpose(0, 1)[:num_train]
train_original_task_id = task_id.transpose(0, 1)[:num_train]
train_opposite_task_id = opposite_task_id.transpose(0, 1)[:num_train]


test_x = x.transpose(0, 1)[num_train:]
test_correct_y = correct_y.transpose(0, 1)[num_train:]
test_incorrect_y = incorrect_y.transpose(0, 1)[num_train:]
test_original_task_id = task_id.transpose(0, 1)[num_train:]
test_opposite_task_id = opposite_task_id.transpose(0, 1)[num_train:]


# Original
pred_original_y = model(train_x, train_original_task_id, train_correct_y, test_x, test_original_task_id)
# Opposite
pred_opposite_y = model(train_x, train_opposite_task_id, train_correct_y, test_x, test_opposite_task_id)


original_y_mean, original_y_var = pred_original_y[..., 0], pred_original_y[..., 1].exp()
opposite_y_mean, opposite_var = pred_opposite_y[..., 0], pred_opposite_y[..., 1].exp()

criterion = torch.nn.GaussianNLLLoss(reduction="none", full=True)
original_loss = criterion(original_y_mean, test_correct_y, var=original_y_var)
opposite_loss = criterion(opposite_y_mean, test_correct_y, var=opposite_var)

for i in range(num_test):
    print("\nTest Point", i)
    print("Original Y", test_correct_y[i].item())
    print("Original Prediction: N({:.2f}, {:.2f})".format(original_y_mean[i].item(), original_y_var[i].sqrt().item()))
    print("Opposite Prediction: N({:.2f}, {:.2f})".format(opposite_y_mean[i].item(), opposite_var[i].sqrt().item()))
    
    print("Incorrect Y", test_incorrect_y[i].item())
    


lily













id_batch = task_invariant_batch(
    batch_size=1,
    seq_len=num_train+num_test,
    num_features=num_features,
    max_num_tasks=2,
    num_tasks=2,
    lengthscale=1.0,
    hyperparameters=None,
    device=device,
)

ood_batch = task_invariant_eval_batch(
    batch_size=1,
    seq_len=num_train+num_test,
    num_features=num_features,
    max_num_tasks=2,
    num_tasks=2,
    lengthscale=1.0,
    hyperparameters=None,
    device=device,
)

batches = [id_batch, ood_batch]
batch_type = ["ID", "OOD"]

for name, batch in zip(batch_type, batches):
    train_x = batch.x[:num_train]
    train_y = batch.y[:num_train]
    train_task_id = batch.task_id[:num_train]

    test_task_id_mask = (batch.task_id[num_train:] == 0).squeeze(-1)
    test_x = batch.x[num_train:][test_task_id_mask].unsqueeze(1)
    test_y = batch.y[num_train:][test_task_id_mask].unsqueeze(1)

    y_pred = model(train_x, train_task_id, train_y, test_x)
    y_mean, y_var = y_pred[..., 0], y_pred[..., 1].exp()

    criterion = torch.nn.GaussianNLLLoss(reduction="none", full=True)
    loss = criterion(y_mean, test_y.squeeze(1), var=y_var)

    print(f"Test Loss: {loss.mean().item()}", name)

