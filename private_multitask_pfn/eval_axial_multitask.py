import torch
from gen_axial_batch import axial_train_batch, axial_test_batch
import json
import os
from train import train as load_model_from_train
import matplotlib.pyplot as plt

def load_model(ckpt_dir, best=True):
# ckpt_dir = "/home/yl9959/mtpfn/ckpt/24-12-04_14-36-42__prior_axial_train__features_3__tasks_1__epochs_30__seqlen_200__attn_standard"
    args_json = f"{ckpt_dir}/args.json"
    with open(args_json, "r") as f:
        args = json.load(f)

    model = load_model_from_train(**args, return_model=True)
    if best:
        model.load_state_dict(torch.load(f"{ckpt_dir}/best_model.pth", weights_only=True))
    else:
        model.load_state_dict(torch.load(f"{ckpt_dir}/final_model.pth", weights_only=True))
    
    print("Loaded model")
    return model

device = "cuda" if torch.cuda.is_available() else "cpu"
# standard attention
ckpt_dir = "/home/yl9959/mtpfn/ckpt/24-12-04_22-11-25__prior_toy_multitask__features_3__tasks_2__epochs_10__seqlen_200__attn_standard_completed"
model = load_model(ckpt_dir).to(device)

# axial attention
# ckpt_dir = "/home/yl9959/mtpfn/ckpt/24-12-04_22-07-45__prior_toy_multitask__features_3__tasks_2__epochs_10__seqlen_200__attn_axial_completed"
ckpt_dir = "/home/yl9959/mtpfn/ckpt/24-12-04_22-23-28__prior_toy_multitask__features_3__tasks_2__epochs_30__seqlen_200__attn_axial_completed"
axial_model = load_model(ckpt_dir).to(device)

torch.manual_seed(1)

num_features = 3
num_train = 50
num_test = 100
batch_size = 1
seq_len = num_train + num_test
num_tasks = 1

x = torch.rand(seq_len, batch_size, num_features, device=device)
y = torch.zeros(seq_len, batch_size, 1, device=device)
task_id = torch.randint(0, num_tasks, (seq_len, batch_size, 1), device=device)

for task in range(num_tasks):
    for feature in range(num_features):
        constant = task + 1 #torch.randn(1, device=device)
        y += constant * torch.pow(x[:, :, feature:feature+1], feature + 1) * (task_id == task).float()
    
x_train = x[:num_train]
task_id_train = task_id[:num_train]
y_train = y[:num_train]
x_test = x[num_train:]
y_test = y[num_train:]

with torch.no_grad():
    y_pred = model(x_train, task_id_train, y_train, x_test)
    y_pred_axial = axial_model(x_train, task_id_train, y_train, x_test)
print(y_pred_axial)
y_mean, y_var = y_pred[..., 0], y_pred[..., 1].exp()
y_mean_axial, y_var_axial = y_pred_axial[..., 0], y_pred_axial[..., 1].exp()

criterion = torch.nn.GaussianNLLLoss(reduction="none", full=True)
loss = criterion(y_mean, y_test.squeeze(1), var=y_var)
loss_axial = criterion(y_mean_axial, y_test.squeeze(1), var=y_var_axial)

for i in range(num_test):
    print(f"ID Test {i}")
    print(f"x_test: {x_test[i].squeeze()}")
    print(f"y_test: {y_test[i].item()}")
    print("Standard: N(%.2f, %.2f), Loss: %.2f" % (y_mean[i].item(), y_var[i].item(), loss[i].item()))
    print("Axial: N(%.2f, %.2f), Loss: %.2f" % (y_mean_axial[i].item(), y_var_axial[i].item(), loss_axial[i].item()))
    print()
    
# reversed x in feature dimension
reversed_x = torch.flip(x, dims=[2])
reversed_task_id = torch.flip(task_id, dims=[2])
reversed_x_train = reversed_x[:num_train]
reversed_task_id_train = reversed_task_id[:num_train]
reversed_x_test = reversed_x[num_train:]

with torch.no_grad():
    reversed_y_pred = model(reversed_x_train, reversed_task_id_train, y_train, reversed_x_test)
    reversed_y_pred_axial = axial_model(reversed_x_train, reversed_task_id_train, y_train, reversed_x_test)
reversed_y_mean, reversed_y_var = reversed_y_pred[..., 0], reversed_y_pred[..., 1].exp()
reversed_y_mean_axial, reversed_y_var_axial = reversed_y_pred_axial[..., 0], reversed_y_pred_axial[..., 1].exp()

reversed_loss = criterion(reversed_y_mean, y_test.squeeze(1), var=reversed_y_var)
reversed_loss_axial = criterion(reversed_y_mean_axial, y_test.squeeze(1), var=reversed_y_var_axial)

for i in range(num_test):
    print(f"OOD Test {i}")
    print(f"Reversed x_test: {reversed_x_test[i].squeeze()}")
    print(f"y_test: {y_test[i].item()}")
    print("Standard: N(%.2f, %.2f), Loss: %.2f" % (reversed_y_mean[i].item(), reversed_y_var[i].item(), reversed_loss[i].item()))
    print("Axial: N(%.2f, %.2f), Loss: %.2f" % (reversed_y_mean_axial[i].item(), reversed_y_var_axial[i].item(), reversed_loss_axial[i].item()))
    print()
    
min_loss = min(loss.min().item(), loss_axial.min().item())
max_loss = max(loss.max().item(), loss_axial.max().item())
plt.hist(loss.cpu().numpy(), bins=20, alpha=0.5, label="Standard", range=(min_loss, max_loss))
plt.hist(loss_axial.cpu().numpy(), bins=20, alpha=0.5, label="Axial", range=(min_loss, max_loss))
plt.legend()
plt.xlabel("NLL")
plt.title("Columns in Train Order")
plt.savefig("hist_loss.png")
plt.clf()

min_loss = min(reversed_loss.min().item(), reversed_loss_axial.min().item())
max_loss = max(reversed_loss.max().item(), reversed_loss_axial.max().item())
plt.hist(reversed_loss.cpu().numpy(), bins=20, alpha=0.5, label="Standard", range=(min_loss, max_loss))
plt.hist(reversed_loss_axial.cpu().numpy(), bins=20, alpha=0.5, label="Axial", range=(min_loss, max_loss))
plt.legend()
plt.xlabel("NLL")
plt.title("Columns in Reverse Order")
plt.savefig("hist_reversed.png")
plt.clf()

# plot all 
min_loss = min(loss.min().item(), loss_axial.min().item(), reversed_loss.min().item(), reversed_loss_axial.min().item())
max_loss = max(loss.max().item(), loss_axial.max().item(), reversed_loss_axial.max().item())

plt.hist(loss.cpu().numpy(), bins=20, alpha=0.5, label="Standard", range=(min_loss, max_loss))
plt.hist(loss_axial.cpu().numpy(), bins=20, alpha=0.5, label="Axial", range=(min_loss, max_loss))
plt.hist(reversed_loss.cpu().numpy(), bins=20, alpha=0.5, label="Reversed Standard", range=(min_loss, max_loss))
plt.hist(reversed_loss_axial.cpu().numpy(), bins=20, alpha=0.5, label="Reversed Axial", range=(min_loss, max_loss))
plt.legend()
plt.xlabel("NLL")
plt.title("All")
plt.savefig("hist_all.png")