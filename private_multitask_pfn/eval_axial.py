import torch
from gen_axial_batch import axial_train_batch, axial_test_batch
import json
import os
from train import train as load_model_from_train

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
ckpt_dir = "/home/yl9959/mtpfn/ckpt/24-12-04_14-36-42__prior_axial_train__features_3__tasks_1__epochs_30__seqlen_200__attn_standard"
model = load_model(ckpt_dir).to(device)

# axial attention
ckpt_dir = "/home/yl9959/mtpfn/ckpt/24-12-04_15-19-42__prior_axial_train__features_3__tasks_1__epochs_30__seqlen_200__attn_axial"
axial_model = load_model(ckpt_dir).to(device)

torch.manual_seed(0)

num_features = 3
num_train = 10
num_test = 3

x = torch.rand(num_train + num_test, 1, num_features, device=device)
y = torch.zeros(num_train + num_test, 1, 1, device=device)
for feature in range(num_features):
    y += torch.pow(x[:, :, feature:feature+1], feature + 1)

x_train = x[:num_train]
y_train = y[:num_train]
x_test = x[num_train:]
y_test = y[num_train:]

with torch.no_grad():
    y_pred = model(x_train, y_train, x_test)
    y_pred_axial = axial_model(x_train, y_train, x_test)
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
reversed_x_train = reversed_x[:num_train]
reversed_x_test = reversed_x[num_train:]

with torch.no_grad():
    reversed_y_pred = model(reversed_x_train, y_train, reversed_x_test)
    reversed_y_pred_axial = axial_model(reversed_x_train, y_train, reversed_x_test)
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