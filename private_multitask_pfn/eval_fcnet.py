import h5py
import numpy as np
import torch
# from ax.fb.utils.storage.manifold import AEManifoldUseCase
# from ax.fb.utils.storage.manifold_torch import AEManifoldTorchClient
import pickle

MANIFOLD_KEYS = {
    "fcnet_naval_propulsion_data": "tree/trbo_dev/22a02b98-85a3-11ef-a873-472b54035c6c",
    "fcnet_parkinsons_telemonitoring_data": "tree/trbo_dev/2ec159c4-85a3-11ef-a873-472b54035c6c",
    "fcnet_protein_structure_data": "tree/trbo_dev/3b527894-85a3-11ef-a873-472b54035c6c",
    "fcnet_slice_localization_data": "tree/trbo_dev/47f999c4-85a3-11ef-a873-472b54035c6c",
}


def save_to_manifold():
    datasets = [
        "fcnet_naval_propulsion_data",
        "fcnet_parkinsons_telemonitoring_data",
        "fcnet_protein_structure_data",
        "fcnet_slice_localization_data",
    ]

    for dataset in datasets:
        lut = {}
        metric_name = "valid_loss"
        with h5py.File(
            f"/home/yl9959/mtpfn/datasets/fcnet_tabular_benchmarks/{dataset}.hdf5"
        ) as fh:
            for _, (k, v) in enumerate(fh.items()):
                lut[k] = float(np.mean(v[metric_name][()][:, -1]))
                # if i > 10000:
                #     break

        filter_rules = {
            "activation_fn_1": "relu",
            "activation_fn_2": "relu",
            "lr_schedule": "cosine",
        }
        xs = []
        ys = []
        for config_string in lut:
            keep = True
            config = eval(config_string)
            for filter_key in filter_rules:
                if config[filter_key] != filter_rules[filter_key]:
                    keep = False
                    break

            if keep:
                result = [
                    config["batch_size"],
                    config["dropout_1"],
                    config["dropout_2"],
                    config["init_lr"],
                    config["n_units_1"],
                    config["n_units_2"],
                ]
                xs.append(result)
                ys.append(lut[config_string])

        xs = torch.tensor(xs)
        for index in [0, 3, 4, 5]:
            xs[:, index] = xs[:, index].log()
        xs -= xs.min(0, keepdim=True)[0]
        xs /= xs.max(0, keepdim=True)[0]

        ys = -torch.tensor(ys)
        results = {
            "x": xs,
            "y": ys,
        }

        pickle.dump(results, open(f"/home/yl9959/mtpfn/datasets/fcnet_tabular_benchmarks/{dataset}.pkl", "wb"))
        # client = AEManifoldTorchClient(AEManifoldUseCase.TRBO_DEV)
        # final_key = client.torch_save(data=results)
        # print(dataset, final_key)
        
if __name__ == "__main__":
    save_to_manifold()


def get_torch_format_fcnet(target_result, source_results, max_num_features):
    target_xs = target_result["x"]
    target_ys = target_result["y"]

    dropped_columns = []
    dropped_values = []
    for _ in range(target_xs.shape[-1] - max_num_features):
        max_rows = 0
        best_value = None
        best_column = None
        for column in range(target_xs.shape[-1]):
            for unique_value in torch.unique(target_xs[:, column]):
                num_rows = len(target_ys[target_xs[:, column] == unique_value].unique())
                if num_rows > max_rows:
                    max_rows = num_rows
                    best_value, best_column = unique_value, column
        mask = target_xs[:, best_column] == best_value
        target_xs = target_xs[mask]
        target_ys = target_ys[mask]
        target_xs = torch.cat(
            (target_xs[:, :best_column], target_xs[:, best_column + 1 :]), dim=-1
        )
        dropped_columns.append(best_column)
        dropped_values.append(best_value)

    sources_xs = []
    sources_ys = []
    for source in source_results:
        source_xs = source["x"]
        source_ys = source["y"]

        for column, value in zip(dropped_columns, dropped_values):
            mask = source_xs[:, column] == value
            source_xs = source_xs[mask]
            source_ys = source_ys[mask]
            source_xs = torch.cat(
                (source_xs[:, :column], source_xs[:, column + 1 :]), dim=-1
            )

        sources_xs.append(source_xs)
        sources_ys.append(source_ys.unsqueeze(-1))

    return target_xs, target_ys.unsqueeze(-1), sources_xs, sources_ys
