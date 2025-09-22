import os
import pickle
import pickle
import numpy as np
import pandas as pd
import torch
# from ax.fb.utils.storage.manifold import AEManifoldUseCase
# from ax.fb.utils.storage.manifold_torch import AEManifoldTorchClient

HPOBENCH_MANIFOLD_KEYS = {
    "rf": "tree/trbo_dev/839dd652-8544-11ef-94ac-4d9af618463b",
    "lr": "tree/trbo_dev/ef4ea9d8-858c-11ef-a657-0908f1ce571c",
    "svm": "tree/trbo_dev/bc90dacc-8544-11ef-94ac-4d9af618463b",
    "xgb": "tree/trbo_dev/d58888be-858e-11ef-a657-0908f1ce571c",
}

default_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def drop_columns(df, relevant_columns, num_final_features):
    df = df.copy()
    # fixes one value per column by choosing the value
    # that leads to the most diverse ys
    dropped_columns = []
    dropped_values = []
    for _ in range(len(relevant_columns) - num_final_features):
        # filter
        max_rows = 0
        best_value = None
        best_column = None
        for column in relevant_columns:
            # Iterate through each unique value in the column
            for value in df[column].unique():
                # Filter the DataFrame
                filtered_df = df[df[column] == value]
                num_rows = filtered_df["function_value"].nunique()

                # Check if this value retains the most rows
                if num_rows > max_rows:
                    max_rows = num_rows
                    best_value, best_column = value, column

        df = df[df[best_column] == best_value]
        df = df.drop(best_column, axis=1)
        
        relevant_columns.remove(best_column)
        dropped_columns.append(best_column)
        dropped_values.append(best_value)

    final_columns = [
        column for column in relevant_columns if column not in dropped_columns
    ]

    return df, final_columns, dropped_columns, dropped_values


def get_torch_format_hpobench(target_df, source_dfs, max_num_features, problem, device=default_device):
    processed_dfs = []
    for df in [target_df] + source_dfs:
        df = df.copy()
        df["function_value"] = df["result.function_value"]
        if problem == "rf":
            relevant_columns = [
                "max_depth",
                "max_features",
                "min_samples_leaf",
                "min_samples_split",
            ]
            df["max_depth"] = np.log(df["max_depth"])
            df["min_samples_split"] = np.log(df["min_samples_split"])
        elif problem == "lr":
            relevant_columns = ["alpha", "eta0"]
            df["alpha"] = np.log(df["alpha"])
            df["eta0"] = np.log(df["eta0"])
        elif problem == "svm":
            relevant_columns = ["C", "gamma"]
            df["C"] = np.log(df["C"])
            df["gamma"] = np.log(df["gamma"])
        elif problem == "xgb":
            relevant_columns = ["colsample_bytree", "eta", "max_depth", "reg_lambda"]
            df["eta"] = np.log(df["eta"])
            df["max_depth"] = np.log(df["max_depth"])
            df["reg_lambda"] = np.log(df["reg_lambda"])
        elif problem == "nn":
            print(df.columns)
            relevant_columns = ["alpha", "batch_size", "depth", "learning_rate_init", "width"]
            df["alpha"] = np.log(df["alpha"])
            df["batch_size"] = np.log(df["batch_size"])
            df["depth"] = (df["depth"] - df["depth"].min()) / (df["depth"].max() - df["depth"].min())
            df["learning_rate_init"] = np.log(df["learning_rate_init"])
            df["width"] = np.log(df["width"])

        df = df.groupby(relevant_columns, as_index=False).mean(numeric_only=True)
        processed_dfs.append(df)

    # eventually remove when we train on more features
    # drops columns to get maximum output diversity in target task
    target_df = processed_dfs[0]
    target_df, final_columns, dropped_columns, dropped_values = drop_columns(
        target_df, relevant_columns, max_num_features
    )

    result_xs = []
    result_ys = []
    for df in processed_dfs:
        # limit to non-dropped columns
        for column, value in zip(dropped_columns, dropped_values):
            df = df[df[column] == value]

        # normalize to [0, 1]
        xs = torch.tensor(df[final_columns].values).double()
        xs = xs - xs.min(0, keepdim=True)[0]
        max_values = xs.max(dim=0, keepdim=True)[0]
        max_values[max_values == 0] = 1
        xs = xs / max_values

        # NEGATE VALID_LOSS
        ys = -torch.tensor(df["function_value"].values).unsqueeze(-1)
        # standardize to mean 0, std 1
        std = torch.where(ys.std(0).isnan(), torch.tensor(1.0), ys.std(0))
        ys = (ys - ys.mean()) / std

        result_xs.append(xs.float().to(device))
        result_ys.append(ys.float().to(device))

    return result_xs[0], result_ys[0], result_xs[1:], result_ys[1:]


def create_train_test(
    target_xs, target_ys, sources_xs, sources_ys, n_target, n_source, max_num_task
):
    train_id = []
    train_x = []
    train_y = []
    for i, (source_xs, source_ys) in enumerate(zip(sources_xs, sources_ys)):
        random_indices = torch.randperm(len(source_xs))[:n_source]
        train_x.append(source_xs[random_indices])
        train_y.append(source_ys[random_indices])
        train_id.append(torch.ones(n_source) * i + 1)

    # add target task
    random_indices = torch.randperm(len(target_xs))
    train_indices = random_indices[:n_target]
    test_indices = random_indices[n_target:]
    train_x.append(target_xs[train_indices])
    train_y.append(target_ys[train_indices])
    train_id.append(torch.zeros(n_target))

    train_x = torch.concat(train_x, 0)
    train_id = torch.concat(train_id, 0)

    train_y = torch.concat(train_y, 0)

    test_id = torch.zeros(len(test_indices))
    test_x = target_xs[test_indices]
    test_y = target_ys[test_indices]

    return (
        train_id,
        train_x,
        train_y,
        test_id,
        test_x,
        test_y,
    )


def save_to_pickle():
    result = {}
    # client = AEManifoldTorchClient(AEManifoldUseCase.TRBO_DEV)
    for task in ["rf", "lr", "svm", "xgb", "nn"]:
        for folder in os.listdir(
            f"/home/yl9959/mtpfn/datasets/hpobench/{task}"
        ):
            if folder.endswith(".zip"):
                continue
            task_id = int(folder)
            print("\tReading", folder)
            df = pd.read_parquet(
                f"/home/yl9959/mtpfn/datasets/hpobench/{task}/{task_id}/{task}_{task_id}_data.parquet.gzip"
            )
            if task == "rf":
                df = df[(df["n_estimators"] == 512) & (df["subsample"] == 1)]
            elif task == "lr":
                df = df[(df["iter"] == 1000) & (df["subsample"] == 1.0)]
            elif task == "svm":
                df = df[(df["subsample"] == 1)]
            elif task == "xgb":
                df = df[(df["n_estimators"] == 2000) & (df["subsample"] == 1)]
            elif task == "nn":
                df = df[(df["iter"] == 243) & (df["seed"] == 8916)]

            result[task_id] = df

        # key = client.torch_save(result)
        pickle.dump(result, open(f"/scratch/yl9959/mtpfn/datasets/hpobench_{task}.pkl", "wb"))
        print(f"Saved {task} to /scratch/yl9959/mtpfn/datasets/hpobench_{task}.pkl", len(result))


if __name__ == "__main__":
    save_to_pickle()