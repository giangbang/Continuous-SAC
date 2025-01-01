import os
import pandas as pd
import matplotlib.pyplot as plt
import random


def get_random_color():
    return (random.random(), random.random(), random.random())


def get_folder(dir):
    items = os.listdir(dir)
    return [item for item in items if os.path.isdir(os.path.join(dir, item))]


def plot_exp_res(exp_dir, algo, env, color, ax):
    dfs = []
    for root, _, files in os.walk(exp_dir):
        for file in files:
            if file.endswith(".csv"):
                file_path = os.path.join(root, file)
                try:
                    df = pd.read_csv(file_path)
                    dfs.append(df)
                except Exception as e:
                    print(f"Error reading the CSV file {file_path}: {e}")
    assert len(dfs) == 1, "single seed for now"
    df = dfs[0]
    ax.plot(
        df["time/total_timesteps"], df["rollout/ep_rew_mean"], color=color, label=algo
    )
    ax.set_title(env)


if __name__ == "__main__":
    results_folder = "results"
    random.seed(2024)

    envs = get_folder(results_folder)
    n_envs = len(envs)

    ncol, nrow = (n_envs + 1) // 2, 2
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 5, nrow * 4))

    plt_colors = dict()

    for i_plt, env in enumerate(envs):
        exp_dir = os.path.join(results_folder, env)
        algos = get_folder(exp_dir)
        for algo in algos:
            plt_colors[algo] = plt_colors.get(algo, get_random_color())
            algo_exp_dir = os.path.join(exp_dir, algo)
            plot_exp_res(
                algo_exp_dir,
                algo,
                env,
                plt_colors[algo],
                axes[i_plt // ncol, i_plt % ncol],
            )

    # Combine all handles and labels for legend
    handles, labels = [], []
    for ax_row in axes.flat:
        for handle, label in zip(*ax_row.get_legend_handles_labels()):
            if label not in labels:
                handles.append(handle)
                labels.append(label)

    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(plt_colors),
        frameon=False,
    )
    plt.tight_layout()
    plt.savefig("assets/plot.png")
