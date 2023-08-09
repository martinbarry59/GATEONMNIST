import torch
import plotly
import pickle
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
import sys

path = os.path.abspath("..")  # +'/generalized_context'
sys.path.insert(1, "{0}/pkg/".format(path))
print(path)
import general_utils
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import random
import matplotlib.colors as colors
import numpy as np
import matplotlib.cm as cm
import copy
import pandas as pd


def load_results(folder):
    file = "average_results.pkl"
    figure_path = folder + "/"
    split_folder = folder.split("_")
    epsilon = split_folder[split_folder.index("epsilon") + 1]
    model = split_folder[split_folder.index("task") + 1]
    epochs = split_folder[split_folder.index("epochs") + 1]
    model_str = model
    conv = ""
    if "conv" in folder:
        conv = "conv"

    save_path = path + "/results/{0}{1}/".format(model_str, conv)

    if "24899" in folder:
        full = "_full"
    else:
        full = ""
    with open(figure_path + file, "rb") as handle:
        dic_samples = pickle.load(handle)

    lambdas = []
    network_models = ["aos", "nas", "cos", "ncs"]
    for network_model in network_models:
        if network_model in folder:
            lambdas += [network_model]
    lambdas = initials_to_lambdas(lambdas)
    if len(lambdas) == 0:
        lambdas = ["neuron age sparse"]
    print(
        "------------ Working on model ",
        model,
        " method ",
        lambdas[0],
        " with epsilon",
        epsilon,
        "------------",
    )
    if len(lambdas) == 0:
        lambdas = ["neuron age sparse"]
    return dic_samples, epsilon, model, epochs, lambdas, full, save_path


def initials_to_lambdas(lambdas):
    new = []
    for lambda_ in lambdas:
        if lambda_ == "nas":
            new += ["neuron age sparse"]
        elif lambda_ == "ncs":
            new += ["no context sparse"]
        elif lambda_ == "cos":
            new += ["context only sparse"]
        elif lambda_ == "aos":
            new += ["age only sparse"]
    return new


def get_color(string):
    default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    colors = {
        "neuron age sparse 0.0": default_colors[0],
        "neuron age sparse 0": default_colors[0],
        "neuron age sparse 1.0": default_colors[1],
        "neuron age sparse 0.5": default_colors[2],
        "neuron age sparse 0.2": default_colors[1],
        "neuron age sparse 0.0024": default_colors[2],
        "no context sparse": default_colors[3],
        "age only sparse": default_colors[4],
        "context only sparse": default_colors[5],
    }  # , 'pink', 'gray', 'olive', 'c']

    return colors[string]


def moving_average(x, window_size):
    # Pad input tensor with zeros to handle edge cases
    padding = (window_size - 1) // 2
    x_padded = torch.nn.functional.pad(x, (padding, padding), mode="constant", value=0)
    x_padded[0], x_padded[1] = x_padded[2], x_padded[2]
    x_padded[-1], x_padded[-2] = x_padded[-3], x_padded[-3]

    # Construct the convolution kernel with weights equal to 1/window_size
    kernel_size = (1, window_size)

    kernel = torch.ones(kernel_size, dtype=x.dtype, device=x.device) / window_size
    # Apply the moving average filter using a 1D convolution
    y = torch.nn.functional.conv1d(
        x_padded.unsqueeze(0).unsqueeze(0), kernel.unsqueeze(0), padding=0, stride=1
    )[0, 0]
    return y


def final_values(
    dic,
    lambdas,
    skips=[],
    epsilon=0,
    fig_num=0,
    full="",
    steps=500,
    ntasks=10,
    save_path=None,
):

    for lambda_ in lambdas:
        key = "test_accuracy_{1}_lambda_{0}"
        max_indx = ntasks * steps
        x = dic[key.format(lambda_, "average")][:max_indx].reshape(-1, steps)
        std = dic[key.format(lambda_, "std")][:max_indx].reshape(-1, steps)

        if "neuron age" in lambda_:
            epsi = " {0}".format(epsilon)
        else:
            epsi = ""

        # plotting
        fig1 = plt.figure(1 + fig_num)
        label = lambda_ + epsi + full
        plt.plot(moving_average(x[:, 2], 5), label=label)  # ,color = get_color(label))
        fig2 = plt.figure(2 + fig_num)

        mean = moving_average(x[:, -2], 5)
        std = moving_average(std[:, -2], 5) / 5 ** (
            0.5
        )  ## number of simulations ran to cumpute std == 5
        plt.plot(mean, label=label, color=get_color(label))
        plt.fill_between(
            torch.arange(len(mean)),
            mean - std,
            mean + std,
            alpha=0.2,
            color=get_color(label),
        )
        plt.ylim(top=1, bottom=0.8)
    fig1.savefig(save_path + "single_batch.pdf")
    fig2.savefig(save_path + "final_accuracies.pdf")


def plot_test_accurary(
    dic, lambdas, epsilon=0, fig_num=0, full="", steps=500, ntasks=10, save_path=None
):

    abss = torch.arange(steps - 2) + 1
    comparison = torch.zeros(len(lambdas), steps - 1)
    number_saved = 0
    fig = plt.figure(3 + fig_num)
    for m in range(len(lambdas)):
        try:
            comparison[m] = dic["test_accuracy_average_lambda_{0}".format(lambdas[m])][
                0 : steps - 1
            ]
            number_saved += 1
        except:
            continue
    if "no context" in lambdas[0]:
        plt.plot(
            abss,
            torch.sum(comparison[:, :-1], dim=0) / number_saved,
            "k",
            label="comparison",
        )
        plt.xscale("log")
    for lambda_ in lambdas:

        accuracies = dic["test_accuracy_average_lambda_{0}".format(lambda_)]

        max_indx = ntasks * steps
        if "age" in lambda_:
            x = accuracies[steps:max_indx].reshape(-1, steps)
        else:
            x = accuracies[steps - 1 : max_indx - 1].reshape(-1, steps)
        v = torch.mean(x[:, 0:], axis=0).clone()[:-1]
        std = torch.std(x[:, 0:], axis=0).clone()[:-1] / x.shape[0] ** (0.5)
        abss = torch.arange(steps) + 1
        if "neuron age" in lambda_:
            epsi = " {0}".format(epsilon)
        else:
            epsi = ""
        label = lambda_ + epsi + full
        plt.plot(abss[:-1], v, label=label, color=get_color(label))
        plt.fill_between(abss[:-1], v - std, v + std, alpha=0.2, color=get_color(label))
        plt.xscale("log")
    plt.savefig(save_path + "retraining_speed.pdf")


def mean_continual_accuracy(
    dic,
    lambdas,
    results,
    epsilon=0,
    fig_num=0,
    full="",
    steps=500,
    ntasks=100,
    save_path=None,
    model=None,
):

    for lambda_ in lambdas:
        mean = []
        remean = []
        std = []
        rate_loss = 0
        fig = plt.figure(4 + fig_num)
        task_detected = []

        for task in range(1, ntasks - 1):
            #             try:
            x = dic[
                "continual_test_accuracy_Task_{1}_average_lambda_{0}".format(
                    lambda_, task
                )
            ]
            task_detected += [1 * (x[-2] < 0.8)]
            rate_loss += 100 * (x[0] - torch.mean(x[1 : ntasks - task])) / ntasks
            mean += [torch.mean(x[: ntasks - task])]
            std += [torch.std(x[: ntasks - task])]
        task_detected = torch.cumsum(
            torch.Tensor(task_detected)[1:], dim=0
        )  # / torch.arange(1,99)
        max_indx = ntasks * steps
        x = dic["test_accuracy_average_lambda_{0}".format(lambda_)][:max_indx].reshape(
            -1, steps
        )
        if "no context" in lambda_:
            remean = torch.mean(torch.max(x[0:1], dim=1).values)
        else:
            remean = torch.mean(torch.max(x, dim=1).values)
        if len(mean) > 0:
            if "neuron age" in lambda_:
                epsi = " {0}".format(epsilon)
            else:
                epsi = ""
            label = lambda_ + epsi + full
            plt.plot(mean, label=label, color=get_color(label))
            results[model][lambda_ + epsi] = {}
            results[model][lambda_ + epsi]["ACC"] = round(
                100 * torch.mean(torch.tensor(mean)).item(), 2
            )
            print(
                results[model][lambda_ + epsi]["ACC"],
                round(100 * torch.mean(torch.tensor(std)).item(), 2),
                round(rate_loss.item(), 2),
            )
            results[model][lambda_ + epsi]["RE_ACC"] = round(100 * remean.item(), 2)
            results[model][lambda_ + epsi]["FR"] = round(rate_loss.item(), 2)
            results[model][lambda_ + epsi]["task_detected"] = task_detected
            plt.fill_between(
                torch.arange(len(mean)),
                torch.Tensor(mean) - torch.Tensor(std),
                torch.Tensor(mean) + torch.Tensor(std),
                alpha=0.2,
                color=get_color(label),
            )
    plt.savefig(save_path + "mean_continual_accuracy_over_time.pdf")
    return results


def count_redundancy(x):
    col_red = torch.sum(1.0 * (x > 0), dim=1)  ## prob to be used per column
    col_red = col_red[
        col_red > 0
    ]  ## take only the one that have been used at least once !
    return round(
        torch.mean(col_red).item(), 2
    )  # return mean probabilty of being reused


def redundancy_matrix(W, model):
    mat = torch.zeros(W.shape[1], W.shape[1])
    for i in range(W.shape[1]):
        for j in range(W.shape[1]):
            act = W[:, i]
            active1 = act > 0  # torch.mean(act) #+ 2 * torch.std(act)

            act = W[:, j]
            active2 = act > 0  # torch.mean(act) #+ 2 * torch.std(act)

            mat[i, j] = torch.mean(1.0 * (active1 * active2))
    return mat


def compute_correl(results, model, path, layer, type_="forward"):
    idx = 100
    results["Redundancy_layer_{0}_{1}".format(layer, type_)] = count_redundancy(
        model.projections_context[layer].weight[:, :idx].detach()
    )

    permutations = torch.arange(idx)
    if "RM" in path:
        angles_idx = torch.argsort(torch.Tensor(model.angles))
        permutations = angles_idx
    if type_ == "forward":
        W = model.projections_context[layer].weight[:, :idx]
    else:
        print("doing conv")
        W = model.conv_context_projections[layer].weight[:, :idx]
    W = torch.nn.ReLU()(W[:, permutations].detach())
    m = torch.corrcoef(W.T) - torch.eye(idx)
    return m, W


def model_correlations(path, lambdas, results, model_str, epsilon):
    total_W = None
    total_net = None
    for type_ in lambdas:
        #         try:

        model_path = path + type_
        print("model: ", type_)

        model = torch.load(model_path, map_location=torch.device("cpu"))
        print(len(model.context_error))
        kill
        if "neuron age" in type_:
            epsi = " {0}".format(epsilon)
        else:
            epsi = ""
        results[model_str][type_ + epsi] = {}
        if "conv" in path:
            for layer in range(model.n_conv):
                m, W = compute_correl(
                    results[model_str][type_ + epsi], model, path, layer, "conv"
                )
                if total_W is None:
                    total_W = W
                    total_net = m
                else:
                    total_W = torch.cat([total_W, W], dim=0)
                    total_net = torch.cat([total_net, m], dim=1)

        for layer in range(model.n_layers - 1):
            m, W = compute_correl(
                results[model_str][type_ + epsi], model, path, layer, "forward"
            )
            if total_W is None:
                total_W = W
                total_net = m
            else:
                total_W = torch.cat([total_W, W], dim=0)
                total_net = torch.cat([total_net, m], dim=1)

        vmin = -0.1
        vmax = 0.5
        norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
        m = torch.corrcoef(total_W.T) - torch.eye(100)
        plt.imshow(m, cmap="bwr", norm=norm)
        plt.savefig(path + "Corr_one.pdf".format(layer, type_), bbox_inches="tight")
        plt.colorbar()
        plt.show()
        vmin = -0.1
        vmax = 0.5
        norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
        plt.imshow(total_net, cmap="bwr", norm=norm)
        plt.savefig(
            path + "Corr_total_net.pdf".format(layer, type_), bbox_inches="tight"
        )
        plt.colorbar()
        plt.show()
    return results
