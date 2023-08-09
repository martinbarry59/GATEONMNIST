import os
import sys

path = "/".join(os.path.dirname(os.path.realpath(__file__)).split("/")[:-1])
sys.path.insert(1, "{0}/pkg/".format(path))
import matplotlib.pyplot as plt
import torch
import torchvision
import numpy as np
import pickle

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_options(parser):

    parser.add_argument(
        "--n_layers", type=int, help="number of hidden layer + 1", default=5
    )
    parser.add_argument(
        "--hidden", type=int, help="number of hidden neurons ", default=5000
    )
    parser.add_argument(
        "--epochs", type=int, help="number of batch training ", default=2500
    )
    parser.add_argument(
        "--approach", type=str, help="pgateon or gateon ", default="ngateon"
    )
    parser.add_argument(
        "--fake_context",
        help="use a fake context for task n (list, deprecated)",
        default=None,
    )
    parser.add_argument(
        "--skip_context",
        type=list,
        help="Skip a context during traing (list) ",
        default=[],
    )
    parser.add_argument(
        "--dic", help="if you want to reuse previous data", default=None
    )
    parser.add_argument(
        "--task",
        type=str.upper,
        choices=["PM", "RM", "SLM"],
        help="which meta task to run ",
        default="PM",
    )
    parser.add_argument(
        "--all_steps", type=bool, help="number of batch training ", default=False
    )
    parser.add_argument(
        "--fed", type=bool, help="feeding the tasks or not", default=False
    )
    parser.add_argument("--conv", type=int, help=" Using conv layers", default=0)
    parser.add_argument("--lr", type=float, help=" learning rate", default=5e-3)
    parser.add_argument(
        "--change_timing", type=int, help="time wihtin the same task ", default=500
    )
    parser.add_argument(
        "--epsilon", type=float, help="number of batch training ", default=0
    )
    parser.add_argument(
        "--lambdas",
        type=str.lower,
        help="can be anything in: ['neuron age sparse','no context sparse','context only sparse','age only sparse']; use initials and dash between models",
        default="nas",
    )
    return parser


def plot_average(
    dic_, keys, skip, file=None, folder=None, minidx=0, maxidx=-1, savemodels=False
):
    if folder is not None and not os.path.exists(path + "/results/" + folder):
        os.makedirs(path + "/results/" + folder)
    savedic = {}
    for key in keys:
        m = 0
        figs = []
        axs = []
        for lambda_, trials in dic_.items():
            all_errors = {}
            all_errors[0] = []
            if lambda_ not in skip:
                to_plot = None
                for dic in trials:
                    if "continual" in key:
                        for task, values in dic[key].items():
                            if task in all_errors.keys():
                                all_errors[task] += [values]
                            else:
                                all_errors[task] = [values]
                    else:
                        all_errors[0] += [dic[key]]

                for plots in range(len(all_errors.keys())):
                    average = torch.mean(torch.Tensor(all_errors[plots]), dim=0)
                    std = torch.std(torch.Tensor(all_errors[plots]), dim=0)
                    if len(figs) <= plots:
                        fig, ax1 = plt.subplots(num=plots)
                        figs += [fig]
                        axs += [ax1]

                    if "continual" in key:
                        plot_key = key + "_Task_{}".format(plots + 1)
                    else:
                        plot_key = key
                    savedic[plot_key + "_std_lambda_{}".format(lambda_)] = std
                    savedic[plot_key + "_average_lambda_{}".format(lambda_)] = average
                    axs[plots].fill_between(
                        torch.arange(average.shape[0]),
                        average - std,
                        average + std,
                        alpha=0.5,
                    )
                    axs[plots].plot(average, label=lambda_)
                    if m == 0:
                        savedic["Context"] = trials[0]["context"]

            m += 1
        for fig_n in range(len(figs)):
            if "continual" in key:
                plot_key = key + "_Task_{}".format(fig_n + 1)
            else:
                plot_key = key
            axs[fig_n].set_title(plot_key.replace("_", " "))
            axs[fig_n].set_xlabel("Stimuli")
            axs[fig_n].set_ylabel("Error")
            figs[fig_n].savefig(
                path + "/results/" + folder + "/" + plot_key + ".pdf",
                bbox_inches="tight",
            )
        plt.close("all")
    if file is not None and savemodels:
        print(path + "/results/" + file)
        f = open(path + "/results/" + folder + "/average_" + file, "wb")
        pickle.dump(savedic, f)
        f = open(path + "/results/" + folder + "/FULL_" + file, "wb")
        pickle.dump(dic_, f)


def plot_end_sim(model=None, info_dic=None, n_layers=2, context=1):
    if model is not None:
        plt.show()
    if info_dic is not None:
        key = "n_layer_{0}_use_context_{1}".format(n_layers, context)
        fig, ax1 = plt.subplots()

        ax2 = ax1.twinx()
        ax1.plot(info_dic["epoch_error"], "g-")
        ax2.plot(info_dic["context"], "b-")

        ax1.set_xlabel("Stimuli")
        ax1.set_ylabel("Error", color="g")
        ax2.set_ylabel("Context", color="b")
        plt.show()


def load_data(train, transform, batch_size, context_size, angle=False, conv=False):
    loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(
            "data", train=train, download=True, transform=transform
        ),
        batch_size=batch_size,
        shuffle=False,
    )
    iter_ = iter(loader)
    images, labels = next(iter_)
    images, labels = images.to(device), labels.to(device)
    if angle != 0:
        images = torchvision.transforms.functional.rotate(images, angle)
    if conv == False:
        images = torch.flatten(images, start_dim=-3, end_dim=-1)
    return images.detach(), labels.detach()


def get_transform(random=True):
    if random:
        rng_permute = np.random.RandomState()
        idx_permute = torch.from_numpy(rng_permute.permutation(784))
    else:
        idx_permute = torch.arange(0, 784)
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Lambda(
                lambda x: x.view(-1)[idx_permute].view(1, 28, 28)
            ),
        ]
    )
    return transform


def remap_values(remapping, x):
    index = torch.bucketize(x.ravel(), remapping[0])
    return remapping[1][index].reshape(x.shape)
