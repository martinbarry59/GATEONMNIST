import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
import sys
import argparse

path = "/".join(os.path.dirname(os.path.realpath(__file__)).split("/")[:-1])
sys.path.insert(1, "{0}/pkg/".format(path))
import torch
import general_utils

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import simulator

parser = general_utils.get_options(argparse.ArgumentParser())
args = parser.parse_args()
if args.approach == "pgateon":
    import pGateON as Network
elif args.approach == "ngateon":
    import nGateON as Network

dic_models = {}
n_average = 5
# simulation
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


models = []


lambdas = initials_to_lambdas(list(map(str, args.lambdas.split("-"))))
print(lambdas)
params = {}
params["lambdas"] = args.lambdas
params["n_layers"] = args.n_layers
params["hidden"] = args.hidden
params["epochs"] = args.epochs
params["approach"] = args.approach
params["fake_context"] = args.fake_context
params["skip_context"] = args.skip_context
params["dic"] = args.dic
params["task"] = args.task
params["all_steps"] = args.all_steps
params["change_timing"] = args.change_timing
params["epsilon"] = args.epsilon
params["lr"] = args.lr
params["conv"] = args.conv
params["fed"] = args.fed
labels = 10
folder = ""
for k, v in params.items():
    print(k, v)
    if k in ["approach", "n_layers", "hidden", "epochs", "epsilon", "task", "lambdas"]:
        folder += "{0}_{1}_".format(k, v)

folder += "conv" * params["conv"]
folder += "fed" * params["fed"]
params["folder"] = folder
n_contexts = 110  # maximum number of contexts
load_models = 0
n_models = len(lambdas)
for i in range(n_models):
    dic_models[lambdas[i]] = []
    for episode in range(n_average):
        str_ = "-----------------episode number {0} for lambda = {1} -----------------".format(
            episode + 1, lambdas[i]
        )
        str_ = str_.center(80, "-")
        if load_models == 0:
            model = Network.network(
                n_contexts,
                params["n_layers"],
                labels,
                params["hidden"],
                epsilon=params["epsilon"],
                lambda_=lambdas[i],
                input_size=784,
                conv=params["conv"],
                fed=params["fed"],
            ).to(device)

        if load_models != 0:

            model = torch.load(path + "/results/models/network_{0}".format(load_models))
            model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])
        info_dic = simulator.run_simulation(model, optimizer, params)

        general_utils.plot_end_sim(
            model, info_dic, n_layers=params["n_layers"], context=1
        )
        dic_models[lambdas[i]] += [info_dic]
        keys = [
            "continual_test_error",
            "test_error",
            "epoch_error",
            "test_accuracy",
            "continual_test_accuracy",
        ]
        skip = []
        tmp_file = "results.pkl".format(params["hidden"], params["epochs"])
        general_utils.plot_average(
            dic_models, keys, skip, tmp_file, folder, savemodels=True
        )


else:
    print("Not existing...")
