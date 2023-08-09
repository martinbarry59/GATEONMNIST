# +
import os
import sys

path = "/".join(os.path.abspath(__file__).split("/")[:-1])
sys.path.insert(1, "{0}/pkg/".format(path))
import torch
import general_utils
import torchvision
import tqdm
import matplotlib.pyplot as plt
import copy
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def save_data(dic, tmp_dic):
    keys = [
        "epoch_error",
        "test_error",
        "continual_test_error",
        "test_accuracy",
        "continual_test_accuracy",
        "change_points",
        "availabilities_synapses",
        "availabilities_neurons",
        "availabilities_filters",
        "context",
    ]
    for key in keys:
        if key in dic.keys():
            dic[key] += tmp_dic[key]
        else:
            dic[key] = tmp_dic[key]
    return dic


class simulator:
    def __init__(self, params, model, optimizer):
        self.epochs = params["epochs"]
        self.folder = params["folder"]
        self.fake_context = params["fake_context"]  ## deprecated
        self.skip_context = params[
            "skip_context"
        ]  ##  array of context you want to skip if you want to skip context number n
        self.dic = params["dic"]  ##  if you want to save data on top of old simulator
        if self.dic is None:
            self.dic = {}
        self.task = params["task"]  ##  which tasks
        self.all_steps = params["all_steps"]  ## if we save at all time steps
        self.change_timing = params[
            "change_timing"
        ]  ## steps before  environment change
        self.model = model
        self.optimizer = optimizer
        if self.task == "SP":
            split_labels = torch.arange(10)
            idx = torch.randperm(split_labels.nelement())
            self.split_labels = (
                split_labels.view(-1)[idx].view(split_labels.size()).split(2, -1)
            )
        self.simulation_init()
        self.modelpath = path + "/../results/" + params["folder"]
        if not os.path.exists(self.modelpath + "/models/"):

            os.makedirs(self.modelpath + "/models/")

    def init_availabilities(self):
        self.tmp_dic["availabilities_synapses"] = {}
        self.tmp_dic["availabilities_neurons"] = {}
        self.tmp_dic["availabilities_filters"] = {}
        if self.all_steps:
            steps = 1
        else:
            steps = self.change_timing
        if self.model.conv:
            for layer in range(self.model.n_conv):
                self.tmp_dic["availabilities_filters"][layer] = (
                    torch.zeros(
                        self.model.conv_availabilities[layer].flatten().shape[0],
                        int(self.epochs / steps),
                    )
                    .detach()
                    .cpu()
                )
        for layer in range(self.model.n_layers - 1):
            self.tmp_dic["availabilities_neurons"][layer] = (
                torch.zeros(
                    self.model.availabilities[layer].flatten()[::2000].shape[0],
                    int(self.epochs / steps),
                )
                .detach()
                .cpu()
            )

    def init_tmp_dic(self):
        tmp_dic = {}
        tmp_dic["active_context"] = [0]
        tmp_dic["epoch_error"] = []
        tmp_dic["test_error"] = []
        tmp_dic["test_accuracy"] = []
        tmp_dic["continual_test_error"] = []
        tmp_dic["reconstruction_errors"] = []
        tmp_dic["continual_test_accuracy"] = {}
        tmp_dic["continual_test_accuracy"][0] = []
        tmp_dic["continual_test_error"] = {}
        tmp_dic["continual_test_error"][0] = []
        tmp_dic["backprop_errors"] = []
        tmp_dic["change_points"] = [0]
        tmp_dic["continual_images"] = []
        tmp_dic["continual_labels"] = []
        tmp_dic["continual_contexts"] = []
        tmp_dic["magnitude"] = []
        tmp_dic["context"] = []
        self.tmp_dic = tmp_dic

    def new_context(self, epoch):
        self.run_continual_error()
        tmp = {}
        tmp[self.model.lambda_] = [self.tmp_dic]
        keys = [
            "continual_test_error",
            "test_error",
            "epoch_error",
            "test_accuracy",
            "continual_test_accuracy",
        ]
        print("max acc", 100 * self.tmp_dic["test_accuracy"][-1].item(), self.best_time)
        general_utils.plot_average(
            tmp, keys, self.skip_context, "results.pkl", "tmp_" + self.folder
        )
        self.best_model.print_availabilities()

        if (
            not self.all_steps and "age" in self.model.lambda_
        ):  ## save availabilities only at change-points
            self.save_availabilities(int(epoch / self.change_timing))
        self.context_n = self.context_n + 1

        self.new_dataset()

    def update_contexts(self):
        test_context = torch.zeros(
            self.test_labels.shape[0], self.model.n_contexts, self.model.n_layers - 1
        ).to(device)
        test_context = self.model.active_context
        self.tmp_dic["continual_contexts"][-1] = test_context
        self.test_context = test_context

    def only_take_labels(self, image, labels):
        indices = torch.zeros_like(labels)
        for label in self.current_labels:
            indices[labels == label] = 1
        indices = indices >= 1
        return image[indices], labels[indices]

    def new_dataset(self):
        self.transform = general_utils.get_transform(random=(self.task == "PM"))
        if self.task == "RM":
            self.angle = torch.randint(360, [1])[0].item()
            self.model.angles += [self.angle]

        self.test_images, self.test_labels = general_utils.load_data(
            False,
            self.transform,
            self.test_batch,
            self.model.n_contexts,
            self.angle,
            self.model.conv,
        )
        self.imavailabilities_all, self.labels_all = general_utils.load_data(
            True,
            self.transform,
            60000,
            self.model.n_contexts,
            self.angle,
            self.model.conv,
        )

        if self.task == "SLM":
            self.remapping = torch.arange(0, self.model.output_size).to(
                device
            ), torch.randperm(self.model.output_size).to(device)
            self.test_labels = general_utils.remap_values(
                self.remapping, self.test_labels
            )
        elif self.task == "SP":
            self.current_labels = self.split_labels[self.context_n % 5]
            self.test_images, self.test_labels = self.only_take_labels(
                self.test_images, self.test_labels
            )

        if self.context_n not in self.tmp_dic["continual_test_accuracy"].keys():
            self.tmp_dic["continual_test_accuracy"][self.context_n] = []
            self.tmp_dic["continual_test_error"][self.context_n] = []
            self.tmp_dic["continual_images"] += [self.test_images.detach().clone()]
            self.tmp_dic["continual_labels"] += [self.test_labels.detach().clone()]
        self.tmp_dic["continual_contexts"] += [self.test_context]

    def simulation_init(self):
        self.angle = 0
        self.context_n = 0
        self.test_context = 0
        self.test_batch = 1000
        self.init_tmp_dic()
        self.new_dataset()
        self.init_availabilities()

    def run_test_errors(self, epoch):

        labels_hot = torch.nn.functional.one_hot(
            self.test_labels.to(torch.int64), self.model.output_size
        ).to(torch.float32)
        test_prediction = self.model.forward(
            self.test_images, context_input=self.test_context
        )
        test_error = torch.mean((test_prediction - labels_hot) ** 2).detach().cpu()
        if test_error < self.best_loss:  # we don't save loss that we won't use

            self.best_loss = test_error
            self.best_model = copy.deepcopy(self.model)
            self.best_time = epoch
            self.tmp_dic["test_error"] += [
                torch.mean((test_prediction - labels_hot) ** 2).detach().cpu()
            ]
            self.tmp_dic["test_accuracy"] += [
                torch.mean(
                    1.0
                    * (
                        torch.argmax(test_prediction, dim=1)
                        == torch.argmax(labels_hot, dim=1)
                    )
                )
                .detach()
                .cpu()
            ]

        else:
            self.tmp_dic["test_error"] += [self.tmp_dic["test_error"][-1]]
            self.tmp_dic["test_accuracy"] += [self.tmp_dic["test_accuracy"][-1]]
        ## Make tests predictions

    def run_continual_error(self):
        for task in range(len(self.tmp_dic["continual_images"])):
            continual_test_images = self.tmp_dic["continual_images"][task]
            labels_hot = (
                torch.nn.functional.one_hot(
                    self.tmp_dic["continual_labels"][task].to(torch.int64),
                    self.model.output_size,
                )
                .to(torch.float32)
                .detach()
            )
            continual_testcontext = self.tmp_dic["continual_contexts"][task]
            continual_test_prediction = self.best_model.forward(
                continual_test_images, context_input=continual_testcontext
            )
            ## Saving errors
            self.tmp_dic["continual_test_accuracy"][task] += [
                torch.mean(
                    1.0
                    * (
                        torch.argmax(continual_test_prediction, dim=1)
                        == torch.argmax(labels_hot, dim=1)
                    )
                )
                .detach()
                .cpu()
            ]
            self.tmp_dic["continual_test_error"][task] += [
                torch.mean((continual_test_prediction - labels_hot) ** 2).detach().cpu()
            ]

    def save_availabilities(self, step):
        if self.model.conv:
            for layer in range(self.model.n_conv):
                self.tmp_dic["availabilities_filters"][layer][:, step] = (
                    self.model.tmp_conv_availabilities[layer].flatten().detach().cpu()
                )
        for layer in range(self.model.n_layers - 1):
            ages = self.model.tmp_availabilities[layer]
            self.tmp_dic["availabilities_neurons"][layer][:, step] = (
                ages.flatten()[::2000].clone().detach().cpu()
            )  # We don't save all its too much especially for p-GateON

    def start_new_context(self, params):
        self.best_loss = np.inf
        self.model = self.best_model
        self.model.fixing_ages()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=params["lr"])


def run_simulation(model, optimizer, params):

    ## initialise simulation
    MNIST_simulator = simulator(params, model, optimizer)

    save_frequency = 1

    ## load first images ##
    index_all = torch.zeros(MNIST_simulator.labels_all.shape[0])
    batch_size = 1000
    index_all[:batch_size] = 1

    MNIST_simulator.best_loss = np.inf
    for epoch in tqdm.tqdm(range(MNIST_simulator.epochs)):
        with torch.no_grad():
            MNIST_simulator.update_contexts()

            ## loading data

            if MNIST_simulator.all_steps and "age" in MNIST_simulator.model.lambda_:
                MNIST_simulator.save_availabilities(epoch)

            ## We go to new context ! ##
            if (epoch + 1) % MNIST_simulator.change_timing == 0:
                MNIST_simulator.new_context(epoch)
                if (
                    MNIST_simulator.model.context_fed == True
                ):  # if we feed the context we automatically update
                    MNIST_simulator.start_new_context(params)

            ## select train images randomly
            index_all = index_all[torch.randperm(len(index_all))] == 1
            images = MNIST_simulator.imavailabilities_all[index_all]
            labels = MNIST_simulator.labels_all[index_all]

            if MNIST_simulator.task == "SLM":
                labels = general_utils.remap_values(MNIST_simulator.remapping, labels)
            elif MNIST_simulator.task == "SP":
                images, labels = MNIST_simulator.only_take_labels(images, labels)

        loss, _, _ = MNIST_simulator.model(
            images, labels, training=True, context_input=MNIST_simulator.context_n
        )

        # if network detected a switch point
        if (
            MNIST_simulator.model.context_fed == False
            and MNIST_simulator.model.new_context == True
        ):
            MNIST_simulator.best_model.set_from_model_context_params(
                MNIST_simulator.model
            )  # set ages for best model
            MNIST_simulator.start_new_context(params)
            MNIST_simulator.model.new_context = False
            loss, _, _ = MNIST_simulator.model(images, labels, training=True)

        MNIST_simulator.optimizer.zero_grad()

        if epoch % save_frequency == 0:
            with torch.no_grad():
                MNIST_simulator.run_test_errors(epoch % MNIST_simulator.change_timing)
                MNIST_simulator.tmp_dic["epoch_error"] += [loss.detach().cpu()]
                MNIST_simulator.tmp_dic["context"] += [MNIST_simulator.context_n]

        loss.backward()
        # obstruct the pathways
        if params["approach"] == "pgateon":
            for layer in range(len(MNIST_simulator.model.FF_layers) - 1):
                MNIST_simulator.model.update_ages(
                    layer, type=MNIST_simulator.model.causality
                )
            if MNIST_simulator.model.conv:
                for layer in range(len(MNIST_simulator.model.FF_convs)):
                    MNIST_simulator.model.update_availabilities_conv(layer)
        MNIST_simulator.model.update_learning_rate()

        MNIST_simulator.optimizer.step()
    torch.save(
        MNIST_simulator.best_model,
        MNIST_simulator.modelpath
        + "/models/network_{0}".format(MNIST_simulator.model.lambda_),
    )
    MNIST_simulator.dic = save_data(MNIST_simulator.dic, MNIST_simulator.tmp_dic)
    return MNIST_simulator.dic


# -
