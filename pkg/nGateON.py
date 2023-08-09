import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import torch.nn.functional as F
import numpy as np

import copy


class network(torch.nn.Module):
    def __init__(
        self,
        n_contexts,
        n_layers,
        output_size,
        hidden=100,
        lambda_=0.1,
        epsilon=0,
        input_size=None,
        conv=True,
        causality="full",
        n_conv=2,
        channels=[1, 256, 152],
        fed = False
    ):
        super().__init__()
        """
        In the constructor we instantiate four parameters and assign them as
        member parameters.
        """
        self.hidden = hidden
        self.n_layers = n_layers
        self.n_contexts = n_contexts
        self.output_size = output_size
        self.lambda_ = lambda_
        self.context_fed = fed
        self.init_others()

        self.epsilon = epsilon
        self.conv = conv
        self.causality = causality
        self.new_context = False

        if conv:
            self.n_conv = n_conv
            channels = channels
        if input_size == None:
            self.input_size = self.output_size
        else:
            self.input_size = input_size

        # instantiating the conv layers
        if self.conv:
            self.conv_availabilities = []
            self.FF_convs = []
            self.conv_context_projections = []
            for layer in range(self.n_conv):

                self.FF_convs += [
                    torch.nn.Conv2d(channels[layer], channels[layer + 1], 3)
                ]
                self.conv_availabilities += [torch.ones(channels[layer + 1]).to(device)]
                proj = torch.nn.Linear(self.n_contexts, channels[layer + 1])
                proj.bias.data.fill_(0.0)
                proj.bias.requires_grad = False
                self.conv_context_projections += [proj]
        self.pool = torch.nn.MaxPool2d(4)
        # instantiating the FC layers
        for layer in range(self.n_layers):

            if self.n_layers == 1:
                FF_linears = torch.nn.Linear(self.input_size, self.output_size)
            elif layer == 0:
                i = self.input_size
                if self.conv:
                    i = int(channels[-1] * 36)  ## fixed for our model
                o = self.hidden
            elif layer == self.n_layers - 1:
                i = self.hidden
                o = self.output_size

            else:
                i = self.hidden
                o = self.hidden

            ## context_projection
            if layer != self.n_layers - 1:
                proj = torch.nn.Linear(self.n_contexts, o)
                proj.bias.data.fill_(0.0)
                proj.bias.requires_grad = False
                self.projections_context += [proj]
            self.batch_norms += [torch.nn.BatchNorm1d(o).to(device)]
            if layer == 0:
                self.availabilities += [torch.ones(o).to(device)]
            else:
                self.availabilities += [torch.ones(o).to(device)]
            self.FF_layers += [torch.nn.Linear(i, o).to(device)]

        if self.conv:
            self.tmp_conv_availabilities = copy.deepcopy(self.conv_availabilities)
            self.conv_context_projections = torch.nn.ModuleList(
                self.conv_context_projections
            )
            self.FF_convs = torch.nn.ModuleList(self.FF_convs)

        self.tmp_availabilities = copy.deepcopy(self.availabilities)
        self.projections_context = torch.nn.ModuleList(self.projections_context)
        self.FF_layers = torch.nn.ModuleList(self.FF_layers)

    def set_from_model_context_params(self, model):
        # We use this function to set the model's parameters when change-point is detected from best accuracy model

        self.confidence_context = model.confidence_context
        self.active_context = model.active_context
        self.context_error = model.context_error
        self.angles = model.angles

    def init_others(self):
        self.FF_layers = []
        self.projections_context = []
        self.availabilities = []
        self.batch_norms = []
        self.angles = []

        ## COntext detection params
        self.context_error_0 = torch.nn.Parameter(torch.zeros((1)), requires_grad=False)
        self.context_error = [self.context_error_0]
        self.confidence_context = [0]
        self.active_context = 0

        self.gatings_list = torch.zeros(self.hidden, 1).to(device).detach()
        self.availabilities_list = torch.ones(self.hidden, 1).to(device).detach()
        self.activities_list = torch.zeros(self.hidden, 1).to(device).detach()

    def ReTanh(self, x):
        return torch.tanh(x * (x > 0)) ** 2

    def print_availabilities(self):
        if "age" in self.lambda_:
            if self.conv:
                for layer in range(self.n_conv):
                    ages = 1.0 * (self.tmp_conv_availabilities[layer] < 0.05)
                    print(
                        "Number of available neuron conv layer {0}: ".format(layer),
                        1 - torch.mean(ages),
                    )
            for layer in range(self.n_layers - 1):
                ages = 1.0 * (self.tmp_availabilities[layer] < 0.05)
                print(
                    "Number of available neuron layer {0}: ".format(layer),
                    1 - torch.mean(ages),
                )

    def update_learning_rate(self):
        if "age" in self.lambda_:
            if self.conv:
                for layer in range(self.n_conv):
                    pre = (
                        torch.ones_like(self.FF_convs[layer].weight[0])
                        .to(device)
                        .detach()
                    )
                    A = self.conv_availabilities[layer].detach()
                    self.FF_convs[layer].bias.grad *= A.detach()

                    lr = (torch.einsum("i,jkl->ijkl", A, pre)).detach()
                    self.FF_convs[layer].weight.grad *= lr
            for layer in range(self.n_layers - 1):
                pre = (
                    torch.ones(self.FF_layers[layer].weight.shape[1])
                    .to(device)
                    .detach()
                )
                A = self.availabilities[layer].detach()
                self.FF_layers[layer].bias.grad *= A.detach()
                lr = (torch.einsum("i,j->ij", A, pre)).detach()
                self.FF_layers[layer].weight.grad *= lr

    def conv_forward(self, input_, context, training):
        current_context = self.get_network_context(context, training, input_.shape[0])

        for layer in range(self.n_conv):
            context = self.ReTanh(self.conv_context_projections[layer](current_context))
            input_ = torch.nn.ReLU()(self.FF_convs[layer](input_))
            context_conv = self.context_to_conv_context_projections(
                context, input_.shape[-2:]
            )
            input_ = input_ * context_conv
            self.update_ages_conv(layer, input_, training)

        input_ = self.pool(input_)
        input_ = torch.flatten(input_, start_dim=1, end_dim=-1)
        return input_

    def layer_forward(self, input_, layer, context, training):
        output = 0
        current_context = self.get_network_context(context, training, input_.shape[0])
        context = self.ReTanh(self.projections_context[layer](current_context))
        input_ = self.FF_layers[layer](input_)

        if not ("no context" in self.lambda_ or "age only" in self.lambda_):
            input_ = input_ * context
        if "sparse" in self.lambda_:

            input_ = torch.nn.ReLU()(self.batch_norms[layer](input_))
        input_ = F.normalize(input_, dim=1)
        if layer == self.n_layers - 2:
            x = self.FF_layers[layer + 1](input_)
            output = x.softmax(dim=1)
        return input_, output, context

    def deviate_from_mean(self, modulation, context):
        N = 50
        bar, std = torch.mean(
            self.context_error[self.active_context][-N:-1]
        ), torch.std(self.context_error[self.active_context][-N:-1])
        return modulation / bar > 1.01

    def fixing_ages(self):
        self.availabilities = copy.deepcopy(self.tmp_availabilities)
        if self.conv:
            self.conv_availabilities = copy.deepcopy(self.tmp_conv_availabilities)

    def update_ages(self, layer, loss, type="full"):
        with torch.no_grad():
            I_pre = self.inputs[layer]
            if layer == self.n_layers - 2:
                lr = 0.01
            else:
                lr = 0.01
            if type == "local":
                pre_shape = I_pre.shape[1]
                if layer == self.n_layers - 2:
                    I_post_big = (
                        self.FF_layers[layer + 1](I_pre)
                        .unsqueeze(2)
                        .repeat(1, 1, pre_shape)
                    )
                    w = (
                        self.FF_layers[layer + 1]
                        .weight.unsqueeze(0)
                        .repeat(I_pre.shape[0], 1, 1)
                    )
                    Iik = w * I_pre.unsqueeze(1).repeat(1, w.shape[1], 1)
                    Z = torch.exp(I_post_big)
                    Z_sum = torch.sum(Z, dim=1)
                    Z_sum_big = Z_sum.unsqueeze(1).repeat(1, Z.shape[1], 1)
                    a = Z_sum_big * Iik
                    b = torch.sum(Z * Iik, dim=1).unsqueeze(1).repeat(1, Z.shape[1], 1)
                    c = (
                        Z * (a - b) / torch.pow(Z_sum_big, 2)
                    )  ## gradient softmax for all
                    c_abs = torch.pow(c, 2)
                    post = torch.mean(c_abs, dim=(0, 1))

                else:
                    I_post = self.FF_layers[layer + 1](I_pre)
                    imp = torch.pow((I_post > 0) * self.context_gatings[layer + 1], 2)

                    batch_prod = (
                        torch.einsum("bi,bk->ik", imp, torch.pow(I_pre, 2)) / 2000
                    )
                    post = (
                        torch.pow(self.FF_layers[layer + 1].weight, 2) * batch_prod
                    )  ## single input importance
                    post = torch.mean(post, dim=0)

                post /= torch.mean(post)
                Delta = lr * self.tmp_availabilities[layer] * (post - self.epsilon)
            elif type == "activity":
                post = torch.mean(I_pre, dim=0).detach()
                post /= torch.mean(post)
                Delta = (
                    lr
                    * (
                        (2 + 1 * (layer == (self.n_layers - 2)))
                        * self.tmp_availabilities[layer]
                        * (post - self.epsilon)
                    ).detach()
                )
            elif type == "full":
                grad = torch.autograd.grad(loss, I_pre, retain_graph=True)[0].detach()

                post = torch.pow(grad * I_pre, 2)
                post = torch.mean(post, dim=0)
                post /= torch.mean(post)
                Delta = lr * self.tmp_availabilities[layer] * (post - self.epsilon)
            self.tmp_availabilities[layer] -= Delta.detach()
            self.tmp_availabilities[layer][self.tmp_availabilities[layer] > 1] = 1
            self.tmp_availabilities[layer][self.tmp_availabilities[layer] < 0] = 0

    def update_ages_conv(self, layer, input_, training):
        post = torch.mean(input_, dim=(0, 2, 3))
        post /= torch.mean(post)
        Delta = 0.004 * self.tmp_conv_availabilities[layer] * (post - self.epsilon)
        self.tmp_conv_availabilities[layer] -= Delta.detach()
        self.tmp_conv_availabilities[layer][self.tmp_conv_availabilities[layer] > 1] = 1
        self.tmp_conv_availabilities[layer][self.tmp_conv_availabilities[layer] < 0] = 0

    def context_to_conv_context_projections(self, context, shape):
        context = context.unsqueeze(2).unsqueeze(2)
        return context.repeat(1, 1, shape[0], shape[1])

    def get_network_context(self, context, training, shape):
        if training or (type(context) is int):
            current_context = (
                torch.nn.functional.one_hot(
                    torch.Tensor([context]).to(torch.int64), self.n_contexts
                )
                .float()
                .to(device)
            )
            current_context = current_context.repeat(shape, 1).detach()
        else:
            current_context = context[:, :, 0].detach()
        return current_context

    def forward_pass(self, input_data, context_input, training=False, target=None):
        modulation = None
        self.inputs = []
        self.context_gatings = []

        if training:
            if not self.context_fed:
                context_input = self.active_context
                self.context_error[self.active_context] = torch.nn.Parameter(
                    torch.cat(
                        [self.context_error[self.active_context], self.context_error_0],
                        dim=0,
                    ),
                    requires_grad=False,
                )
            else:
                self.active_context = context_input
        if self.conv:
            input_ = self.conv_forward(input_data, context_input, training)

        else:
            input_ = input_data
        for layer in range(len(self.FF_layers) - 1):

            input_, output, context_mag = self.layer_forward(
                input_, layer, context_input, training
            )
            self.inputs += [input_]
            self.context_gatings += [context_mag]
            if layer == (self.n_layers - 2) and training:
                self.inputs += [output]

        if target is not None:
            modulation = torch.nn.CrossEntropyLoss()(output, target)

        with torch.no_grad():
            if training and "age" in self.lambda_ and not self.context_fed:

                if (
                    self.deviate_from_mean(modulation, self.active_context)
                    and self.confidence_context[self.active_context] > 0.9
                ):
                    print("change of context", self.active_context)

                    reactivation = False
                    self.new_context = True
                    for context in range(len(self.context_error)):
                        _, m = self.forward_pass(
                            input_data, context, training=False, target=target
                        )
                        print("testing task ", context, "error ")
                        if not self.deviate_from_mean(m, context):
                            reactivation = True
                            break
                    if reactivation:
                        print("reactivating context {0}".format(context))
                        self.active_context = context
                    else:
                        print("no reactivation")
                        self.confidence_context += [0]
                        self.active_context = len(self.context_error)
                        self.context_error += [self.context_error_0]

                    print(
                        "number of tasks",
                        len(self.angles),
                        "contexts",
                        len(self.context_error),
                    )
                    print("angles so far", self.angles)
                else:
                    self.confidence_context[self.active_context] += (
                        1 - self.confidence_context[self.active_context]
                    ) * 0.005

                self.context_error[self.active_context][-1] = modulation.detach()

        if training and not self.new_context:
            for layer in range(len(self.FF_layers) - 1):
                self.update_ages(layer, modulation, type=self.causality)
        return output, modulation

    def forward(self, buffer, ground_truth=None, context_input=0, training=False):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        output, modulation = self.forward_pass(
            buffer, context_input, training, target=ground_truth
        )
        if training:
            error = modulation

            return error, output, modulation
        else:
            return output.detach()
