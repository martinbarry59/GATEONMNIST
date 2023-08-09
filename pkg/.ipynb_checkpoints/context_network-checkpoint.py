import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
from scipy.stats import energy_distance
import geomloss
class network(torch.nn.Module):
    def __init__(self,n_contexts, n_layers,output_size , hidden = 100,lambda_ = 0.1, input_size = None,channels = []):
        super().__init__()
        """
        In the constructor we instantiate four parameters and assign them as
        member parameters.
        """
        self.hidden = hidden
        self.n_layers = n_layers
        self.channels = channels
        self.n_contexts = n_contexts
        self.output_size = output_size
        self.lambda_ = lambda_
        self.effective_gating = []
        self.context_per_layer = False
        self.context_fed = True
        self.init_others()
        self.lambda_context = torch.zeros(n_contexts)     
        self.old_inputs = []
        self.context_meanput = []
        self.confidence_context = []
        
        if input_size == None:
            self.input_size = self.output_size
        else:
            self.input_size = input_size
        for layer in range(self.n_layers):
            if self.n_layers ==  1:
                FF_linears = torch.nn.Linear(self.input_size, self.output_size)
            elif layer == 0:
                i = self.input_size
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
                a = (6 * (self.n_contexts + self.hidden)) ** (1/2)
                # torch.nn.init.uniform_(proj.weight,- 1 * a, 0.1 * a )
                proj.bias.data.fill_(0.0)
                proj.bias.requires_grad = False
                self.projections_context += [proj]
            self.batch_norms +=[torch.nn.BatchNorm1d(o).to(device)]
            if layer == 0 :
                self.availabilities += [torch.zeros(o).to(device)]
            else:
                self.availabilities += [torch.ones(o).to(device)]
            self.FF_layers += [torch.nn.Linear(i, o).to(device)]
            self.error_layers += [torch.nn.Linear(o, o).to(device)]
            if self.context_per_layer:
                self.old_inputs += [torch.zeros(2000, o).to(device)]
                self.context_meanput += [[0]]
                self.confidence_context += [[0]]
            elif layer == self.n_layers - 2:
                self.old_inputs += [torch.zeros(2000, o).to(device)]
                self.context_meanput += [[0]]
                self.confidence_context += [[0]]

            
        self.projections_context = torch.nn.ModuleList(self.projections_context)
        self.FF_layers = torch.nn.ModuleList(self.FF_layers)
        
    def init_others(self):
        self.FF_layers = []
        self.error_layers = []
        self.projections_context = []
        self.age_synapses = []
        self.availabilities = []
        self.batch_norms = []
        self.batch_norms_conv = []
        self.projections_context_error = []
        self.projections_context_error1 = []
        self.active_context = 0
        self.detected_contexts = 1
        self.list_contexts = [0]
        self.context_confidence = torch.zeros(1)
        self.context_error = torch.zeros((self.n_layers - 1,1))
        if self.context_per_layer:## use context for each layer
            self.layer_context = torch.zeros((self.n_layers - 1))
        else: ## use one context for the network
            self.layer_context = torch.zeros((1))
        self.output_error = torch.zeros(1)
    def ReTanh(self,x):
        return torch.tanh( x * (x > 0)) ** 2
    def print_availabilities(self):
        if 'neuron' in self.lambda_:
            for layer in range(self.n_layers - 1):
                ages = 1. * (self.availabilities[layer] < 0.05)
                print('Number of available neuron layer {0}: '.format(layer),
                        1 - torch.mean(ages))
    def lateral_inhibition(self,activities):
        std = torch.std(activities, 1).unsqueeze(1).repeat(1, activities.shape[1])
        mean = torch.mean(activities, 1).unsqueeze(1).repeat(1, activities.shape[1])
        return torch.nn.ReLU()(activities - (mean + 2 * std))
    def update_learning_rate(self):
        if 'neuron age' in self.lambda_:                
            for layer in range(self.n_layers - 1):
                pre = torch.ones(self.FF_layers[layer].weight.shape[1]).to(device).detach()
                A = self.availabilities[layer].detach()
                self.FF_layers[layer].bias.grad *= A.detach()
                lr = (torch.einsum('i,j->ij', A, pre)).detach()
                self.FF_layers[layer].weight.grad *= lr
    def layer_forward(self, input_,layer,context, training):
        output = 0
        if training:
            current_context = torch.nn.functional.one_hot(torch.Tensor([context]).to(torch.int64), self.n_contexts).float().to(device)
            current_context = current_context.repeat(input_.shape[0], 1).detach()
        else:
            current_context = context[:,:,layer].detach()
        context_idx = current_context[0,:] > 0
        context = self.ReTanh(self.projections_context[layer](current_context))
        input_ = self.FF_layers[layer](input_)
        
       
        if not ('no context' in self.lambda_ or 'age only' in self.lambda_):
            input_ = input_ * context
        if 'sparse' in self.lambda_:
            
            input_ =  torch.nn.ReLU()(self.batch_norms[layer](input_))
            #input_ = self.lateral_inhibition(input_)
        input_ = F.normalize(input_, dim = 1)
        if layer == self.n_layers - 2:
            x = self.FF_layers[layer+1](input_)
            output = x.softmax(dim = 1)
        return input_, output
    def compute_layer_distance(self, layer, y, output, target, context_idx):
        x = self.old_inputs[layer * self.context_per_layer].detach()
        dist = 2 * geomloss.SamplesLoss('energy')(x,y).detach()
        deviation = dist / torch.mean(self.context_error[layer* self.context_per_layer][-11:-1])
        output_deviation = 0
        if layer == self.n_layers - 2:
            modulation = torch.nn.CrossEntropyLoss()(output,target)
        else:
            modulation = 0
        output_deviation = (modulation / torch.mean(self.output_error[-11:-1])).detach()
        return dist, deviation, output_deviation, modulation
    def forward_pass(self,input_, context_input,training = False, target = None):
        modulation = None
        escape_condition = 10
        batch_std = 0
        batch_mean = 0
        if training:
            self.context_error = torch.cat([self.context_error,torch.zeros(self.n_layers-1,1)],dim = 1)
            self.output_error = torch.cat([self.output_error,torch.zeros(1)],dim = 0)
        for layer in range(len(self.FF_layers) - 1):
            if training:
                if not self.context_fed:
                    context_layer = layer * self.context_per_layer
                    context_input = int(self.layer_context[context_layer])
                else:
                    self.layer_context[0] = context_input
            input_, output = self.layer_forward(input_, layer, context_input, training)
            if training and 'age' in self.lambda_ and not self.context_fed:
                if self.context_per_layer or (not self.context_per_layer) * layer == (len(self.FF_layers) - 2):
                    dist, deviation, output_deviation, modulation = self.compute_layer_distance(layer, input_, output, target, context_input)
                    if (deviation > escape_condition or output_deviation > 1.1) and self.confidence_context[context_layer][context_input] > 0.9:
                        print(layer,'change of context', deviation)
                        reactivation = False
                        
                        for context in range(len(self.context_meanput[context_layer])-1):
                            y, o = self.layer_forward(input_, layer, context, training)
                            dist_tmp, deviation_tmp, output_deviation_tmp,modulation_tmp = self.compute_layer_distance(layer, y, o, target, context)
                            if deviation < escape_condition:
                                reactivation = True
                                input_ = y
                                output = o
                                break
                        if reactivation:
                            print('reactivating context {0}'.format(context))
                            self.layer_context[context_layer] = context
                            modulation = modulation_tmp
                            dist = dist_tmp
                        else:
                            print('no reactivation context')
                            self.layer_context[context_layer] = len(self.context_meanput[context_layer])
                            self.confidence_context[context_layer] += [0]
                            self.context_meanput[context_layer] += [0]
                    else:
                        self.confidence_context[context_layer][context_input] += (1 - self.confidence_context[context_layer][context_input]) * 0.01
                
                    context_input = int(self.layer_context[context_layer])
                    self.context_meanput[context_layer][context_input] = self.old_inputs[context_layer].detach()
                    self.context_error[context_layer][-1] = dist.detach()
                    self.old_inputs[context_layer] = input_
                if layer == (self.n_layers - 2):
                    self.output_error[-1] = modulation.detach()
            batch_std += torch.mean(torch.std(input_,dim = 0))
            batch_mean += torch.mean(torch.mean(torch.abs(input_),dim = 1))
            if layer == (self.n_layers -2) and training:
                modulation = torch.nn.CrossEntropyLoss()(output,target) + 0 * batch_std + 10 * batch_mean
            if training and 'age' in self.lambda_:
                post = torch.mean(input_,dim =0).detach()
                Delta = (1 * self.availabilities[layer] * (post)).detach()
                self.availabilities[layer] -= Delta.detach()
                self.availabilities[layer][self.availabilities[layer] > 1] = 1
             
        return output, modulation
    def forward(self, buffer,ground_truth = None, context_input = 0, training = False):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        output, modulation = self.forward_pass(buffer, context_input, training, target = ground_truth)
        if training:
            error = modulation
            
            return error, output, modulation
        else:
            return output.detach()


