# +
import os
import sys
path = '/'.join(os.path.abspath(__file__).split('/')[:-1])
sys.path.insert(1, '{0}/pkg/'.format(path))
import torch
import general_utils
import torchvision
import tqdm
import matplotlib.pyplot as plt
import torch_cka
import copy
import numpy as np
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
original_transform = None
def save_data(dic,tmp_dic):
    keys = ['epoch_error','test_error','continual_test_error','test_accuracy',
            'continual_test_accuracy','change_points','ages_synapses','ages_neurons','ages_filters','context']
    for key in keys:
        if key in dic.keys():
            dic[key] += tmp_dic[key]
        else:
            dic[key] = tmp_dic[key]
    return dic
class simulator():
    def __init__(self,params, model, optimizer):
        self.epochs = params['epoch']
        self.fake_context = params['fake_context'] ## deprecated
        self.skip_context = params['skip_context'] ##  array of context you want to skip if you want to skip context number n 
        self.dic = params['dic'] ##  if you want to save data on top of old simulator
        if self.dic is None:
            self.dic = {}
        self.incongruent_label = params['incongruent_label'] ##  whehther we swutch labels
        self.incongruent_input = params['incongruent_input'] ##  whehther transform input
        self.all_steps = params['all_steps'] ## if we save at all time steps
        self.change_timing = params['change_timing'] ## steps before  environment change
        self.model = model
        self.optimizer = optimizer
        self.random_angle = params['random_angle']
        self.simulation_init()
        print(path)
        self.modelpath = path +'/../results/' + params['folder']
        if not os.path.exists(self.modelpath+'/models/'):
            
            os.makedirs(self.modelpath+'/models/')
    def init_ages(self):
        self.tmp_dic['ages_synapses'] = {}
        self.tmp_dic['ages_neurons'] = {}
        self.tmp_dic['ages_filters'] = {}
        if self.all_steps:
            steps = 1
        else:
            steps = self.change_timing
        for layer in range(self.model.n_layers - 1):
            if layer == 0:
    #             ages_synapses[layer]= torch.zeros(model.hidden * model.input_size,int(epochs/change_timing)).detach().cpu()
                 self.tmp_dic['ages_neurons'][layer] = torch.zeros(self.model.hidden,int(self.epochs/steps)).detach().cpu()
            else:
    #             ages_synapses[layer]= torch.zeros(model.hidden * model.hidden,int(epochs/change_timing)).detach().cpu()
                 self.tmp_dic['ages_neurons'][layer] = torch.zeros(self.model.hidden,int(self.epochs/steps)).detach().cpu()

    def init_tmp_dic(self):
        tmp_dic = {}
        tmp_dic['active_context'] = [0]
        tmp_dic['epoch_error'] = []
        tmp_dic['test_error'] = []
        tmp_dic['test_accuracy'] = []
        tmp_dic['continual_test_error'] = []
        tmp_dic['reconstruction_errors'] = []
        tmp_dic['continual_test_accuracy'] = {}
        tmp_dic['continual_test_accuracy'][0] = []
        tmp_dic['continual_test_error'] = {}
        tmp_dic['continual_test_error'][0] = []
        tmp_dic['backprop_errors'] = []
        tmp_dic['change_points'] = [0]
        tmp_dic['continual_images'] = []
        tmp_dic['continual_labels'] = []
        tmp_dic['continual_contexts'] = []
        tmp_dic['magnitude']  = []
        tmp_dic['context']  = []
        self.tmp_dic = tmp_dic
    def new_context(self):
        
        self.converged = False
        if self.fake_context is not  None:
            self.context = self.fake_context[self.context_n]
        else:
            self.context = self.context_n
        if self.incongruent_input:
            ## either permute pixels or rotate 
            if not self.random_angle:            
                self.transform = general_utils.get_transform(random = True)
            else:
                self.angle = torch.randint(360,[1])[0].item()
        self.test_images, self.test_labels =  general_utils.load_data(False, self.transform, 1000, self.model.n_contexts, self.angle)
        self.images_all, self.labels_all =  general_utils.load_data(True, self.transform, 60000, self.model.n_contexts, self.angle)
    #   
        self.remapping = None
        if self.incongruent_label:
            self.remapping = torch.arange(0, self.model.output_size).to(device), torch.randperm(self.model.output_size).to(device)
            self.test_labels = general_utils.remap_values(self.remapping, self.test_labels)
        if self.context_n not in self.tmp_dic['continual_test_accuracy'].keys() and self.context_n < 10 and (self.context_observed % 1) == 0:
            self.tmp_dic['continual_test_accuracy'][self.context_n] = []
            self.tmp_dic['continual_test_error'][self.context_n] = []
            self.tmp_dic['continual_images'] += [self.test_images.detach().clone()]
            self.tmp_dic['continual_labels'] += [self.test_labels.detach().clone()]
        self.tmp_dic['continual_contexts'] += [self.test_context]
        plt.imshow(self.images_all[1].reshape(28, 28).detach().cpu())
        plt.show()
        self.context_observed += 1
        # -
    def update_contexts(self):
        test_context = torch.zeros(1000, self.model.n_contexts, self.model.n_layers - 1).to(device)
        for layer in range(self.model.n_layers - 1):
            test_context[:,int(self.model.layer_context[layer * self.model.context_per_layer]),layer] = 1
        self.active_contexts[:, self.context_n] = self.model.layer_context
        self.tmp_dic['continual_contexts'][-1] = test_context
        self.test_context = test_context

    def simulation_init(self):
        self.converged = False
        self.angle = 0
        self.context_n = 0
        self.context = 0
        self.transform = general_utils.get_transform(random = False)
        self.original_transform = general_utils.get_transform(random = False)
        self.init_tmp_dic()
        self.test_images, self.test_labels =  general_utils.load_data(False, self.transform, 1000, self.model.n_contexts, self.angle)
        self.remapping = torch.arange(0, self.model.output_size).to(device), torch.arange(0, self.model.output_size).to(device)
        self.test_labels = general_utils.remap_values(self.remapping, self.test_labels)
        self.tmp_dic['continual_images'] += [self.test_images.detach().clone()]
        self.tmp_dic['continual_labels'] += [self.test_labels.detach().clone()]
        self.tmp_dic['continual_contexts'] = [None]
        self.active_contexts = torch.zeros(self.model.n_layers-1,1)
        self.images_all, self.labels_all =  general_utils.load_data(True, self.transform, 60000,self.model.n_contexts, self.angle)
        self.init_ages()
    def run_test_errors(self, epoch):
        labels_hot = torch.nn.functional.one_hot(self.test_labels.to(torch.int64), self.model.output_size).to(torch.float32)
        test_prediction =self.model.forward(self.test_images, context_input = self.test_context)
        test_error = torch.mean((test_prediction - labels_hot) ** 2).detach().cpu()
        if test_error <self.best_loss:
            self.best_loss = test_error
            self.best_model = copy.deepcopy(self.model)
            self.tmp_dic['test_error'] += [torch.mean((test_prediction - labels_hot) ** 2).detach().cpu()]
            self.tmp_dic['test_accuracy'] += [torch.mean(1. * (torch.argmax(test_prediction,dim = 1) == torch.argmax(labels_hot,dim = 1))).detach().cpu()]
            for task in range(len(self.tmp_dic['continual_images'])):
                continual_test_images = self.tmp_dic['continual_images'][task]
                labels_hot = torch.nn.functional.one_hot(self.tmp_dic['continual_labels'][task].to(torch.int64), self.model.output_size).to(torch.float32).detach()
                continual_testcontext = self.tmp_dic['continual_contexts'][task]
                continual_test_prediction = self.model.forward(continual_test_images,context_input =  continual_testcontext)
                ## Saving errors
                self.tmp_dic['continual_test_accuracy'][task] += [torch.mean(1.*(torch.argmax(continual_test_prediction,dim = 1) == torch.argmax(labels_hot,dim = 1))).detach().cpu()]
                self.tmp_dic['continual_test_error'][task] += [torch.mean((continual_test_prediction - labels_hot)**2).detach().cpu()]
        else:
            self.tmp_dic['test_error'] += [self.tmp_dic['test_error'][-1]]
            self.tmp_dic['test_accuracy'] += [self.tmp_dic['test_accuracy'][-1]]
            ## Saving errors
            for task in range(len(self.tmp_dic['continual_images'])):
                self.tmp_dic['continual_test_accuracy'][task] += [self.tmp_dic['continual_test_accuracy'][task][-1]]
                self.tmp_dic['continual_test_error'][task] += [self.tmp_dic['continual_test_error'][task][-1]]
        ## Make tests predictions
        
    def update_vizu_ages(self, step):
        for layer in range(self.model.n_layers-1):
            ages = self.model.availabilities[layer]
            self.tmp_dic['ages_neurons'][layer][:,step] = ages.flatten().clone().detach().cpu()

def run_simulation(model, optimizer, params):

    
    
    ## initialise simulation
    MNIST_simulator = simulator(params, model, optimizer)
        
    save_frequency = 1

    ## load first images ##
    index_all = torch.zeros(60000)
    batch_size = 2000
    index_all[:batch_size] = 1
    
    MNIST_simulator.best_loss=np.inf
    best_model=copy.deepcopy(model)
    MNIST_simulator.context_observed = 0
    for epoch in tqdm.tqdm(range(MNIST_simulator.epochs)):
        with torch.no_grad():
            MNIST_simulator.update_contexts()
            
            
            ## loading data
            
            if MNIST_simulator.all_steps and 'age' in MNIST_simulator.model.lambda_:
                MNIST_simulator.update_vizu_ages(epoch)
            
            ## We go to new context ! ##
            if ((epoch + 1) % MNIST_simulator.change_timing == 0):
                converged = False
                # for layer in range(len(MNIST_simulator.model.availabilities)):
                #     MNIST_simulator.model.availabilities[layer][MNIST_simulator.model.availabilities[layer] > .05] = 1
                MNIST_simulator.best_loss=np.inf
                MNIST_simulator.model = MNIST_simulator.best_model
                torch.save(model, MNIST_simulator.modelpath+'/models/network_{0}'.format(MNIST_simulator.model.lambda_))
                if not epoch == (MNIST_simulator.epochs - 1):
                    MNIST_simulator.context_n = MNIST_simulator.context_n + 1
                    MNIST_simulator.active_contexts = torch.cat([MNIST_simulator.active_contexts,torch.zeros(MNIST_simulator.model.n_layers-1,1)],dim = 1)
                    if MNIST_simulator.context_n in MNIST_simulator.skip_context:
                        continue
                    MNIST_simulator.new_context()
                    

                MNIST_simulator.model.print_availabilities()
                if not MNIST_simulator.all_steps  and 'age' in MNIST_simulator.model.lambda_:
                        MNIST_simulator.update_vizu_ages(int(epoch / MNIST_simulator.change_timing))
            if epoch % save_frequency == 0:
                
                    MNIST_simulator.run_test_errors(epoch % MNIST_simulator.change_timing)

            if not MNIST_simulator.converged and MNIST_simulator.tmp_dic['test_accuracy'][-1] > 0.9:
                print('time to 90: ', epoch % MNIST_simulator.change_timing)
                MNIST_simulator.converged = True
            if (epoch-1) % MNIST_simulator.change_timing == 0:
                MNIST_simulator.optimizer = torch.optim.Adam(MNIST_simulator.model.parameters(), lr = 2e-2)
    

            ## select train images randomly
            index_all = index_all[torch.randperm(len(index_all))] == 1
            images = MNIST_simulator.images_all[index_all]
            labels = MNIST_simulator.labels_all[index_all]
            if MNIST_simulator.incongruent_label:
                labels = general_utils.remap_values(MNIST_simulator.remapping, labels)
        loss, y_pred, modulation = MNIST_simulator.model(images, labels, training = True, context_input = MNIST_simulator.context_n)

        MNIST_simulator.optimizer.zero_grad()
        if epoch % save_frequency == 0:
            MNIST_simulator.tmp_dic['epoch_error'] += [loss.detach().cpu()]
            MNIST_simulator.tmp_dic['context'] += [MNIST_simulator.context]

        loss.backward()
        MNIST_simulator.model.update_learning_rate()
        MNIST_simulator.optimizer.step()
        
    MNIST_simulator.dic = save_data(MNIST_simulator.dic,MNIST_simulator.tmp_dic)
    return MNIST_simulator.dic
# -


