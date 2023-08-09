import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
import sys
path = "/".join(os.path.dirname(os.path.realpath(__file__)).split("/")[:-1])
sys.path.insert(1, "{0}/pkg/".format(path))
import matplotlib.pyplot as plt
import torch
import context_network
import general_utils
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


run = 'MNIST' #'MAZE' #
samples = []
torch.autograd.set_detect_anomaly(True)
import cProfile, pstats
#importlib.reload(context_network)
dic_samples = {}
n_layers  = 5
channels  = [1]#[1,30,30]
hidden = 5000
n_average = 1
# simulation
models = []
if run == 'MNIST':
    import MNIST_network_utils as n_utils
  #  importlib.reload(n_utils)
    lambdas = ['neuron age sparse','no context sparse', 'context only sparse','age only sparse']
    params = {}
    params['epoch'] = 15000
    params['fake_context'] = None ## deprecated
    params['skip_context'] = [] ##  array of context you want to skip if you want to skip context number n 
    params['dic'] = None ##  if you want to save data on top of old simulator
    params['incongruent_label'] = True##  whehther we swutch labels
    params['incongruent_input'] = False ##  whehther transform input
    params['random_angle'] = False ##  whehther transform input
    params['all_steps'] = False ## if we save at all time steps
    params['change_timing'] = 150 ## steps before  environment change
    labels = 10
    text_incongruency = ('incongruent_label' * params['incongruent_label'] 
                         + 'incongruent_input' * params['incongruent_input'])
    angle_line = '' + '_angles'*params['random_angle']
    folder = 'n_layers_{0}_hidden_{1}_epochs_{2}_{3}{4}'.format(n_layers, hidden, params['epoch'],text_incongruency,angle_line)
    params['folder'] = folder
    
    
    n_contexts = 110
    load_models = 0
n_samples = len(lambdas)
for i in range(n_samples):
    dic_samples[lambdas[i]] = []
    for episode in range(n_average):
        if 'no context' in lambdas[i]:
            print('Not using context')
            use_context = 0
        else:
            print('Using context')
            use_context = 1
        str_ = '-----------------episode number {0} for lambda = {1} -----------------'.format(episode + 1,lambdas[i])
        str_ = str_.center(80, '-')
        if run == 'MNIST':
            if load_models == 0:
                model = context_network.network(n_contexts, n_layers, labels, hidden, channels = channels,
                                lambda_ = lambdas[i], input_size = 784,
                               ).to(device)

            if load_models != 0:
                
                model = torch.load(path+'/results/models/network_{0}'.format(load_models))
                model.to(device)
                
                print('reuse')

            optimizer = torch.optim.Adam(model.parameters(), lr = 2e-2)
            skip_context = []
            info_dic= n_utils.run_simulation(model, optimizer,params)
            plt.plot(model.context_error[:-1,100:].T)
            plt.show()
            for layer in range(model.n_layers - 1):
                print('--Layer N {0}--'.format(layer))
                if 'age' in lambdas[i]:
                    plt.plot(info_dic['ages_neurons'][layer].T)
                    plt.show()
            
            general_utils.plot_end_sim(model, info_dic,n_layers = n_layers, context = 1)
            
        elif run == 'MAZE':
            model = network(n_contexts, n_layers, number_rooms, hidden, lambda_ = lambdas[i]).to(device)
            if use_context:
                optimizer = torch.optim.SGD(model.parameters(), lr = 1e-5)
                
            else:
                optimizer = torch.optim.SGD(model.parameters(), lr = 1e-5)

            X,True_transition = n_utils.full_inputs(n_contexts,n_mazes,number_rooms,simulations[0],use_context=use_context)
            
            info_dic = n_utils.run_simulation(model,optimizer, simulations[0], X.to(device), True_transition.to(device),
                                         use_context = use_context ,skip_context = skip_context, 
                                        fake_context = None)
            
        dic_samples[lambdas[i]] += [info_dic]
        if run == 'MNIST':
            keys = ['continual_test_error','test_error','epoch_error','test_accuracy','continual_test_accuracy']
        elif run == 'MAZE':
            keys = ['error','test_error']  
        skip = []
        tmp_file = "results.pkl".format(hidden,params['epoch'])
        general_utils.plot_average(dic_samples,keys, skip,tmp_file,folder)



else:
    print('Not existing...')
n_samples = len(lambdas)
for i in range(n_samples):
    dic_samples[lambdas[i]] = []
    for episode in range(n_average):
        if 'no context' in lambdas[i]:
            print('Not using context')
            use_context = 0
        else:
            print('Using context')
            use_context = 1
        str_ = '-----------------episode number {0} for lambda = {1} -----------------'.format(episode + 1,lambdas[i])
        str_ = str_.center(80, '-')
        print(str_)
        if run == 'MNIST':
            model = network(n_contexts, n_layers, labels, hidden, channels = channels,
                            lambda_ = lambdas[i], input_size = 784,
                           ).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr = 1e-2)

            skip_context = []
            info_dic= n_utils.run_simulation(model,optimizer, epochs,
                                         use_context = use_context ,skip_context = skip_context, 
                                         fake_context = None,replay = 0,incongruent_label = incongruent_label,
                                         incongruent_input = incongruent_input, change_timing = 250,all_steps = False)
            for layer in range(model.n_layers):
                print('--Layer N {0}--'.format(layer))
                plt.plot(info_dic['ages_neurons'][layer].T)
                plt.show()
            for layer in range(model.conv_layers):
                print('--conv Layer N {0}--'.format(layer))
                plt.plot(info_dic['ages_filters'][layer].T)
                plt.show()
            
            general_utils.plot_end_sim(model, info_dic,n_layers = n_layers, context = 1)
        elif run == 'MAZE':
            model = network(n_contexts, n_layers, number_rooms, hidden, lambda_ = lambdas[i]).to(device)
            if use_context:
                optimizer = torch.optim.SGD(model.parameters(), lr = 1e-5)
                
            else:
                optimizer = torch.optim.SGD(model.parameters(), lr = 1e-5)

            X,True_transition = n_utils.full_inputs(n_contexts,n_mazes,number_rooms,simulations[0],use_context=use_context)
            
            info_dic = n_utils.run_simulation(model,optimizer, simulations[0], X.to(device), True_transition.to(device),
                                         use_context = use_context ,skip_context = skip_context, 
                                        fake_context = None)

        dic_samples[lambdas[i]] += [info_dic]
        if run == 'MNIST':
            keys = ['continual_test_error','test_error','epoch_error','test_accuracy','continual_test_accuracy']
        elif run == 'MAZE':
            keys = ['error','test_error']
        skip = []
        tmp_file = "results.pkl".format(hidden,epochs)
        folder = 'n_layers_{0}_hidden_{1}_conv_layers_{4}_epochs_{2}_{3}'.format(n_layers, hidden, epochs,text_incongruency,model.conv_layers)
        general_utils.plot_average(dic_samples,keys, skip,tmp_file,folder)
        del model

