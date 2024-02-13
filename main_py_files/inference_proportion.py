## this file is used for inference for proportional model

import sys
import os
import numpy as np
import argparse
import pandas as pd
import torch
from torch import nn
from configparser import ConfigParser
curr_pth = os.path.abspath('../')
sys.path.append(curr_pth)
from proportion_model.src.run_workflow import build_config_map
from utils.utilities import create_directory,ReaderWriter,build_performance_dfs
from proportion_model.src.run_workflow import test_run


cmd_opt = argparse.ArgumentParser(description='Argparser for data')
cmd_opt.add_argument('-transfer_learning',type=str,default ='',help = ' transfer learning')
cmd_opt.add_argument('-data_invivo',type=str,default ='',help = 'experimental type')
cmd_opt.add_argument('-model_invivo',type=str,default = '',help = 'model type')
cmd_opt.add_argument('-data_cell_name',type=str,default = '',help = 'data cell name')
cmd_opt.add_argument('-model_cell_name',type=str,default = '',help = 'model cell name')
cmd_opt.add_argument('-cell_name',type=str,default ='',help = 'name of the cell')
cmd_opt.add_argument('-data_name',  type=str,default = '', help = 'directory of the data')
cmd_opt.add_argument('-saved_model',  type=str,default = '', help = 'name of the model used for inference')
cmd_opt.add_argument('-config_file', type=str, default='config_file.ini', help='')
args, _ = cmd_opt.parse_known_args()


def proportion_inference(args):


    config = ConfigParser()
    config.read(args.config_file)

    if 'Inference_proportiona_model' in config:
        print('we are here, inference param')
        params = config['Inference_proportiona_model']
        
        args.data_invivo = config.getboolean('Inference_proportiona_model', 'data_invivo')
        args.model_invivo = config.getboolean('Inference_proportiona_model', 'model_invivo')
        args.transfer_learning = config.getboolean('Inference_proportiona_model', 'transfer_learning')
        args.model_cell_name = params['model_cell_name']
        args.model_cell_name = params['model_cell_name']
        args.data_cell_name = params['data_cell_name']
        args.data_name = params['data_name']
        args.saved_model = params['saved_model']

        print('we are printing out args', args)

        transfer_learning = args.transfer_learning
        print('is it transfer learning:', transfer_learning)

        # Rest of the code...

    else:
        raise ValueError("The 'InferenceConfig' section is missing in the configuration file.")

    
    #args.data_invivo = False
    #args.model_invivo = False
   ## define the parameters
    k = 5
    embed_dim = 128
    num_attn_heads = 8
    num_trf_units = 1
    pdropout = 0.25
    activ_func = nn.ReLU()
    multp_factor = 2
    multihead_type = 'Narrow'
    weight_decay = 1e-4
    batch_size = 400
    num_epochs = 100
    optim_tup = None
    experiment_desc = ''
    # possible losses are {'klloss', 'CEloss', 'MSEloss'}
    loss_func = 'klloss'
    trf_tup = (embed_dim, 
               num_attn_heads, 
               num_trf_units, 
               pdropout, 
               activ_func, 
               multp_factor, 
               multihead_type,
               weight_decay, 
               batch_size,
               num_epochs)

    opt_adendum={'inp_seqlen':20, 'outp_seqlen':20, 'weight_haplotypes':True, 'mask_nontarget_bases':True}

    mconfig, options = build_config_map(experiment_desc, 'HaplotypeEncoderEncoder', optim_tup, trf_tup,
                                        opt_adendum=opt_adendum,loss_func=loss_func)

    data_fold = '../dataset'
    working_dir = '../proportion_model'

    ### define if it is transfer learning inference or not;
    ### it is transfer learning:
    ### 1) when both are in vitro, if the data and model does not match, 
    ### 2) when one is in vivo and one is in vitro
    '''
    if args.model_invivo == args.data_invivo:
        if args.data_name == args.saved_model:
            transfer_learning = False
        else:
            transfer_learning = True
    else: 
        transfer_learning = True
    '''
    transfer_learning = args.transfer_learning
    print('is it transfer learning:', transfer_learning)
    
    if args.data_invivo:
        print('data_invivo', args.data_invivo)
        absolute_dir=  '../dataset/invivo/'+ args.data_cell_name
    else:
        print('we are running this')
        absolute_dir='../dataset'

    input_type = 'protospacer_PAM'  

    data_dir  = absolute_dir + '/' + input_type + '/'+ f'{args.data_name}_proportions_encenc_two_model'

    print('reading dataset from:', data_dir)
    dpartitions = ReaderWriter.read_data(os.path.join(data_dir, f'data_partitions.pkl'))
    num_runs = len(dpartitions)
    datatensor_partitions =  ReaderWriter.read_data(os.path.join(data_dir, f'dtensor_partitions.torch'))

    gpu_indices = list(range(len(dpartitions)))
    run_gpu_map = {i:0 for i, indx in enumerate(gpu_indices)}
    if input_type == 'protospacer':
        options['inp_seqlen'] = 20
        options['outp_seqlen'] = 20

    elif input_type == 'protospacer_PAM':
        options['inp_seqlen'] = 24
        options['outp_seqlen'] = 24

    elif input_type == 'protospacer_PAM_overhangs':
        options['inp_seqlen'] = 24 + 2*k
        options['outp_seqlen'] = 24 + 2*k

    else:
        print('specify the input type')


    ## get the model
    if not args.model_invivo:
        model_path = os.path.join(working_dir, 'output', 
                                          'experiment_run_proportions_encenc_two_model',f'{args.saved_model}_proportions_encenc_two_model', 
                                          input_type,'exp_version_0')
    else:
        model_path = os.path.join(working_dir, 'output', 
                                          'experiment_run_proportions_encenc_two_model',args.model_cell_name, 
                                  f'{args.saved_model}_proportions_encenc_two_model',
                                         input_type, 'exp_version_0')
    print("\n"*3)    
    print("*"*25)
    print('loading model from:', model_path)

    ## define the output path
    if not transfer_learning:
        test_path = os.path.join(model_path)

    ## for transfer learning inference
    else:
        
        print('we are running transfer learning')
        if args.data_invivo and not args.model_invivo:
            print('we are running vitro model and vivo data')
            test_path = os.path.join(working_dir, 
                          'output', 
                          'experiment_run_proportions_encenc_two_model', 
                                 'transfer_learning', 'vitro_model_vivo_data', args.data_cell_name, args.data_name, 
                          input_type)
            
        elif not args.data_invivo and args.model_invivo:
            print('we are running vivo model and vitro data')

            test_path = os.path.join(working_dir, 
                          'output', 
                          'experiment_run_proportions_encenc_two_model', 
                                 'transfer_learning', 'vivo_model_vitro_data', args.model_cell_name, args.data_name, 
                          input_type)
        elif args.data_invivo and args.model_invivo:
            
            if args.data_name != args.saved_model:
                print('we are running transfer learning within vivo/vitro')
                test_path = os.path.join(working_dir, 
                          'output', 
                          'experiment_run_proportions_encenc_two_model', 
                                 'transfer_learning', f'model_{args.saved_model}_data_{args.data_name}', args.cell_name, args.data_name, 
                          input_type)
           
            if args.data_cell_name != args.model_cell_name:
                
                print('we are running transfer learning within vivo')
                test_path = os.path.join(working_dir, 
                          'output', 
                          'experiment_run_proportions_encenc_two_model', 
                                 'transfer_learning', f'model_{args.model_cell_name}_data_{args.data_cell_name}', args.data_name, 
                          input_type)
            
                
        else:
            raise ValueError("the conditions does not match for transfer learning")
    
    print("\n"*3)
    print("*"*25)
    print('the results are save at:', test_path)
    config_map = mconfig, options

    ## run inference
    
    print('model path', model_path)
    test_run(datatensor_partitions, config_map, model_path, test_path, run_gpu_map, num_epochs=1)

    ##  peformance report                                                                             # get the performance results on training data
    num_runs = len(datatensor_partitions)
   
    train_performance = build_performance_dfs(test_path, num_runs, 'train', 'continuous')
    train_performance.to_csv(test_path+'/train_peformance.csv')                                                                          # get the performance results on validation dat
    build_performance_dfs(test_path, num_runs, 'validation', 'continuous')                                                                              
    test_peformance = build_performance_dfs(test_path, num_runs, 'test', 'continuous')
    test_peformance.to_csv(test_path+'/test_peformance.csv')  
    print("\n"*3)
    print("*"*25)
    print(test_peformance)



if __name__ == "__main__":
    args, _ = cmd_opt.parse_known_args()
    proportion_inference(args)
   
    