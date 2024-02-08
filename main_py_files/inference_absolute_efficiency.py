## this file is used to inference for the test data for absolute efficiency model

import os
import numpy as np
import pandas as pd
import torch
import argparse
from configparser import ConfigParser
working_path = '../absolute_efficiency_model'
import sys
sys.path.append(working_path)
print(working_path)
from models.data_process import get_data_ready,get_datatensor_partitions
from src.utils import create_directory, one_hot_encode, get_device, ReaderWriter, print_eval_results
from models.trainval_workflow import run_inference
from src.utils import compute_eval_results_df


cmd_opt = argparse.ArgumentParser(description='Argparser for data')
cmd_opt.add_argument('-transfer_learning',type=str,default ='',help = ' transfer learning')
cmd_opt.add_argument('-saved_model',  type=str,default = '', help = 'name of the model used for inference')
cmd_opt.add_argument('-data_invivo',type=str,default = '',help = 'experimental type')
cmd_opt.add_argument('-model_invivo',type=str,default = '',help = 'model type')
cmd_opt.add_argument('-data_cell_name',type=str,default = '',help = 'data cell name')
cmd_opt.add_argument('-model_cell_name',type=str,default = '',help = 'model cell name')
cmd_opt.add_argument('-cell_name',type=str,default ='',help = 'name of the cell')
cmd_opt.add_argument('-model_name',  type=str,default = 'CNN', help = 'name of the model')
cmd_opt.add_argument('-exp_name',  type=str, help = 'name of the experiment')
cmd_opt.add_argument('-data_dir',  type=str,default = './data/', help = 'directory of the data')
cmd_opt.add_argument('-data_name',  type=str,default = '', help = 'directory of the data')
#cmd_opt.add_argument('-target_dir',  type=str, default='processed',  help = 'folder name to save the processed data')
cmd_opt.add_argument('-working_dir',  type=str, default='./', help = 'the main working directory')
#cmd_opt.add_argument('-output_path', type=str, help='path to save the trained model')
#cmd_opt.add_argument('-model_path', type=str, help='path to trained model')
cmd_opt.add_argument('-random_seed', type=int,default=42)
cmd_opt.add_argument('-config_file', type=str, default='config_file.ini', help='')
#cmd_opt.add_argument('-config_file', type=str, default='config.ini', help='Path to the configuration file')
#cmd_opt.add_argument('-epoch_num', type=int, default =200, help='number of training epochs')
args, _ = cmd_opt.parse_known_args()
#data_invivo = args.data_invivo
#model_invivo= args.model_invivo




def main_inference(args):
    config = ConfigParser()
    config.read(args.config_file)

    if 'Inference_absolute_efficiency' in config:
        params = config['Inference_absolute_efficiency']
        
        args.data_invivo = config.getboolean('Inference_absolute_efficiency', 'data_invivo')
        args.model_invivo = config.getboolean('Inference_absolute_efficiency', 'model_invivo')
        args.transfer_learning = config.getboolean('Inference_absolute_efficiency', 'transfer_learning')
        args.saved_model = params['saved_model']
        args.model_cell_name = params['model_cell_name']
        args.data_cell_name = params['data_cell_name']
        args.data_name = params['data_name']

        print('we are printing out args', args)

        transfer_learning = args.transfer_learning
        print('is it transfer learning:', transfer_learning)

        # Rest of the code...

    else:
        raise ValueError("The 'InferenceConfig' section is missing in the configuration file.")

    
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
        absolute_dir=  '../dataset/invivo'+ args.data_cell_name
    else:
        absolute_dir='../dataset'

    args.working_dir = working_path 

    #args.data_name = data_name
    gpu_index = 0
    res_desc = {}
    version=2
    for model_name in ['CNN']:
        args.model_name =  model_name# {'RNN','CNN', 'Transformer'}
        res_desc[model_name] = {}
        for exp_name in [ 'protospacer_PAM']:
            args.exp_name = exp_name
            args.data_dir  = absolute_dir + '/' + args.exp_name + '/'+ f'{args.data_name}_proportions_encenc_two_model'
            
            
            
            
            
            if not args.model_invivo:
                model_path = os.path.join(args.working_dir, 
                                      'output', 
                                      f'{model_name}_v{version}',args.saved_model, 
                                      exp_name)
            else:
                model_path = os.path.join(args.working_dir, 
                                      'output', 
                                      f'{model_name}_v{version}','invivo',args.model_cell_name, args.saved_model, 
                                      exp_name)
                
                
            print("\n"*3)    
            print("*"*25)
            print('loading model from:', model_path)
            
            print("\n"*3)    
            print("*"*25)
            print('reading dataset from:', args.data_dir)
            dpartitions, datatensor_partitions = get_data_ready(args, 
                                                                normalize_opt='max',
                                                                train_size=0.9, 
                                                                fdtype=torch.float32)

            train_val_path = os.path.join(model_path, 'train_val')
            
            
            
            
            if not transfer_learning:
                test_path = os.path.join(model_path, 'test')
                
            ## for transfer learning inference
            else:
                print('we are running transfer learning')
                if args.data_invivo and not args.model_invivo:
                    print('we are running vitro model and vivo data')
                    test_path = os.path.join(args.working_dir, 
                                  'output', 
                                  f'{model_name}_v{version}', 
                                         'transfer_learning', 'vitro_model_vivo_data', args.data_cell_name, args.data_name, 
                                  exp_name)
                elif not args.data_invivo and args.model_invivo:
                    print('we are running vivo model and vitro data')
                    
                    test_path = os.path.join(args.working_dir, 
                                  'output', 
                                  f'{model_name}_v{version}', 
                                         'transfer_learning', 'vivo_model_vitro_data', args.cell_name, args.data_name, 
                                  exp_name)
                elif args.data_invivo == args.model_invivo:
                    if args.data_name != args.saved_model:
                        print('we are running transfer learning within vivo/vitro')
                        test_path = os.path.join(args.working_dir, 
                                  'output', 
                                  f'{model_name}_v{version}', 
                                         'transfer_learning', f'model_{args.saved_model}_data_{args.data_name}', args.cell_name, args.data_name, 
                                  exp_name)
                    if args.data_cell_name != args.model_cell_name:
                
                        print('we are running transfer learning within vivo')
                        test_path = os.path.join(args.working_dir, 
                                  'output', 
                                  f'{model_name}_v{version}', 
                                         'transfer_learning', f'model_{args.model_cell_name}_data_{args.data_cell_name}', args.data_name, 
                                  exp_name)
                    
                else:
                    raise ValueError("the conditions does not match for transfer learning")
                    
            print("\n"*3)
            print("*"*25)
            print('the results are save at:', test_path)
            print(f'Running model: {model_name}, exp_name: {exp_name}, saved at {train_val_path}')
            a, b = run_inference(datatensor_partitions, 
                                 train_val_path, 
                                 test_path, 
                                 gpu_index,
                                 to_gpu=True)
            print('='*15)
            res_desc[model_name][exp_name] = compute_eval_results_df(test_path, len(dpartitions)) 
            return res_desc
        
        
if __name__ == "__main__":
    args, _ = cmd_opt.parse_known_args()
    res_desc = main_inference(args)
    print(res_desc)
    
