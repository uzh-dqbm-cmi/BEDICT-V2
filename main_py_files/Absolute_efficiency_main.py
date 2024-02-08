## this file is called to train absolute efficiency model
import os
import numpy as np
import pandas as pd
import torch
import argparse
working_path = '../absolute_efficiency_model'
import sys
sys.path.append(working_path)
print(working_path)
from models.data_process import  get_data_ready, get_datatensor_partitions
from models.trainval_workflow import run_trainevaltest_workflow
from models.hyperparam import build_config_map
from src.utils import  compute_eval_results_df, ReaderWriter, one_hot_encode, get_device


cmd_opt = argparse.ArgumentParser(description='Argparser for data')
cmd_opt.add_argument('-model_name',  type=str, help = 'name of the model')
cmd_opt.add_argument('-exp_name',  type=str, help = 'name of the experiment')
cmd_opt.add_argument('-data_name',  type=str,default = '', help = 'name of the data')
cmd_opt.add_argument('-data_dir',  type=str,default = './data/', help = 'directory of the data')
#cmd_opt.add_argument('-target_dir',  type=str, default='processed',  help = 'folder name to save the processed data')
cmd_opt.add_argument('-working_dir',  type=str, default='./', help = 'the main working directory')
#cmd_opt.add_argument('-output_path', type=str, help='path to save the trained model')
cmd_opt.add_argument('-random_seed', type=int,default=42)
cmd_opt.add_argument('-epoch_num', type=int, default =200, help='number of training epochs')
args, _ = cmd_opt.parse_known_args()


def get_hyperparam_config(args):
    to_gpu = True
    gpu_index = 0
    optim_tup = None

    if args.model_name == 'CNN':
        k = 2
        # l2_reg = 0.01
        l2_reg = 0.01
        batch_size = 100
        num_epochs = 300
        model_config_tup = (k, l2_reg, batch_size, num_epochs)

        mlpembedder_tup = None
        if args.exp_name == 'protospacer':
            input_dim = 20
        if args.exp_name == 'protospacer_PAM':
            input_dim = 24
            print(input_dim)
        if args.exp_name == 'protospacer_PAM_overhangs':
            input_dim = 34

        # loss_func_name = 'MSEloss'
        # loss_func_name = 'SmoothL1loss'
        perfmetric_name = 'spearman'
        loss_func_name = 'klloss'

    if args.model_name == 'RNN':
        embed_dim = 64
        hidden_dim = 64
        z_dim = 32
        num_hidden_layers = 2
        bidirection = True
        p_dropout = 0.1
        rnn_class = torch.nn.GRU
        nonlin_func = torch.nn.ReLU
        pooling_mode = 'none'
        l2_reg = 1e-5
        batch_size = 1500
        num_epochs = 500

        model_config_tup = (embed_dim, hidden_dim, z_dim, num_hidden_layers, bidirection,
                            p_dropout, rnn_class, nonlin_func, pooling_mode, l2_reg, batch_size, num_epochs)

        # input_dim, embed_dim, mlp_embed_factor, nonlin_func, p_dropout, num_encoder_units
        if args.exp_name == 'protospacer_PAM':
            mlpembedder_tup = None
            input_dim = 24
        elif args.exp_name == 'protospacer_PAM_overhangs':
            mlpembedder_tup = None
            input_dim = 34

        else:
            mlpembedder_tup = None
            input_dim = 20

        loss_func_name = 'SmoothL1loss'
        perfmetric_name = 'pearson'

    mconfig, options = build_config_map(args.model_name,
                                        optim_tup,
                                        model_config_tup,
                                        mlpembedder_tup,
                                        loss_func=loss_func_name)

    # print('we are here')
    options['input_size'] = input_dim
    options['loss_func'] = loss_func_name  # to refactor
    options['model_name'] = args.model_name
    options['perfmetric_name'] = perfmetric_name
    return mconfig, options


def main_train(data_name, screen_name,in_vivo, args):
    dsettypes = ['train', 'validation', 'test']
    gpu_index = 0
    res_desc = {}
    version = 2
    args.data_name = data_name
    for model_name in ['CNN']:  # [ 'RNN'  ,'FFN','CNN', 'RNN','Transformer']:
        print(model_name)
        args.model_name = model_name  # {'RNN','CNN', 'Transformer'}
        res_desc[model_name] = {}
        for exp_name in ['protospacer_PAM']:  # ,  'protospacer_PAM_overhangs']:#, 'protospacer_PAM','protospacer_PAM_overhangs']:
            args.exp_name = exp_name
            print(exp_name)
            args.data_dir = absolute_dir + '/' + args.exp_name + '/' + args.data_name + '_proportions_encenc_two_model'
            print(args.data_dir)
            
            if in_vivo:
                model_path = os.path.join(args.working_dir,
                                          'output',
                                          f'{model_name}_v{version}', 'invivo', screen_name, args.data_name,
                                          exp_name)
            else:
                model_path = os.path.join(args.working_dir,
                                          'output',
                                          f'{model_name}_v{version}', args.data_name,
                                          exp_name)
                
            dpartitions, datatensor_partitions = get_data_ready(args,
                                                                normalize_opt='max',
                                                                train_size=0.9,
                                                                fdtype=torch.float32,
                                                                plot_y_distrib=False)
            mconfig, options = get_hyperparam_config(args)
            print(options['input_size'])
            print(options['loss_func'])

            #         options['num_epochs'] = 10 # use this if you want to test a whole workflow run for all models using 10 epochs

            perfmetric_name = options['perfmetric_name']
            train_val_path = os.path.join(model_path, 'train_val')
            test_path = os.path.join(model_path, 'test')

            print(f'Running model: {model_name}, exp_name: {exp_name}, saved at {train_val_path}')
            perfmetric_run_map, score_run_dict = run_trainevaltest_workflow(datatensor_partitions,
                                                                            (mconfig, options),
                                                                            train_val_path,
                                                                            dsettypes,
                                                                            perfmetric_name,
                                                                            gpu_index,
                                                                            to_gpu=True)
            print('=' * 15)
            res_desc[model_name][exp_name] = compute_eval_results_df(train_val_path, len(dpartitions))


            
in_vivo = False
screen_name  = ''
dsettypes = ['train', 'validation','test']
gpu_index = 0
res_desc = {}
version=2
model_name = 'CNN'
print(model_name)
args.model_name =  model_name # {'RNN','CNN', 'Transformer'}
res_desc[model_name] = {}
args.working_dir = working_path

if in_vivo:
    absolute_dir=  '../dataset/invivo/' + screen_name
#args.data_name = 'ABEmax-NG'
else:
    absolute_dir=  '../dataset'
print(absolute_dir)

if not in_vivo:
    #for data_name in ['ABEmax-SpRY', 'ABEmax-SpCas9', 'ABEmax-NG', 'ABE8e-NG', 'ABE8e-SpRY', 'ABE8e-SpCas9']:
    for data_name in ['ABE8e-NG']:
        main_train(data_name, screen_name, in_vivo, args)


elif screen_name == 'Liver_LentiAAV':
    for editor_name in ['ABE8e-SpRY','ABEmax-SpRY']:
    #for editor_name in ['ABE8e-SpRY']:
        print('Running dataset:', editor_name)
        main_train(editor_name, screen_name, in_vivo, args)
        print('The end')
        
        
elif screen_name == 'Liver_LentiLNP':
    for editor_name in ['ABE8e-NG', 'ABE8e-SpRY', 'ABE8e-SpCas9','ABEmax-SpRY']:
    #for editor_name in ['ABE8e-SpRY']:
        print('Running dataset:', editor_name)
        main_train(editor_name, screen_name, in_vivo, args)
        print('The end')
        
elif screen_name == 'Liver_SBApproach':
    #for editor_name in ['ABE8e-SpRY']:
    for editor_name in ['ABEmax-SpRY']:
        print('Running dataset:', editor_name)
        main_train(editor_name, screen_name, in_vivo, args)
        print('The end')

else:
    print('please sepecify the screening type')