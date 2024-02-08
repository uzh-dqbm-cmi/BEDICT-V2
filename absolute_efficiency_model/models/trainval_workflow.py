import os
import numpy as np
import torch
from .FFN import RegressionFFNN
from .CNN import PredictionCNN
from .RNN import PredictionRNN
from .Transformer import PredictionTransformer
from src.utils import get_device, create_directory, ReaderWriter
from src.utils import perfmetric_report_regression, build_regression_df, plot_loss, dump_dict_content
from .data_process import construct_load_dataloaders
import torch.nn as nn

def create_perfmetric_map(dsettypes):
    perfmetric_map = {}
    for dsettype in dsettypes:
        perfmetric_map[dsettype] = []
    return perfmetric_map


def train_epoch(model, train_loader, optimizer, criterion, cyc_scheduler,loss_func_name, device='cpu'):
    model.train()
    train_loss = 0
    pred_class = []
    ref_class = []
    pred_nonlin_func = torch.nn.LogSoftmax(dim=-1)
    
    for batch_idx, sbatch in enumerate(train_loader):
        optimizer.zero_grad()

        if len(sbatch) == 3:
           
            x, y, seq_id = sbatch
            x = x.to(device)
            x_f = None
            y = y.to(device)
            
        elif len(sbatch) == 4:
            x, x_f, y, seq_id = sbatch
            x = x.to(device)
            x_f = x_f.to(device)
            y = y.to(device)
       
        
        out = model(x, x_f)
        #print(out.shape)
        out = pred_nonlin_func(out)
        #print(out.shape)
        if isinstance(out,tuple):# case of Transformer
            output, __, __ = out
        else:
            output = out
        #print(y.shape)
        #print(criterion)
        loss = criterion(output, y)
        
        loss = loss[:,0].sum()+ loss[:,1].sum()
        #print(loss)
        #print('train mini batch loss', loss)
        loss.backward()
        optimizer.step()
        cyc_scheduler.step()

        train_loss += loss.item()
        if loss_func_name == 'klloss':
            pred_prob = torch.exp(output[:,0])
            #print(pred_prob.max())
            true_prob = y[:,0]
        else: 
            pred_prob = output
            true_prob = y
            
        pred_class.extend(pred_prob.tolist())
        ref_class.extend(true_prob.tolist())
            
    train_loss /= len(train_loader)
    
    return train_loss, pred_class, ref_class,  optimizer, cyc_scheduler, seq_id

def model_eval(model, val_loader, criterion, loss_func_name,  device='cpu'):
    model.eval()
    val_loss = 0
    pred_class = []
    ref_class = []
    pred_nonlin_func = torch.nn.LogSoftmax(dim=-1)
    with torch.no_grad():
        for batch_idx, sbatch in enumerate(val_loader):
            if len(sbatch) == 3:
                x, y,seq_id = sbatch
                #print(y)
                x = x.to(device)
                
                x_f = None
                y = y.to(device)
            elif len(sbatch) == 4:
                x, x_f, y,seq_id = sbatch
                x = x.to(device)
                x_f = x_f.to(device)
                y = y.to(device)
            
            out = model(x, x_f)
            out = pred_nonlin_func(out)
            if isinstance(out,tuple):# case of Transformer
                output, __, __ = out
            else:
                output = out
            
            
            loss = criterion(output, y)
            
            #loss = loss[:,0].sum()+ 5*loss[:,1].sum()
            loss = loss[:,0].sum()+ loss[:,1].sum()
   
            val_loss += loss.item()
            
            if loss_func_name == 'klloss':
                pred_prob = torch.exp(output[:,0])
                true_prob = y[:,0]
            else: 
                pred_prob = output
                true_prob = y
            
            pred_class.extend(pred_prob.tolist())
            ref_class.extend(true_prob.tolist())
            
        val_loss /= len(val_loader)

    return val_loss, pred_class, ref_class,seq_id

def run_trainevaltest_workflow(datatensor_partitions, config_map, train_val_dir, dsettypes, perfmetric_name, gpu_indx, to_gpu=True):
    
    device = get_device(to_gpu, gpu_indx)  # gpu device
    print(device)
 
    if perfmetric_name == 'spearman':
        tmetric = 0
    elif perfmetric_name == 'pearson':
        tmetric = 1
    
    perfmetric_run_map = {}
    score_run_dict = {}
    
    config, options = config_map

   
    
    loss_func_name = options.get('loss_func', 'SmoothL1loss')
    
    if loss_func_name == 'SmoothL1loss':
        loss_func = torch.nn.SmoothL1Loss(reduction='mean')
    elif loss_func_name == 'MSEloss':
        loss_func = torch.nn.MSELoss(reduction='mean')
    elif loss_func_name == 'klloss':
        loss_func = nn.KLDivLoss(reduction="none")
    
    print('loss function', loss_func)
    num_runs = len(datatensor_partitions)
    for run_num in range(num_runs):
        
        perfmetric_map = create_perfmetric_map(dsettypes) #{'train':[], 'validation':[], 'test':[]}
        perfmetric_run_map[run_num] = perfmetric_map

        data_partition = datatensor_partitions[run_num]

        wrk_dir = create_directory('run_{}'.format(run_num), train_val_dir)
        m_state_dict_dir = create_directory('model_statedict', wrk_dir)
        fig_dir = create_directory('figures', wrk_dir)


        dataloader_config = config['dataloader_config']
        cld = construct_load_dataloaders(data_partition, dsettypes, 'regression', dataloader_config, wrk_dir)
        data_loaders, epoch_loss_avgbatch, score_dict,  flog_out = cld


        score_run_dict[run_num] = score_dict

        fdtype = options['fdtype']

        model_config = config['model_config']
        num_epochs = options.get('num_epochs', 500)
        print('number of epochs', num_epochs)

        
        model_name = options.get('model_name')
        mlpembedder_config = config.get('mlpembedder_config', None)

        input_size = options.get('input_size')

        # legacy support
        if input_size is None:
            if mlpembedder_config is not None:
                input_size = 20 + mlpembedder_config.input_dim
            else:
                input_size = 20
                
        if model_name == 'FFN':
            model = RegressionFFNN(80, model_config.h, mlp_embedder_config=mlpembedder_config)

        if model_name == 'CNN':
            input_dim = options.get('input_size')
            model = PredictionCNN(k=model_config.k,input_dim = input_dim, mlp_embedder_config=mlpembedder_config)
            
        elif model_name == 'RNN':
            input_dim = options.get('input_size')
            model = PredictionRNN(input_dim=input_size,
                                    embed_dim=model_config.embed_dim,
                                    hidden_dim=model_config.hidden_dim, 
                                    z_dim=model_config.z_dim,
                                    outp_dim=1,
                                    seq_len=input_dim,
                                    device=device,
                                    num_hiddenlayers=model_config.num_hidden_layers, 
                                    bidirection= model_config.bidirection, 
                                    rnn_pdropout=model_config.p_dropout, 
                                    rnn_class=model_config.rnn_class, 
                                    nonlinear_func=model_config.nonlin_func,
                                    pooling_mode=model_config.pooling_mode,
                                    mlp_embedder_config=mlpembedder_config,
                                    fdtype = fdtype)
            
        elif model_name == 'Transformer':
            model = PredictionTransformer(input_size=input_size,
                                        embed_size=model_config.embed_dim, 
                                        num_nucleotides=4, 
                                        seq_length=20, 
                                        num_attn_heads=model_config.num_attn_heads, 
                                        mlp_embed_factor=model_config.mlp_embed_factor, 
                                        nonlin_func=model_config.nonlin_func, 
                                        pdropout=model_config.p_dropout, 
                                        num_transformer_units=model_config.num_transformer_units,
                                        pos_embed_concat_opt=model_config.pos_embed_concat_opt,
                                        pooling_mode=model_config.pooling_opt,
                                        multihead_type=model_config.multihead_type,
                                        mlp_embedder_config=mlpembedder_config,
                                        num_classes=1)


        model.type(fdtype).to(device)

        # load optimizer config
        if('train' in data_loaders):
            optim_config = config['optimizer_config']

        if(not optim_config):
            weight_decay = options.get('weight_decay', 1e-4)
            print('weight_decay', weight_decay)
            num_iter = len(data_loaders['train'])  # num_train_samples/batch_size
            c_step_size = int(np.ceil(5*num_iter))  # this should be 2-10 times num_iter
            base_lr = 3e-4
            max_lr = 5*base_lr  # 3-5 times base_lr
            optimizer = torch.optim.Adam(model.parameters(), weight_decay=weight_decay, lr=base_lr)
            cyc_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr, max_lr, step_size_up=c_step_size,
                                                            mode='triangular', cycle_momentum=False)




        config_dir = create_directory('config', wrk_dir)
        ReaderWriter.dump_data(config, os.path.join(config_dir, 'mconfig.pkl'))
        ReaderWriter.dump_data(options, os.path.join(config_dir, 'exp_options.pkl'))


        best_loss = float('inf')
        best_perfscore = 0.
        best_model = None


        train_dataloader = data_loaders['train']
        valid_dataloader = data_loaders['validation']
        test_dataloader = data_loaders['test']


        for epoch in range(num_epochs):
            
            train_loss, pred_class, ref_class,  optimizer, cyc_scheduler,seq_id = train_epoch(model, 
                                                                                       train_dataloader, 
                                                                                       optimizer,
                                                                                       loss_func,
                                                                                       cyc_scheduler,loss_func_name, 
                                                                                       device=device)

            
            epoch_loss_avgbatch['train'].append(train_loss)
            
            
            modelscore_train = perfmetric_report_regression(np.array(pred_class), np.array(ref_class), epoch,  flog_out['train'])
            perfmetric_map['train'].append(modelscore_train.correlation)

            print('x'*25)
            print('we are validation phase')
            valid_loss, valid_y_pred, valid_y,val_id = model_eval(model, valid_dataloader, loss_func, loss_func_name,device)
            
            
            test_loss, test_y_pred, test_y, test_id = model_eval(model, test_dataloader, loss_func, loss_func_name,device)
            
            
            
            epoch_loss_avgbatch['validation'].append(valid_loss)
            epoch_loss_avgbatch['test'].append(test_loss)

         
            modelscore_validation = perfmetric_report_regression(np.array(valid_y_pred), np.array(valid_y), epoch, flog_out['validation'])
            print('x'*25)
            
            
            modelscore_test = perfmetric_report_regression(np.array(test_y_pred), np.array(test_y), epoch, flog_out['test'])
            print('x'*25)

            perfmetric_map['validation'].append(modelscore_validation.correlation)
            perfmetric_map['test'].append(modelscore_test.correlation)
            
            
            
            
            if modelscore_validation.correlation[tmetric] > best_perfscore: # use spearman or pearson as performance metric
                best_perfscore = modelscore_validation.correlation[tmetric]

                score_dict['train'] = modelscore_train
                score_dict['validation'] = modelscore_validation
                score_dict['test'] = modelscore_test


                print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {train_loss:.4f}, Validation Loss: {valid_loss:.4f}, Test loss: {test_loss:.4f}")
                print(f"Epoch {epoch+1}/{num_epochs}, best {perfmetric_name} corr. so far: {best_perfscore:.4f}")
                print('~'*25)



                ### save the validation performance
                val_predictions_df = build_regression_df(valid_y, valid_y_pred, val_id)
                val_predictions_df.to_csv(os.path.join(wrk_dir, 'predictions_validation.csv'))


                ### save the test performance
                test_predictions_df = build_regression_df(test_y, test_y_pred, test_id)
                test_predictions_df.to_csv(os.path.join(wrk_dir, 'predictions_test.csv'))

                ### save the model state
                best_model = model.state_dict()
                torch.save(best_model, os.path.join(m_state_dict_dir, '{}.pkl'.format(model_name)))



        if(num_epochs > 1):
            plot_loss(epoch_loss_avgbatch, fig_dir)
            dump_dict_content(score_dict, list(score_dict.keys()), 'score', wrk_dir)
    return perfmetric_run_map, score_run_dict


def run_inference(datatensor_partitions, train_val_dir, test_dir, gpu_indx, to_gpu=True):

    device = get_device(to_gpu, gpu_indx)  # gpu device
    print(device)
    
    perfmetric_run_map = {}
    score_run_dict = {}    

    perfmetric_run_map = {}
    score_run_dict = {}    
    dsettypes = ['test']

    for run_num in range(len(datatensor_partitions)):
        
        perfmetric_map = create_perfmetric_map(dsettypes) #{'test':[]}
        perfmetric_run_map[run_num] = perfmetric_map

        data_partition = datatensor_partitions[run_num]

        wrk_dir = create_directory('run_{}'.format(run_num), train_val_dir)
        
        state_dict_pth = None
        if os.path.exists(wrk_dir):
            # load state_dict pth
            state_dict_pth = os.path.join(wrk_dir, 'model_statedict')
            
            # load mconfig
            mconfig = ReaderWriter.read_data(os.path.join(wrk_dir, 'config', 'mconfig.pkl'))
            
            # load exp_options
            exp_options = ReaderWriter.read_data(os.path.join(wrk_dir, 'config', 'exp_options.pkl'))
            
        # create a test directory
        test_pth =  create_directory('run_{}'.format(run_num), test_dir)
        

        loss_func_name = exp_options['loss_func']
        print(loss_func_name)
        if loss_func_name == 'SmoothL1loss':
            loss_func = torch.nn.SmoothL1Loss(reduction='mean')
        elif loss_func_name == 'MSEloss':
            loss_func = torch.nn.MSELoss(reduction='mean')
        elif loss_func_name == 'klloss':
            loss_func = nn.KLDivLoss(reduction="none")
            

        dataloader_config = mconfig['dataloader_config']
        cld = construct_load_dataloaders(data_partition, dsettypes, 'regression', dataloader_config, wrk_dir)
        data_loaders, epoch_loss_avgbatch, score_dict,  flog_out = cld


        score_run_dict[run_num] = score_dict

        fdtype = exp_options['fdtype']
        num_epochs = 1
        print('number of epochs', num_epochs)

        
        
        model_config = mconfig['model_config']
        mlpembedder_config = mconfig.get('mlpembedder_config', None)


        input_size = exp_options.get('input_size')
        # legacy support
        if input_size is None:
            if mlpembedder_config is not None:
                input_size = 20 + mlpembedder_config.input_dim
            else:
                input_size = 20

        
        model_name = exp_options.get('model_name')
        print('model_name:', model_name)
        print('input_size:', input_size)

        if model_name == 'Transformer':
            model = PredictionTransformer(input_size=input_size,
                                            embed_size=model_config.embed_dim, 
                                            num_nucleotides=4, 
                                            seq_length=20, 
                                            num_attn_heads=model_config.num_attn_heads, 
                                            mlp_embed_factor=model_config.mlp_embed_factor, 
                                            nonlin_func=model_config.nonlin_func, 
                                            pdropout=model_config.p_dropout, 
                                            num_transformer_units=model_config.num_transformer_units,
                                            pos_embed_concat_opt=model_config.pos_embed_concat_opt,
                                            pooling_mode=model_config.pooling_opt,
                                            multihead_type=model_config.multihead_type,
                                            mlp_embedder_config=mlpembedder_config,
                                            num_classes=1)
        elif model_name == 'CNN':
            input_dim = exp_options.get('input_size')
            model = PredictionCNN(k=model_config.k,input_dim = input_dim, mlp_embedder_config=mlpembedder_config)
            #model = PredictionCNN(k=model_config.k, mlp_embedder_config=mlpembedder_config)
            
        elif model_name == 'FFN':
            model = RegressionFFNN(80, model_config.h, mlp_embedder_config=mlpembedder_config)

        elif model_name == 'RNN':
            input_dim = exp_options.get('input_size')
            model = PredictionRNN(input_dim=input_size,
                                    embed_dim=model_config.embed_dim,
                                    hidden_dim=model_config.hidden_dim, 
                                    z_dim=model_config.z_dim,
                                    outp_dim=1,
                                    seq_len=input_dim,
                                    device=device,
                                    num_hiddenlayers=model_config.num_hidden_layers, 
                                    bidirection= model_config.bidirection, 
                                    rnn_pdropout=model_config.p_dropout, 
                                    rnn_class=model_config.rnn_class, 
                                    nonlinear_func=model_config.nonlin_func,
                                    pooling_mode=model_config.pooling_mode,
                                    mlp_embedder_config=mlpembedder_config,
                                    fdtype = fdtype)
        
        if(state_dict_pth):  # load state dictionary of saved models
            model.load_state_dict(torch.load(os.path.join(state_dict_pth, f'{model_name}.pkl'), map_location=device))


        model.type(fdtype).to(device)

        
        # save the model config in the test directory as logging mechanism
        config_dir = create_directory('config', test_pth)
        ReaderWriter.dump_data(mconfig, os.path.join(config_dir, 'mconfig.pkl'))
        ReaderWriter.dump_data(exp_options, os.path.join(config_dir, 'exp_options.pkl'))


        test_dataloader = data_loaders['test']

        for epoch in range(num_epochs):

            
            test_loss, test_y_pred, test_y, test_id = model_eval(model, test_dataloader, loss_func,loss_func_name, device)
            
            #print(test_y_pred)

            epoch_loss_avgbatch['test'].append(test_loss)

            
            modelscore_test = perfmetric_report_regression(np.array(test_y_pred), np.array(test_y), epoch, flog_out['test'])
            print('x'*25)

            perfmetric_map['test'].append(modelscore_test.correlation)
            score_dict['test'] = modelscore_test


            ### save the test performance
            test_predictions_df = build_regression_df(test_y, test_y_pred,test_id)
            print('saving the result to ',test_pth)
            test_predictions_df.to_csv(os.path.join(test_pth, 'predictions_test.csv'))

            ### save the model state 
            sdir = create_directory('model_statedict', test_pth)
            torch.save(model.state_dict(),os.path.join(sdir, f'{model_name}.pkl'))

    return perfmetric_run_map, score_run_dict