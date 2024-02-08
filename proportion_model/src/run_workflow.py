import os
import datetime
import itertools
import numpy as np
import torch
from torch import nn
import torch.multiprocessing as mp
from .utilities import get_device, create_directory, ReaderWriter, dump_dict_content, \
                       perfmetric_report_cont,compute_harmonic_mean, \
                       plot_loss, plot_xy, build_predictions_df, build_classification_df, build_probscores_df,\
                       update_Adamoptimizer_lr_momentum_, compute_lr_scheduler, compute_momentum_scheduler, restrict_grad_
from .model import Encoder, MLPDecoder, HaplotypeEncoderEncoder, MaskGenerator
from .dataset import construct_load_dataloaders
from .hyperparam import Haplotype_Trf_HyperparamConfig, get_hyperparam_options
#from .loss import CELoss



def generate_models_config(hyperparam_config, optim_config, 
                           experiment_desc, model_name, run_num, fdtype,
                           opt_adendum=None,
                           loss_func='klloss'):
    dataloader_config = {'batch_size': hyperparam_config.batch_size,
                         'num_workers': 0}
    config = {'dataloader_config': dataloader_config,
              'model_config': hyperparam_config,
              'optimizer_config': optim_config
             }

    options = {'experiment_desc': experiment_desc,
               'run_num': run_num,
               'model_name': model_name,
               'num_epochs': hyperparam_config.num_epochs,
               'weight_decay': hyperparam_config.l2_reg,
               'fdtype':fdtype,
               'to_gpu':True,
               'loss_func':loss_func
               }
    if opt_adendum:
        options.update(opt_adendum)
    return config, options


def build_custom_config_map(experiment_desc, model_name, loss_func='nllloss'):
    if(model_name in {'HaplotypeTransformer', 'HaplotypeEncoderEncoder'}):
        hyperparam_config = Haplotype_Trf_HyperparamConfig(32, 8, 12, 0.3, nn.ReLU(), 2, 'Narrow', 0, 200, 20)
    optim_config = {'lr':{'l0':1e-4, 'lmax':5*1e-4},
                    'momentum':{'m0':0.85, 'mmax':0.95},
                    'annealing_percent':0.1,
                    'stop_crit':5}
    run_num = -1 
    fdtype = torch.float32
    mconfig, options = generate_models_config(hyperparam_config, optim_config, 
                                              experiment_desc, model_name, run_num, fdtype, 
                                              loss_func=loss_func)
    return mconfig, options

def build_config_map(experiment_desc, model_name, optim_tup, trf_tup, opt_adendum=None,loss_func='klloss'):
    if(model_name in {'HaplotypeTransformer', 'HaplotypeEncoderEncoder'}):
        hyperparam_config = Haplotype_Trf_HyperparamConfig(*trf_tup)
    if optim_tup:
        optim_config = {'lr':{'l0':optim_tup[0], 'lmax':optim_tup[1]},
                        'momentum':{'m0':optim_tup[2], 'mmax':optim_tup[3]},
                        'annealing_percent':optim_tup[4],
                        'stop_crit':optim_tup[5]}
    else:
        optim_config = {}
    run_num = -1 
    fdtype = torch.float32
    mconfig, options = generate_models_config(hyperparam_config, optim_config,
                                              experiment_desc, model_name, 
                                              run_num, fdtype, 
                                              opt_adendum=opt_adendum,
                                              loss_func=loss_func)
    return mconfig, options

# def process_multilayer_multihead_attn(attn_dict, seqs_id):
#     attn_dict_perseq = {}
#     for l in attn_dict:
#         for h in attn_dict[l]:
#             tmp = attn_dict[l][h].detach().cpu()
#             for count, seq_id in enumerate(seqs_id):
#                 if(seq_id not in attn_dict_perseq):
#                     attn_dict_perseq[seq_id] = {} 
#                 if(l in attn_dict_perseq[seq_id]):
#                     attn_dict_perseq[seq_id][l].update({h:tmp[count]})
#                 else:
#                     attn_dict_perseq[seq_id][l] = {h:tmp[count]}
#     return attn_dict_perseq

def hyperparam_model_search(data_partitions, experiment_desc, model_name,
                            root_dir, run_gpu_map, loss_func='klloss', 
                            fdtype=torch.float32, num_epochs=25,
                            prob_interval_truemax=0.05, prob_estim=0.95, random_seed=42
                            ):
    # run_num = get_random_run(len(data_partitions), random_seed=random_seed)
    run_num = 0
    dsettypes = ['train', 'validation']
    hyperparam_options = get_hyperparam_options(prob_interval_truemax, prob_estim, model_name)
    data_partition = data_partitions[run_num]
    for counter, hyperparam_config in enumerate(hyperparam_options):
        mconfig, options = generate_models_config(hyperparam_config, {}, 
                                                  experiment_desc, model_name, 
                                                  run_num, fdtype, loss_func=loss_func)
        options['num_epochs'] = num_epochs # override number of ephocs here
        options['train_flag'] = True
        print("Running experiment {} config #{}".format(experiment_desc, counter))
        path = os.path.join(root_dir, 
                            'run_{}'.format(run_num),
                            'config_{}'.format(counter))
        wrk_dir = create_directory(path)

        if model_name == 'HaplotypeEncoderEncoder':
            run_cont_HapEncEnc(data_partition, dsettypes, mconfig, options, wrk_dir,
                        state_dict_dir=None, to_gpu=True, gpu_index=run_gpu_map[run_num])
        print("-"*15)

def get_weights_for_haplotypes(target_prob):
    """define weighting scheme for each haplotype depending on the target probability"""
    # cond_between = (target_prob >=0.1) & (target_prob <=0.85)
    # cond_extreme = ~ cond_between
    # weights = cond_between*2.0 + cond_extreme*1.
    cond_zero = target_prob <=0.1
    cond_above = ~ cond_zero
    weights = cond_zero*0.01 + cond_above*10.
    # weights has same shape as target_prob (bsize, num_haplotypes)
    return weights

def run_cont_HapEncEnc(data_partition, dsettypes, config, options, wrk_dir, state_dict_dir=None, to_gpu=True, gpu_index=0):
    pid = "{}".format(os.getpid())  # process id description
    # get data loader config
    dataloader_config = config['dataloader_config']
    cld = construct_load_dataloaders(data_partition, dsettypes, 'cont', dataloader_config, wrk_dir)
    # dictionaries by dsettypes
    data_loaders, epoch_loss_avgbatch, score_dict, flog_out, = cld
    # print(flog_out)
    device = get_device(to_gpu, gpu_index)  # gpu device
    fdtype = options['fdtype']
    loss_type = options.get('loss_func', 'klloss')

    #
    if loss_type == 'klloss':
        loss_func = nn.KLDivLoss(reduction='none')
    elif loss_type == 'CEloss':
        loss_func = CELoss(reduction='none')
    elif loss_type == 'MSEloss':
        loss_func = nn.MSELoss(reduction='none')
    # TODO: figure out weighting scheme to pass
    # for now we define the weighting in :func:`get_weights_for_haplotypes`
    weight_haplotypes = options.get('weight_haplotypes', False) # default no weighting is applied
    # default use only target bases when predicting the probability of a haplotype/bystander sequence
    mask_other_bases = options.get('mask_nontarget_bases', True) 

    num_epochs = options.get('num_epochs', 50)
    run_num = options.get('run_num')
    
    # parse config dict
    model_config = config['model_config']
    model_name = options['model_name']

    if(model_name == 'HaplotypeEncoderEncoder'):
        inp_seqlen = options.get('inp_seqlen')
        encoder = Encoder(model_config.embed_dim,
                          num_nucleotides=4, 
                          seq_length=inp_seqlen, 
                          num_attn_heads=model_config.num_attn_heads, 
                          mlp_embed_factor=model_config.mlp_embed_factor,
                          nonlin_func=model_config.nonlin_func, 
                          pdropout=model_config.p_dropout, 
                          num_encoder_units=model_config.num_transformer_units, 
                          pooling_mode='attn', 
                          multihead_type=model_config.multihead_type)

        outp_seqlen = options.get('outp_seqlen')
        encoder_byst = Encoder(model_config.embed_dim,
                          num_nucleotides=4, 
                          seq_length=outp_seqlen, 
                          num_attn_heads=model_config.num_attn_heads, 
                          mlp_embed_factor=model_config.mlp_embed_factor,
                          nonlin_func=model_config.nonlin_func, 
                          pdropout=model_config.p_dropout, 
                          num_encoder_units=model_config.num_transformer_units, 
                          pooling_mode='attn', 
                          multihead_type=model_config.multihead_type)
        # in this setup we have to have inp_seqlen and outp_seqlen to be equal
        assert inp_seqlen == outp_seqlen
        print('inp_seqlen:', inp_seqlen, 'outp_seqlen:', outp_seqlen)
        

        out_dec = MLPDecoder(inp_dim=2*model_config.embed_dim, 
                             outp_dim=2,
                             seq_length=inp_seqlen)
        encoder_decoder = HaplotypeEncoderEncoder(encoder=encoder, 
                                                  encoder_bystander=encoder_byst,
                                                  mlp_decoder=out_dec)
    

    # define optimizer and group parameters
    models_param = list(encoder_decoder.parameters())
    models = [(encoder_decoder, model_name)]

    if(state_dict_dir):  # load state dictionary of saved models
        for m, m_name in models:
            m.load_state_dict(torch.load(os.path.join(state_dict_dir, '{}.pkl'.format(m_name)), map_location=device))

    # update models fdtype and move to device
    for m, m_name in models:
        m.type(fdtype).to(device)

    if('train' in data_loaders):
        optim_config = config['optimizer_config']
        if(not optim_config):
            weight_decay = options.get('weight_decay', 1e-4)
            # see paper Cyclical Learning rates for Training Neural Networks for parameters' choice
            # `https://arxive.org/pdf/1506.01186.pdf`
            # pytorch version >1.1, scheduler should be called after optimizer
            # for cyclical lr scheduler, it should be called after each batch update
            num_iter = len(data_loaders['train'])  # num_train_samples/batch_size
            c_step_size = int(np.ceil(5*num_iter))  # this should be 2-10 times num_iter
            base_lr = 3e-4
            max_lr = 5*base_lr  # 3-5 times base_lr
            optimizer = torch.optim.Adam(models_param, weight_decay=weight_decay, lr=base_lr)
            cyc_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr, max_lr, step_size_up=c_step_size,
                                                            mode='triangular', cycle_momentum=False)
        else:
            weight_decay = options.get('weight_decay', 1e-4)
            l0 = optim_config['lr']['l0']
            lmax = optim_config['lr']['lmax']
            momen_0 = optim_config['momentum']['m0']
            momen_max = optim_config['momentum']['mmax']
            annealing_percent = optim_config['annealing_percent']
            # pytorch version >1.1, scheduler should be called after optimizer
            num_iter = len(data_loaders['train']) # num of minibatches
            optimizer = torch.optim.Adam(models_param, betas=(momen_max, 0.99), weight_decay=weight_decay, lr=l0)
            lrates = compute_lr_scheduler(l0, lmax, num_iter+1, annealing_percent)
            momentum_rates = compute_momentum_scheduler(momen_0, momen_max, num_iter+1, annealing_percent)
    

    if ('validation' in data_loaders):
        m_state_dict_dir = create_directory(os.path.join(wrk_dir, 'model_statedict'))

    if(num_epochs > 1):
        fig_dir = create_directory(os.path.join(wrk_dir, 'figures'))

    # dump config dictionaries on disk
    config_dir = create_directory(os.path.join(wrk_dir, 'config'))
    ReaderWriter.dump_data(config, os.path.join(config_dir, 'mconfig.pkl'))
    ReaderWriter.dump_data(options, os.path.join(config_dir, 'exp_options.pkl'))

    # store attention weights for validation and test set
    seqid_fattnw_map = {dsettype: {} for dsettype in data_loaders if dsettype in {'test'}}

    mask_gen = MaskGenerator()
    
    if loss_type in {'CEloss', 'klloss'}:
        print(loss_type, '|| logsoftmax')
        pred_nonlin_func = torch.nn.LogSoftmax(dim=-1)
    else:
        print(loss_type, '|| softmax')
        pred_nonlin_func = torch.nn.Softmax(dim=-1)

    for epoch in range(num_epochs):
        # print("-"*35)
        now = datetime.datetime.now()
        
        for dsettype in dsettypes:
            print("device: {} | experiment_desc: {} | run_num: {} | epoch: {} | dsettype: {} | pid: {}"
                  "".format(device, options.get('experiment_desc'), run_num, epoch, dsettype, pid))
            pred_score = []
            ref_score = []
            seqs_ids_lst = []
            seqs_inp_lst = []
            outpseqs_ids_lst = []
            data_loader = data_loaders[dsettype]
            
            # get a hold of the dataset to get pointer to dictionaries we will use
            # {indx[int]:(seq_id, inp_seq)}
            indx_seqid_map = data_loader.dataset.dtensor.indx_seqid_map
           
            # {indx[int]:[outpseq_lst]}
            inpseq_outpseq_map = data_loader.dataset.dtensor.inpseq_outpseq_map


            # total_num_samples = len(data_loader.dataset)
            epoch_loss = 0.

            if(dsettype == 'train'):  # should be only for train
                for m, m_name in models:
                    m.train()
            else:
                for m, m_name in models:
                    m.eval()

            # going over batches
            
            for indx_batch, sbatch in enumerate(data_loader):
                
                
                # zero model grad
                if(dsettype == 'train'):
                    optimizer.zero_grad()
                # print(len(sbatch))
                Xinp_enc, Xinp_dec, num_haplotype, mask_targetbase_enc, target_conv_onehot, target_prob, b_indx = sbatch
                
                # list of input sequence ids
                b_seqid = [indx_seqid_map[c.item()][0] for c in b_indx]
                # list of input sequence
                b_inpseq = [indx_seqid_map[c.item()][1] for c in b_indx]


                Xinp_enc = Xinp_enc.to(device)
                Xinp_dec = Xinp_dec.to(device)
                #print('Xinp_enc', Xinp_enc.shape)
                #print('Xinp_dec', Xinp_dec.shape)
                mask_targetbase_enc = mask_targetbase_enc.to(device)
                target_conv_onehot = target_conv_onehot.to(device)
                target_prob = target_prob.to(device)
                     
                with torch.set_grad_enabled(dsettype == 'train'):
                    pred_logprob, fattn_norm_dec, attn_mlayer_mhead_dec_dict= encoder_decoder(Xinp_enc, Xinp_dec)
                
                    #  print('pred_logprob.shape:', pred_logprob.shape)
                    # (bsize, num_haplotypes, seqlen, 1, 2)
                    pred_logprob_resh = pred_logprob.unsqueeze(-2)
                    #  print('pred_logprob_resh.shape:', pred_logprob_resh.shape)
                    #  print('target_conv_onehot.shape:', target_conv_onehot.shape)
                    # (bsize, num_haplotypes, seqlen, 2, 1)
                    conv_onehot_resh = target_conv_onehot.unsqueeze(-1)
                    # (bsize, num_haplotypes, seqlen, 1, 1)
                    out = pred_logprob_resh.matmul(conv_onehot_resh.type(fdtype))
                    # (bsize, num_haplotypes, seqlen)
                    out = out.squeeze(-1).squeeze(-1)
                    
                    # this will zero out non target base contriubtion before doing the sum
                    # (bsize, num_haplotypes)
                    if mask_other_bases:
                        pred_hap_logprob = (out*mask_targetbase_enc).sum(axis=-1)
                    else:
                        pred_hap_logprob = out.sum(axis=-1)
                    
                    # this will zero out contribution of the padded entries in the haplotypes dimension
                    haplo_mask = mask_gen.create_haplotype_mask(pred_hap_logprob.shape, num_haplotype).to(device)

                    # we recompute logsoftmax or softmax on the output sequences (enforcing the distribution)
                    pred_hap_logprob = pred_nonlin_func(pred_hap_logprob+haplo_mask)

                    l = loss_func(pred_hap_logprob, target_prob)

                    if weight_haplotypes:
                        weights = get_weights_for_haplotypes(target_prob)
                        l = l*weights
                    # in case of using nn.KLDivLoss, this is equivalent to having reduction='batchmean'
                    batch_loss = l.sum(axis=-1).mean()

                
                    if(dsettype in seqid_fattnw_map):
                        # reshape to (bsize, num_haplotypes, seq_len, seq_len)
                        fattn_norm_dec_resh =fattn_norm_dec.reshape(list(Xinp_dec.shape)+[Xinp_dec.shape[-1]])
                        seqid_fattnw_map[dsettype].update({seqid:fattn_norm_dec_resh[c].detach().cpu() for c, seqid in enumerate(b_seqid)})
                    
                    if loss_type in {'klloss', 'CEloss'}:
                        for c in range(pred_hap_logprob.shape[0]):
                            pred_score.extend(torch.exp(pred_hap_logprob[c,:num_haplotype[c].item()]).view(-1).tolist())
                    else:
                        for c in range(pred_hap_logprob.shape[0]):
                            pred_score.extend((pred_hap_logprob[c,:num_haplotype[c].item()]).view(-1).tolist())


                    for c in range(target_prob.shape[0]):
                        ref_score.extend(target_prob[c,:num_haplotype[c].item()].view(-1).tolist())

                    for c, elm in enumerate(b_seqid):
                        seqs_ids_lst.extend([elm]*num_haplotype[c].item())

                    for c, elm in enumerate(b_inpseq):
                        seqs_inp_lst.extend([elm]*num_haplotype[c].item()) 

                    for elm in b_indx:
                        outpseqs_ids_lst.extend(inpseq_outpseq_map[elm.item()])

                if(dsettype == 'train'):
                    # print("computing loss")
                    # backward step (i.e. compute gradients)
                    batch_loss.backward()
                    # optimzer step -- update weights
                    optimizer.step()
                    # after each batch step the scheduler
                    if(not optim_config):
                        cyc_scheduler.step()
                    else:
                        update_Adamoptimizer_lr_momentum_(optimizer, 
                                                          lrates[indx_batch+1], 
                                                          momentum_rates[indx_batch+1])
                        # print(optimizer)

                epoch_loss += batch_loss.item()


            epoch_loss_avgbatch[dsettype].append(epoch_loss/len(data_loader))
            modelscore = perfmetric_report_cont(pred_score, ref_score, 
                                                epoch_loss_avgbatch[dsettype][-1], 
                                                epoch, flog_out[dsettype])
            #print('model score', modelscore)
            perf = compute_harmonic_mean(modelscore.spearman_corr,
                                         modelscore.pearson_corr)
            best_score = compute_harmonic_mean(score_dict[dsettype].spearman_corr,
                                               score_dict[dsettype].pearson_corr)

            if(perf > best_score):
                score_dict[dsettype] = modelscore
                if(dsettype == 'validation'):
                    for m, m_name in models:
                        torch.save(m.state_dict(), os.path.join(m_state_dict_dir, '{}.pkl'.format(m_name)))
                        ReaderWriter.dump_data({'epoch':epoch+1}, os.path.join(m_state_dict_dir, 'best_epoch.pkl'))
                    # dump attention weights for the validation data for the best peforming model
                    # dump_dict_content(seqid_fattnw_map, ['validation'], 'seqid_fattnw_map', wrk_dir)
                    # dump_dict_content(seqid_mlhattnw_map, ['validation'], 'seqid_mlhattnw_map', wrk_dir)
                elif(dsettype == 'test'):
                    # dump attention weights for the validation data
                    dump_dict_content(seqid_fattnw_map, ['test'], 'seqid_fattnw_map', wrk_dir)
                    # dump_dict_content(seqid_mlhattnw_map, ['test'], 'seqid_mlhattnw_map', wrk_dir)
                    # save predictions for test
                # if dsettype in {'test', 'validation'}: # you can add validation to the mix
                if dsettype in {'test'}:
                    #print('we are here')
                    predictions_df = build_predictions_df(zip(seqs_ids_lst, seqs_inp_lst), outpseqs_ids_lst, ref_score, pred_score)
                    predictions_path = os.path.join(wrk_dir, f'predictions_{dsettype}.csv')
                    predictions_df.to_csv(predictions_path)
                    #print(predictions_path)

    if(num_epochs > 1):
        for dsettype in epoch_loss_avgbatch:
            plot_xy(np.arange(num_epochs), epoch_loss_avgbatch[dsettype], 
                    'number of epochs', 
                    f'{loss_type} loss', 
                    'epoch batch average loss', 
                    dsettype,
                    fig_dir)
    # dump_scores
    dump_dict_content(score_dict, list(score_dict.keys()), 'score', wrk_dir)


# def train_val_partition(datatensor_partition, config_map, tr_val_dir, run_gpu_map, queue):
#     """To use this in multiprocessing module"""
#     mconfig, options = config_map
#     num_epochs = options['num_epochs']
#     print('number of epochs:', num_epochs)
#     # note: datatensor_partition and run_gpu_map are 1-entry dictionaries
#     gpu_index = list(run_gpu_map.values())[0]
#     partition_index = list(run_gpu_map.keys())[0]
#     print('-- partition_index:', partition_index, 'gpu_index:', gpu_index, '--')
#     train_val_run(datatensor_partition, config_map, tr_val_dir, run_gpu_map, num_epochs=num_epochs)
#     queue.put(gpu_index)

def train_test_partition(datatensor_partition, config_map, tr_val_dir, run_gpu_map, queue):
    """To use this in multiprocessing module"""
    mconfig, options = config_map
    num_epochs = options['num_epochs']
    print('number of epochs:', num_epochs)
    # note: datatensor_partition and run_gpu_map are 1-entry dictionaries
    gpu_index = list(run_gpu_map.values())[0]
    partition_index = list(run_gpu_map.keys())[0]
    print('-- partition_index:', partition_index, 'gpu_index:', gpu_index, '--')
    train_val_run(datatensor_partition, config_map, tr_val_dir, run_gpu_map, num_epochs=num_epochs)
    test_run(datatensor_partition, config_map, tr_val_dir, tr_val_dir, run_gpu_map, num_epochs=1)
    queue.put(gpu_index)

def train_val_run(datatensor_partitions, config_map, train_val_dir, run_gpu_map, num_epochs=20):
    dsettypes = ['train', 'validation']
    mconfig, options = config_map
    options['num_epochs'] = num_epochs  # override number of epochs using user specified value
    options['train_flag'] = True

    for run_num in datatensor_partitions:
        # update options run num to the current run
        options['run_num'] = run_num
        data_partition = datatensor_partitions[run_num]
        # tr_val_dir = create_directory(train_val_dir)
        path = os.path.join(train_val_dir, 'train_val', 'run_{}'.format(run_num))
        wrk_dir = create_directory(path)
        # print(wrk_dir)
        config_num = os.path.basename(train_val_dir)
        print(f'config_{config_num}')

        if options.get('model_name') == 'HaplotypeEncoderEncoder':
            run_cont_HapEncEnc(data_partition, dsettypes, mconfig, options, wrk_dir,
                        state_dict_dir=None, to_gpu=True, gpu_index=run_gpu_map[run_num])
            

def test_run(datatensor_partitions, config_map, train_val_dir, test_dir, run_gpu_map, num_epochs=1):
    dsettypes = ['test']
    device = torch.cuda.is_available()
    mconfig, options = config_map
    options['num_epochs'] = num_epochs  # override number of epochs using user specified value
    options['train_flag'] = False
    for run_num in datatensor_partitions:
        # update options fold num to the current fold
        options['run_num'] = run_num
        data_partition = datatensor_partitions[run_num]
        train_dir = os.path.join(train_val_dir, 'train_val', 'run_{}'.format(run_num))
        
        train_dir = create_directory('run_{}'.format(run_num), os.path.join(train_val_dir, 'train_val'))
        
        if os.path.exists(train_dir):
            # load state_dict pth
            state_dict_pth = os.path.join(train_dir, 'model_statedict')
            path = os.path.join(test_dir, dsettypes[0])
            test_wrk_dir = create_directory('run_{}'.format(run_num), path)
            print('test dir', test_wrk_dir)
            
            if options.get('model_name') == 'HaplotypeEncoderEncoder':
                run_cont_HapEncEnc(data_partition, dsettypes, mconfig, options, test_wrk_dir,
                            state_dict_dir=state_dict_pth, to_gpu=device, gpu_index=run_gpu_map[run_num])
        else:
            print('WARNING: train dir not found: {}'.format(path))

def test_one_vs_rest(datatensor_partitions, config_map, train_val_dir, test_dir, run_gpu_map, target_run, num_epochs=1):
    dsettypes = ['train', 'validation', 'test']
    mconfig, options = config_map
    options['num_epochs'] = num_epochs  # override number of epochs using user specified value
    options['train_flag'] = False
    for run_num in datatensor_partitions:
        # update options fold num to the current fold
        options['run_num'] = run_num
        data_partition = datatensor_partitions[run_num]
        train_dir = create_directory(os.path.join(train_val_dir, 'train_val', 'run_{}'.format(target_run)))
        if os.path.exists(train_dir):
            # load state_dict pth
            state_dict_pth = os.path.join(train_dir, 'model_statedict')
            path = os.path.join(test_dir, 'test', 'run_{}_{}'.format(target_run, run_num))
            test_wrk_dir = create_directory(path)

            if options.get('model_name') == 'HaplotypeEncoderEncoder':
                run_cont_HapEncEnc(data_partition, dsettypes, mconfig, options, test_wrk_dir,
                            state_dict_dir=state_dict_pth, to_gpu=True, gpu_index=run_gpu_map[run_num])
    
        else:
            print('WARNING: train dir not found: {}'.format(path))
