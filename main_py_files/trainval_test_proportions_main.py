 ## lets train the model on all the dataset
if __name__ == '__main__':
    import os
    import numpy as np
    import pandas as pd
    import torch
    from torch import nn

    curr_pth = os.path.abspath('../')
    import sys
    sys.path.append(curr_pth)
    from  proportion_model.src.run_workflow import train_test_partition
    from utils.utilities import create_directory,ReaderWriter,build_performance_dfs
    from proportion_model.src.run_workflow import build_config_map
    import torch.multiprocessing as mp
    import datetime
    mp.set_start_method("spawn", force=True)


    def main(editor_name, input_type,version_name, invivo, screen_name ):

        #1. read data from disk
        curr_pth = os.path.abspath('../')

        data_dir = create_directory(os.path.join(curr_pth, 'dataset',input_type))
        if invivo is True:
            print('running in vivo data', screen_name)
            data_dir = create_directory(os.path.join(curr_pth, 'dataset', 'final_dataset','invivo',screen_name, input_type))


        suffix = 'proportions_encenc_two_model'
        fname = f'{editor_name}_{suffix}'
        target_data_dir = create_directory(os.path.join(data_dir,fname))
        dpartitions = ReaderWriter.read_data(os.path.join(target_data_dir, f'data_partitions.pkl'))
        num_runs = len(dpartitions)
        datatensor_partitions =  ReaderWriter.read_data(os.path.join(target_data_dir, f'dtensor_partitions.torch'))
        #2. set up environment, device, parameters
        gpu_indices = list(range(len(dpartitions)))
        run_gpu_maps = {i:indx for i, indx in enumerate(gpu_indices)}

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
        #batch_size = 40
        num_epochs = 150
        optim_tup = None
        experiment_desc = f'{fname}_schwank'
        # possible losses are {'klloss', 'CEloss', 'MSEloss'}
        loss_func = 'klloss'
        # loss_func = 'CEloss'
        # loss_func = 'MSEloss'
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

        #options['num_epochs'] = 300
        config_map = mconfig, options
        run_gpu_map = {0:run_gpu_maps[0]}


        ## 3. define ouput dir
        experiment_desc = f'{fname}'
        output_dir = os.path.abspath('../proportion_model/output')

        exp_dir = create_directory(os.path.join(output_dir, f'experiment_run_{suffix}', experiment_desc, input_type))
        if invivo is True:
            exp_dir = create_directory(
                os.path.join(output_dir, f'experiment_run_{suffix}',screen_name,  experiment_desc, input_type))

        time_stamp = version_name
        tr_val_dir = create_directory(f'exp_{time_stamp}', exp_dir)
        print(tr_val_dir)

        ## 4. run the model
        # https://github.com/facebookresearch/maskrcnn-benchmark/issues/103
        n_gpu = torch.cuda.device_count()
        print('number of available gpu',n_gpu) 
        queue = mp.Queue()
        q_processes = []

        num_partitions = len(datatensor_partitions)
        print('number of partitions:', num_partitions)

        # n_gpu = len(gpu_indices)
        config_map = (mconfig, options)
        
        if n_gpu>0:
            
            torch.multiprocessing.set_sharing_strategy('file_system')


            def spawn_q_process(q_process):
                print(">>> spawning hyperparam search process")
                q_process.start()

            def join_q_process(q_process):
                q_process.join()
                print("<<< joined hyperparam search process")

            def create_q_process(datatensor_partition, config_map, tr_val_dir, run_gpu_map, queue):
                return mp.Process(target=train_test_partition, args=(datatensor_partition,
                                                                                          config_map,
                                                                                          tr_val_dir,
                                                                                          run_gpu_map, 
                                                                                          queue))


            


            for q_i in range(min(n_gpu, num_partitions)):
                print('q_i:', q_i)
                datatensor_partition = {q_i:datatensor_partitions[q_i]}
                run_gpu_map = {q_i:run_gpu_maps[q_i]}
                q_process = create_q_process(datatensor_partition=datatensor_partition,
                                             config_map=config_map,
                                             tr_val_dir=tr_val_dir,
                                             run_gpu_map=run_gpu_map, 
                                             queue=queue)
                q_processes.append(q_process)
                spawn_q_process(q_process)

            spawned_processes = n_gpu

            print("*"*25)
            for q_i in range(num_partitions):
            #for q_i in range(1):
                join_q_process(q_processes[q_i])
                released_gpu_num = queue.get()
                print("released_gpu_num:", released_gpu_num)
                if(spawned_processes < num_partitions):
                    q_i_upd = q_i + n_gpu
                    print('q_i:', q_i, 'q_i_updated:', q_i_upd)
                    datatensor_partition = {q_i_upd:datatensor_partitions[q_i_upd]}

                    run_gpu_map = {q_i_upd:released_gpu_num}
                    q_process = create_q_process(datatensor_partition=datatensor_partition,
                                                 config_map=config_map,
                                                 tr_val_dir=tr_val_dir,
                                                 run_gpu_map=run_gpu_map, 
                                                 queue=queue)
                    q_processes.append(q_process)
                    spawn_q_process(q_process)
                    spawned_processes = spawned_processes + 1
       
        else:
            queue = mp.Queue()
            for i in range( num_partitions):
                datatensor_partition = {i:datatensor_partitions[i]}
            
                train_test_partition(datatensor_partition,config_map,tr_val_dir,run_gpu_map,queue)
            
            
            
            
        ## 5. peformance report                                                                             # get the performance results on training data
        num_runs = len(datatensor_partitions)
        train_performance = build_performance_dfs(tr_val_dir, num_runs, 'train', 'continuous')
        train_performance.to_csv(tr_val_dir+'/train_peformance.csv')                                                                          # get the performance results on validation dat
        build_performance_dfs(tr_val_dir, num_runs, 'validation', 'continuous')                                                                              
        test_peformance = build_performance_dfs(tr_val_dir, num_runs, 'test', 'continuous')
        test_peformance.to_csv(tr_val_dir+'/test_peformance.csv')  


        ## check best epoch for best performing models so far                                                                                ## check best epoch for best performing models so far
        for run_num in range(len(datatensor_partitions)):
        #for run_num in range(1):
            print('run_num:', run_num)
            print(ReaderWriter.read_data(os.path.join(f'{tr_val_dir}/train_val/run_{run_num}/model_statedict/best_epoch.pkl')))
            print()

###### function ends here###############################
    invivo = False
    #screen_name = 'Liver_SBApproach'
    #screen_name = ''
    #'ABE8e-NG', 'ABE8e-SpRY', 'ABE8e-SpCas9','ABEmax-NG',
    
    if not invivo:
        for editor_name in ['ABE8e-NG', 'ABE8e-SpRY', 'ABE8e-SpCas9','ABEmax-NG','ABEmax-SpCas9','ABEmax-SpRY']:
        #for editor_name in ['ABEmax-SpRY']:
            print('Running dataset:', editor_name)
            for input_type in ['protospacer_PAM']: #['protospacer','protospacer_PAM','protospacer_PAM_overhangs']:
                print('Running the model with input',input_type)
                main(editor_name, input_type,'version_0', False, '')
                print('The end')

    elif screen_name == 'Liver_LentiAAV':
        for editor_name in ['ABE8e-SpRY','ABEmax-SpRY']:
        #for editor_name in ['ABE8e-SpRY']:
            print('Running dataset:', editor_name)
            for input_type in ['protospacer_PAM']: #['protospacer','protospacer_PAM','protospacer_PAM_overhangs']:
                print('Running the model with input',input_type)
                main(editor_name, input_type,'version_0', invivo, screen_name )
                print('The end')
    elif screen_name == 'Liver_LentiLNP':
        for editor_name in ['ABE8e-NG', 'ABE8e-SpRY', 'ABE8e-SpCas9','ABEmax-SpRY']:
        #for editor_name in ['ABE8e-SpRY']:
            print('Running dataset:', editor_name)
            for input_type in ['protospacer_PAM']: #['protospacer','protospacer_PAM','protospacer_PAM_overhangs']:
                print('Running the model with input',input_type)
                main(editor_name, input_type,'version_0', invivo, screen_name )
                print('The end')
    elif screen_name == 'Liver_SBApproach':
        #for editor_name in ['ABE8e-SpRY']:
        for editor_name in ['ABEmax-SpRY']:
            print('Running dataset:', editor_name)
            for input_type in ['protospacer_PAM']: #['protospacer','protospacer_PAM','protospacer_PAM_overhangs']:
                print('Running the model with input',input_type)
                main(editor_name, input_type,'version_0' ,invivo, screen_name )
                print('The end')

    else:
        print('please sepecify the screening type')