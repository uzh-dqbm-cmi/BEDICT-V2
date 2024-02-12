## prepare the data
import os
import random
import numpy as np
import pandas as pd
import torch
import sys
from tqdm import tqdm
current_pth = os.path.abspath('../')
sys.path.append(current_pth)
from utils.dataset import validate_random_partitions,generate_partition_datatensor
from utils.sequence_process import SeqProcessConfig, HaplotypeSeqProcessor,validate_df
from utils.data_preprocess import get_char, drop_nan_missing_values,drop_wilde_type,renormalize,add_absolute_efficiency
from utils.data_preprocess import get_train_test_val,transform_genseq_upper
from utils.utilities import ReaderWriter
from utils.data_preprocess import create_directory

data_dir = create_directory(os.path.join(current_pth, 'dataset'))

##1. read the data
input_type = 'protospacer_PAM'  #'protospacer','protospacer_PAM','protospacer_PAM_overhangs'
num_splits = 3 # number of folds

def process_save_data(editor_name,input_type , num_splits, exp_name, invivo ):
    
    if not invivo:
        if editor_name == 'ABEmax-SpRY':
            print('Loading dataset ', editor_name )
            df = pd.read_csv(os.path.join(data_dir,'HEK_BLDLib_10d_'f'{editor_name}_1_Seq1_ProportionTable_merged.txt'),
                         header=0, 
                         delimiter='\t')
        else:
            print('Loading dataset ', editor_name )
            df = pd.read_csv(os.path.join(data_dir,'HEK_BLDLib_10d_'f'{editor_name}_1_Merged_ProportionTable_merged.txt'),
                         header=0, 
                         delimiter='\t')

    else:
        df = pd.read_csv(os.path.join(data_dir,'merged_invivo',f'{exp_name}_'f'{editor_name}_2_Seq1_ProportionTable_merged.txt'),
                 header=0, 
                 delimiter='\t')
    ## unify the name of the columns    
    df.rename(columns={'rname':'ID'}, inplace=True)
    df.rename(columns={'refSeq':'Reference'}, inplace=True)
    df.rename(columns={'seq':'Outcome'}, inplace=True)
    df.rename(columns={'refseq_pam':'PAM'}, inplace=True)
    df['Proportion'] = df['genoCount']/df['allCount']
    N1 = len(df)
    df.duplicated(['Outcome', 'Reference']).sum()

    ## 2. clean the data
    df = drop_nan_missing_values(df)
    ##3. add two clomuns for absolute efficiency score and socre for wild type
    prg_counter=0
    dfg = df.groupby(by='Reference') 
    pbar = tqdm(total=dfg.ngroups)
    df_clean = dfg.apply(add_absolute_efficiency, pbar,prg_counter)
    pbar.close()
    df_clean.reset_index(inplace=True, drop=True)

    ##4. drop the sequences that has only wild type
    filtered_df = df_clean.drop(df_clean[df_clean['wild_type'] ==1].index)

    # define column names pointers to use throughout, tseq_col and outcome_col would be modified if we need to use diffferent input outputs
    id_col = 'ID' # sequnce id column name
    outcome_prop_col = ['Proportion','absolute_efficiency']  # propotion of edits column name  
    df = filtered_df.copy()


    if input_type == 'protospacer':
        extended_df = df
        tseq_col = 'Reference' # target sequence (wild-type) column name
        outcome_col = 'Outcome' # edit sequence (i.e. edited outcome sequence) column name  

    elif input_type == 'protospacer_PAM':
        protospacer_PAM =pd.DataFrame(df['Reference'] + df['PAM'])
        protospacer_PAM.columns = ['protospacer_PAM']
        output_PAM =pd.DataFrame(df['Outcome'] + df['PAM'])
        output_PAM.columns = ['Outcome_PAM']
        extended_df = pd.concat([df,protospacer_PAM, output_PAM],axis=1)
        tseq_col = 'protospacer_PAM' # target sequence (wild-type) column name
        outcome_col = 'Outcome_PAM' # edit sequence (i.e. edited outcome sequence) column name 


    elif input_type == 'protospacer_PAM_overhangs':

        df_left = pd.DataFrame({'refseq_leftoverhang': df['refseq_leftoverhang'].str[-k:]})
        df_right = pd.DataFrame({'refseq_rightoverhang': df['refseq_rightoverhang'].str[:k]})
        protospacer_PAM =pd.DataFrame(df_left['refseq_leftoverhang']+ df['Reference'] + df['PAM']+ df_right['refseq_rightoverhang'])
        protospacer_PAM.columns = ['protospacer_PAM_overhangs']
        output_PAM =pd.DataFrame(df_left['refseq_leftoverhang']+ df['Outcome'] + df['PAM']+df_right['refseq_rightoverhang'])
        output_PAM.columns = ['Outcome_PAM_overhangs']
        extended_df = pd.concat([df,protospacer_PAM, output_PAM],axis=1)
        tseq_col = 'protospacer_PAM_overhangs' # target sequence (wild-type) column name
        outcome_col = 'Outcome_PAM_overhangs' # edit sequence (i.e. edited outcome sequence) column name

    else:
        print('specify the input type')

    print(tseq_col)
    print(outcome_col)


    ## 5. prepare the data for proportion model
    gdf = extended_df.groupby(by='Reference',group_keys=False)
    pbar = tqdm(total=gdf.ngroups)
    proportion_df = gdf.apply(drop_wilde_type, pbar)
    proportion_df.reset_index(inplace=True, drop=True)
    ## renormalize 
    dfg = proportion_df.groupby(by='Reference',group_keys=False)
    normalized_proportion_df = dfg.apply(renormalize)
    normalized_proportion_df.reset_index(inplace=True, drop=True)
    proportion_df = normalized_proportion_df.copy()

    ## 6. prepare the data for absolute efficiency model
    absolute_efficiency_df=extended_df.groupby('Reference').first().reset_index()

    proportion_df = transform_genseq_upper(proportion_df, ['Outcome', 'Reference'])
    assert df.duplicated(['Outcome', 'Reference']).sum() == 0

    target_conv_nucl = {'ABEmax-NG':('A', 'G'),'ABE8e-NG':('A', 'G'), 'ABE8e-SpCas9':('A', 'G'), 'ABE8e-SpRY':('A', 'G'), 'ABEmax-SpCas9':('A','G'),'ABEmax-SpRY':('A', 'G'),}


    # define sequence processor config
    seqconfig = SeqProcessConfig(20, (1,20), (1,20), 1)
    seq_processor = HaplotypeSeqProcessor(editor_name, target_conv_nucl[editor_name], seqconfig)
    ## removing duplicates (in terms of output sequence)
    df = seq_processor.preprocess_df(proportion_df, ['ID', 'Reference'], 'Outcome')


    proc_df = seq_processor.process_inp_outp_df(proportion_df, id_col, tseq_col, outcome_col, outcome_prop_col[0])

    ## creat bystandar data tensor
    from utils.dataset import HaplotypeDataTensor
    hap_dtensor = HaplotypeDataTensor(seqconfig)
    hap_dtensor.generate_tensor_from_df(proc_df, target_conv_nucl[editor_name], outcome_prop_col[0])

    assert len(df['Reference'].unique()) == len(hap_dtensor.indx_seqid_map.values())

    ### create data partition
    seq_assign_df = pd.DataFrame(hap_dtensor.indx_seqid_map.values())
    seq_assign_df.columns = ['seq_id', 'Input_seq']

    temp = df.copy()
    temp.drop_duplicates(subset=['Reference'], ignore_index=True, inplace=True, keep='first')
    validate_df(temp)

    seqid_info  = pd.merge(left=seq_assign_df,
                           right=temp,
                           how='left', 
                           left_on=['seq_id'], 
                           right_on=['ID'])

    dpartitions = get_train_test_val(seqid_info,  run_num = num_splits, random_state=42)
    validate_random_partitions(dpartitions, seqid_info.index.tolist(), val_set_portion=0.5)
    dtensor_partitions = generate_partition_datatensor(hap_dtensor, dpartitions)

    ### generate data for the absolute efficiency model
    y = np.array(seqid_info['absolute_efficiency'])
    y = y.astype(np.float32)
    ID = seqid_info['ID']
    protospacer = seqid_info[tseq_col].apply(get_char)
    num_nucl = len(protospacer.columns)
    protospacer.columns = [f'B_encoded{i}' for  i in range(1, num_nucl+1)]
    protospacer.replace(['A', 'C', 'T', 'G'], [0,1,2,3], inplace=True)
    x_protospacer = np.array(protospacer)
    processed_data = [x_protospacer, ID ,y]

    fname = f'{editor_name}_proportions_encenc_two_model'
    if not invivo:
        target_dir = create_directory(os.path.join(data_dir, input_type, fname))
    else:
        target_dir = create_directory(os.path.join(data_dir,'invivo',exp_name, input_type, fname))

    # dump partitions and dtensor
    ReaderWriter.dump_data(dpartitions, os.path.join(target_dir, f'data_partitions.pkl'))
    ReaderWriter.dump_data(hap_dtensor, os.path.join(target_dir, f'hap_dtensor.torch'))
    ReaderWriter.dump_data(dtensor_partitions, os.path.join(target_dir, f'dtensor_partitions.torch'))
    ReaderWriter.dump_data(processed_data, os.path.join(target_dir, 'list_of_x_f_y.pkl'))
    print('finished processing', editor_name, input_type, 'number of partitions', num_splits)
    print('saving in', target_dir )
    


############specify the screening method and editor##########    
invivo=False
exp_name = ''
if not invivo:
    #for editor_name in ['ABEmax-SpRY','ABEmax-NG','ABEmax-SpCas9', 'ABE8e-NG', 'ABE8e-SpRY','ABE8e-SpCas9' ]:
    for editor_name in ['ABEmax-NG']:
        process_save_data(editor_name,input_type, num_splits,exp_name,invivo)
        
else:
    if exp_name == 'Liver_LentiAAV':
        for editor_name in ['ABEmax-SpRY', 'ABE8e-SpRY']:
            process_save_data(editor_name,input_type, num_splits,exp_name,invivo )
            
    if exp_name == 'Liver_SBApproach':
        for editor_name in ['ABE8e-SpRY']:
            process_save_data(editor_name,input_type, num_splits,exp_name,invivo)
    if exp_name == 'Liver_LentiLNP':
        for editor_name in ['ABEmax-SpRY','ABE8e-SpRY', 'ABE8e-NG','ABE8e-SpCas9' ]:
            process_save_data(editor_name,input_type,num_splits,exp_name,invivo)
        
