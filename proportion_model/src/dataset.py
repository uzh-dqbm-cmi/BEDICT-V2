import os
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence as torch_pad_sequence
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit
from tqdm import tqdm
from .utilities import ContModelScore


class HaplotypeDataTensor(Dataset):

    def __init__(self, seqconfig):
        self.seqconfig = seqconfig
        
    # def _encode_to_one_hot(self, mask, n_dims=None):
    #     """ turn matrix with labels into one-hot encoding using the max number of classes detected"""
    #     original_mask_shape = mask.shape
    #     mask = mask.type(torch.LongTensor).view(-1, 1)
    #     if n_dims is None:
    #         n_dims = int(torch.max(mask)) + 1
    #     one_hot = torch.zeros(mask.shape[0], n_dims).scatter_(1, mask, 1)
    #     one_hot = one_hot.view(*original_mask_shape, -1)
    #     return one_hot

    def generate_tensor_from_df(self, proc_df, tb_cb_nucl, outcome_prop_col):
        # create the tensors we need
        # N is total number of input sequences
        print('Generating tensors using sequence config:\n', self.seqconfig)
        Xinp_enc = [] # tensor, (N x inp_sequence_len)
        Xinp_dec = [] # list of tensors, (N x num_haplotypes x outp_sequence_len)

        mask_inp_targetbase = [] # list of tensors, (N x num_haplotypes x outp_sequence_len)
        target_conv = [] # list of tensors, (N x num_haplotypes x outp_sequence_len)

        target_prob = [] # list of tensors, (N x num_haplotypes)

        indx_seqid_map = {} # dict, int_id:(seqid, target_seq)
        inpseq_outpseq_map = {} # dict([]), int_id:[outp_seq1, out_seq2, ....]

        seqconfig = self.seqconfig

        #seq_len = seqconfig.seq_len
        seq_len = len(proc_df['Inp_seq'][0])
        print('input seq length', seq_len)

        tb_nucl, cb_nucl = tb_cb_nucl # target base, conversion base (i.e. A->G for ABE base editor)
                                      #                                    C->T for CBE base editor
        num_outseq = [] # tensor (N, num_haplotypes)
        padding_value = 4 # to change this depending on the DNA letters/vocab we are using
        
        # output sequence will be from 0:end of editable window indx

        for gr_name, gr_df in tqdm(proc_df.groupby(by=['seq_id', 'Inp_seq'])):
            
            Xinp_enc.append(gr_df[[f'Inp_B{i}' for i in range(1,seq_len+1)]].values[0,:])
            Xinp_dec.append(gr_df[[f'Outp_B{i}' for i in range(1,seq_len+1)]].values[:,0:])
            num_outseq.append(Xinp_dec[-1].shape[0])
            
            mask_inp_targetbase.append(gr_df[[f'Inp_M{i}' for i in range(1,seq_len+1)]].values[:,0:])
            target_conv.append(gr_df[[f'conv{tb_nucl}{cb_nucl}_{i}' for i in range(1,seq_len+1)]].values[:,0:])

            if outcome_prop_col is not None:
                target_prob.append(gr_df[outcome_prop_col].values)
#             print(target_prob[-1])
            
            inpseq_id = len(indx_seqid_map)
            # {indx:(seq_id, inp_seq)}
            indx_seqid_map[inpseq_id] = gr_name
            inpseq_outpseq_map[inpseq_id] = gr_df['Outp_seq'].values.tolist()
        
        # placeholder in case we want to create a mask for the input and outcome sequence in the future
        mask_enc = None
        mask_enc_outcomeseq = None

        # tensorize
        print('--- tensorizing ---')
        device_cpu = torch.device('cpu')
        # (N, inp_sequence_len)
        Xinp_enc = np.array(Xinp_enc)
        print(Xinp_enc.shape)
        self.Xinp_enc = torch.from_numpy(Xinp_enc).long().to(device_cpu)
        #self.Xinp_enc = torch.tensor(Xinp_enc).long().to(device_cpu)
        # (N, 1, inp_sequence_len)
        # self.Xinp_enc = self.Xinp_enc.reshape(self.Xinp_enc.shape[0], 1, self.Xinp_enc.shape[1])
        
        # (N, num_haplotypes, sequence_len)
        self.Xinp_dec = torch_pad_sequence([torch.tensor(arr).long().to(device_cpu) for arr in Xinp_dec], 
                                           batch_first=True, 
                                           padding_value=padding_value)
        # (N, )
        self.num_outseq = torch.tensor(num_outseq).long().to(device_cpu)
        # (N x num_haplotypes x outp_sequence_len)
        self.mask_inp_targetbase = torch_pad_sequence([torch.tensor(arr).long().to(device_cpu) for arr in mask_inp_targetbase],
                                                      batch_first=True, 
                                                      padding_value=0)
        self.target_conv_onehot = torch_pad_sequence([torch.nn.functional.one_hot(torch.from_numpy(arr).long().to(device_cpu), num_classes=2) for arr in target_conv],
                                                     batch_first=True,
                                                     padding_value =0)
        if outcome_prop_col is not None:
            self.target_prob = torch_pad_sequence([torch.tensor(arr).float().to(device_cpu) for arr in target_prob],
                                                  batch_first=True,
                                                  padding_value=0.0)
        else:
            self.target_prob = None

        self.mask_enc = mask_enc
        self.mask_enc_outcomeseq = mask_enc_outcomeseq

        
        self.num_samples = len(self.Xinp_enc) # int, number of sequences
        self.indx_seqid_map = indx_seqid_map
        self.inpseq_outpseq_map = inpseq_outpseq_map
        print('--- end ---')

    # def hap_collate(self, batch):
    #     # pack batches in a list for now
    #     # to be used in dataloader object
    #     return [item for item in batch]

    def __getitem__(self, indx):

        if self.target_prob is None:
            return_target_prob = -1.
        else:
            return_target_prob = self.target_prob[indx]
       
        return(self.Xinp_enc[indx], 
               self.Xinp_dec[indx],
               self.num_outseq[indx],
               self.mask_inp_targetbase[indx],
               self.target_conv_onehot[indx],
               return_target_prob,
               indx)     

    def __len__(self):
        return(self.num_samples)

class PartitionDataTensor(Dataset):

    def __init__(self, dtensor, partition_ids, dsettype, run_num):
        self.dtensor = dtensor  # instance of :class:`HaplotypeDataTensor`
        self.partition_ids = partition_ids  # list of sequence indices
        self.dsettype = dsettype  # string, dataset type (i.e. train, validation, test)
        self.run_num = run_num  # int, run number
        self.num_samples = len(self.partition_ids[:])  # int, number of docs in the partition

    def __getitem__(self, indx):
        target_id = self.partition_ids[indx]
        return self.dtensor[target_id]

    def __len__(self):
        return(self.num_samples)

def print_data_example(elm):
    Xinp_enc, Xinp_dec, num_haplotype, mask_targetbase_enc, target_conv_onehot, target_prob, indx = elm 
    print('Xinp_enc:\n', Xinp_enc, 'shape:', Xinp_enc.shape)
    print('Xinp_dec:\n',Xinp_dec, 'shape:',Xinp_dec.shape)
    print('num of outcome sequences:\n', num_haplotype, 'shape:', num_haplotype.shape)
    print('mask_targetbase_enc:\n', mask_targetbase_enc, 'shape:', mask_targetbase_enc.shape)
    print('target_conv_onehot:\n', target_conv_onehot, 'shape:', target_conv_onehot.shape)
    if target_prob is not None:
        print('target_prob:\n',target_prob, 'shape:',target_prob.shape)
    else:
        print('target_prob:None')
    print('indx:', indx)
    # print('seqid:', seqid)

# def print_data_example(elm):
#     Xinp_enc, Xinp_dec, num_haplotype, mask_enc, mask_enc_outcomeseq, mask_targetbase_enc, target_prob, indx, seqid, __ = elm 
#     print('Xinp_enc:\n', Xinp_enc, 'shape:', Xinp_enc.shape)
#     print('Xinp_dec:\n',Xinp_dec, 'shape:',Xinp_dec.shape)
#     print('num of outcome sequences:\n', num_haplotype, 'shape:', num_haplotype.shape)
#     print('mask_targetbase_enc:\n', mask_targetbase_enc, 'shape:', mask_targetbase_enc.shape)
#     print('mask_enc:\n', mask_enc)
#     if mask_enc is not None:
#         print('shape:',mask_enc.shape)
#     print('mask_enc_outcomeseq:\n', mask_enc_outcomeseq)
#     if mask_enc_outcomeseq is not None:
#         print('shape:',mask_enc_outcomeseq.shape)

#     if target_prob is not None:
#         print('target_prob:\n',target_prob, 'shape:',target_prob.shape)
#     else:
#         print('target_prob:None')
#     print('indx:', indx)
#     print('seqid:', seqid)

def get_stratified_partitions(seqids, num_splits=5, val_set_portion=0.1, random_state=42):
    """Generate multi-run shuffle split using sequence ids
    Args:
        seqids: list, (integer list of sequence ids)
        num_splits: int, number of runs
        tr_set_portion: float, % of train set from train/val set portion
        random_state: int, to initiate random seed
    """
    
    sss_trval = ShuffleSplit(n_splits=num_splits, random_state=random_state, test_size=val_set_portion)
    data_partitions = {}
    train_index = seqids
    X = np.zeros(len(train_index))
    
    run_num = 0
    
    for tr_index, val_index in sss_trval.split(X):
    
        data_partitions[run_num] = {'train': tr_index,
                                    'validation': val_index}
        print("run_num:", run_num)
        print('train data:{}/{} = {}'.format(len(tr_index), len(X), len(tr_index)/len(X)))
        print()
        print('validation data:{}/{} = {}'.format(len(val_index), len(X), len(val_index)/len(X)))
        run_num += 1
        print("-"*25)
    return(data_partitions)

def validate_partitions(data_partitions, seqs_id, val_set_portion=0.1):
    if(not isinstance(seqs_id, set)):
        seqs_id = set(seqs_id)
    for run_num in data_partitions:
        print('run_num', run_num)
        tr_ids = data_partitions[run_num]['train']
        val_ids = data_partitions[run_num]['validation']
        tr_val = set(tr_ids).intersection(val_ids)
        valset_size = len(val_ids)
        num_seqs = len(tr_ids) + len(val_ids) 
        print('expected validation set size:', val_set_portion*num_seqs, '; actual validation set size:', len(val_ids))
        print('expected train set size:', (1-val_set_portion)*num_seqs, '; actual test set size:', len(tr_ids))
        print()
        assert np.abs(val_set_portion*num_seqs - len(val_ids)) <= 2  # valid difference range
        assert np.abs(((1-val_set_portion)*num_seqs)- len(tr_ids)) <= 2
        # assert there is no overlap among train, val and test partition within a fold
        assert len(tr_val) == 0

        s_union = set(tr_ids).union(val_ids)
        assert len(s_union) == num_seqs
    print('-'*25)
    print("passed intersection and overlap test (i.e. train, validation sets are not intersecting in each")

# similar to criscas.dataset but without stratification
# used for schwank experimental data :)
def get_random_partitions(df, num_splits=5, val_set_portion=0.5, random_state=42):
    """Generate multi-run shuffle split using sequence ids
    Args:
        seqids: list, (integer list of sequence ids)
        num_splits: int, number of runs
        tr_set_portion: float, % of train set from train/val set portion
        random_state: int, to initiate random seed
    """
    
    sss_trval = ShuffleSplit(n_splits=num_splits, random_state=random_state, test_size=val_set_portion)
    data_partitions = {}
    # df is dataframe with seq_id and seq_type :)
    train_index = df.loc[df['seq_type']==0].index.tolist()
    df_index = df.loc[df['seq_type']==1].index # index corresponding to val/test sequences

    X = np.zeros(len(df_index))
    
    run_num = 0
    
    for val_index, test_index in sss_trval.split(X):
    
        data_partitions[run_num] = {'train': train_index,
                                    'validation': df_index[val_index].tolist(),
                                    'test': df_index[test_index].tolist()}
        print("run_num:", run_num)
        print('train data:{}/{} = {}'.format(len(train_index), len(df), len(train_index)/len(df)))
        print()
        print('validation data:{}/{} = {}'.format(len(val_index), len(df), len(val_index)/len(df)))
        print('test data:{}/{} = {}'.format(len(test_index), len(df), len(test_index)/len(df)))

        run_num += 1
        print("-"*25)
    return(data_partitions)

def validate_random_partitions(data_partitions, seqs_id, val_set_portion=0.5):
    if(not isinstance(seqs_id, set)):
        seqs_id = set(seqs_id)
    for run_num in data_partitions:
        print('run_num', run_num)
        tr_ids = data_partitions[run_num]['train']
        val_ids = data_partitions[run_num]['validation']
        te_ids = data_partitions[run_num]['test']

        tr_val = set(tr_ids).intersection(val_ids)
        tr_te = set(tr_ids).intersection(te_ids)
        te_val = set(te_ids).intersection(val_ids)

        valset_size = len(te_ids) + len(val_ids)
        num_seqs = len(tr_ids) + len(val_ids) + len(te_ids)
        print('expected validation set size:', val_set_portion*valset_size, '; actual validation set size:', len(val_ids))
        print('expected test set size:', (1-val_set_portion)*valset_size, '; actual test set size:', len(te_ids))
        print()
        assert np.abs(val_set_portion*valset_size - len(val_ids)) <= 2  # valid difference range
        assert np.abs(((1-val_set_portion)*valset_size)- len(te_ids)) <= 2
        # assert there is no overlap among train, val and test partition within a fold
        for s in (tr_val, tr_te, te_val):
            assert len(s) == 0

        s_union = set(tr_ids).union(val_ids).union(te_ids)
        assert len(s_union) == num_seqs
    print('-'*25)
    print("passed intersection and overlap test (i.e. train, validation and test sets are not",
          "intersecting in each fold and the concatenation of test sets from each fold is",
          "equivalent to the whole dataset)")

    
# def report_label_distrib(labels):
#     classes, counts = np.unique(labels, return_counts=True)
#     norm_counts = counts/counts.sum()
#     for i, label in enumerate(classes):
#         print("class:", label, "norm count:", norm_counts[i])

def generate_partition_datatensor(dtensor, data_partitions):
    datatensor_partitions = {}
    for run_num in data_partitions:
        datatensor_partitions[run_num] = {}
        for dsettype in data_partitions[run_num]:
            target_ids = data_partitions[run_num][dsettype]
            datatensor_partition = PartitionDataTensor(dtensor, target_ids, dsettype, run_num)
            datatensor_partitions[run_num][dsettype] = datatensor_partition
    return(datatensor_partitions)


def hap_collate(batch):
    # pack batches in a list for now
    # to be used in dataloader object
    return [item for item in batch]

def construct_load_dataloaders(dataset_fold, dsettypes, score_type, config, wrk_dir):
    """construct dataloaders for the dataset for one run or fold
       Args:
            dataset_fold: dictionary,
                          example: {'train': <neural.dataset.PartitionDataTensor at 0x1cec95c96a0>,
                                    'validation': <neural.dataset.PartitionDataTensor at 0x1cec95c9208>,
                                    'test': <neural.dataset.PartitionDataTensor at 0x1cec95c9240>,
                                    'class_weights': tensor([0.6957, 1.7778])
                                   }
            score_type:  str, either {'categ', 'cont', 'ordinal'}
            dsettype: list, ['train', 'validation', 'test']
            config: dict, {'batch_size': int, 'num_workers': int}
            wrk_dir: string, folder path
    """

    # setup data loaders
    data_loaders = {}
    epoch_loss_avgbatch = {}
    flog_out = {}
    score_dict = {}

    for dsettype in dsettypes:
        if(dsettype == 'train'):
            shuffle = True
            sampler = None
        else:
            shuffle = False
            sampler = None
        
        # hap_collate = dataset_fold[dsettype].dtensor.hap_collate
        # print(dataset_fold[dsettype].dtensor.hap_collate)

        data_loaders[dsettype] = DataLoader(dataset_fold[dsettype],
                                            batch_size=config['batch_size'],
                                            shuffle=shuffle,
                                            num_workers=config['num_workers'],
                                            sampler=sampler)

        epoch_loss_avgbatch[dsettype] = []

        if(score_type == 'cont'):
            #  (best_epoch, spearman_correlation, pvalue)
            score_dict[dsettype] = ContModelScore(0, 0.0, 0.0)
        if(wrk_dir):
            flog_out[dsettype] = os.path.join(wrk_dir, dsettype + ".log")
        else:
            flog_out[dsettype] = None

    return (data_loaders, epoch_loss_avgbatch, score_dict, flog_out)