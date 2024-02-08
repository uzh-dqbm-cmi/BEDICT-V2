import os
import shutil
import pickle
import string
import torch
import numpy as np
import scipy
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

class ContModelScore:
    def __init__(self, best_epoch_indx, spearman_corr, pearson_corr):
        self.best_epoch_indx = best_epoch_indx
        self.spearman_corr = spearman_corr
        self.pearson_corr = pearson_corr

    def __repr__(self):
        desc = " best_epoch_indx:{}\n spearman_corr:{}\n pearson_corr:{}\n" \
               "".format(self.best_epoch_indx, self.spearman_corr, self.pearson_corr)
        return desc

def get_performance_results(target_dir, num_runs, dsettype, ref_run, task_type='categ'):

    if task_type == 'categ':
        metric_names = ('auc', 'aupr', 'macro_f1')
    elif task_type == 'continuous':
        metric_names = ('spearman_corr', 'pearson_corr')
    elif task_type == 'ordinal':
        metric_names = ('mae', 'mse')

    num_metrics = len(metric_names)
    all_perf = {}
    perf_dict = [{} for i in range(num_metrics)]

    if dsettype in {'train', 'validation'} and ref_run is None:
        prefix = 'train_val'
    else:
        prefix = 'test'

    for run_num in range(num_runs):
        
        if ref_run is not None:
            runname = 'run_{}_{}'.format(ref_run, run_num)

        else:
            runname = 'run_{}'.format(run_num)

        run_dir = os.path.join(target_dir,
                               '{}'.format(prefix),
                               runname)

        score_file = os.path.join(run_dir, 'score_{}.pkl'.format(dsettype))
        # print(score_file)
        if os.path.isfile(score_file):
            mscore = ReaderWriter.read_data(score_file)

            if task_type == 'categ':
                perf_dict[0][runname] = mscore.auc
                perf_dict[1][runname] = mscore.aupr
                perf_dict[2][runname] = mscore.macro_f1

            elif task_type == 'continuous':
                perf_dict[0][runname] = mscore.spearman_corr
                perf_dict[1][runname] = mscore.pearson_corr

            elif task_type == 'ordinal':
                perf_dict[0][runname] = mscore.mae
                perf_dict[1][runname] = mscore.mse

    perf_df_lst = []
    for i in range(num_metrics):
        all_perf = perf_dict[i]
        all_perf_df = pd.DataFrame(all_perf, index=[metric_names[i]])
        median = all_perf_df.median(axis=1)
        mean = all_perf_df.mean(axis=1)
        stddev = all_perf_df.std(axis=1)
        all_perf_df['mean'] = mean
        all_perf_df['median'] = median
        all_perf_df['stddev'] = stddev
        perf_df_lst.append(all_perf_df.sort_values('mean', ascending=False))
    
    return pd.concat(perf_df_lst, axis=0)


def build_performance_dfs(target_dir, num_runs, dsettype, task_type, ref_run=None):
    target_dir = create_directory(target_dir)
    return get_performance_results(target_dir, num_runs, dsettype, ref_run, task_type=task_type)


def update_Adamoptimizer_lr_momentum_(optm, lr, momen):
    """in-place update for learning rate and momentum for Adam optimizer"""
    for pg in optm.param_groups:
        pg['lr'] = lr
        pg['betas'] = (momen, pg['betas'][-1])

def compute_lr_scheduler(l0, lmax, num_iter, annealing_percent):
    num_annealing_iter = np.floor(annealing_percent * num_iter)
    num_iter_upd = num_iter - num_annealing_iter
    
    x = [0, np.ceil(num_iter_upd/2.0), num_iter_upd, num_iter]
    y = [l0, lmax, l0, l0/100.0]
    tck = scipy.interpolate.splrep(x, y, k=1, s=0)
    
    xnew = np.arange(0, num_iter)
    lrates = scipy.interpolate.splev(xnew, tck, der=0)
    return lrates

def compute_momentum_scheduler(momen_0, momen_max, num_iter, annealing_percent):
    num_annealing_iter = np.floor(annealing_percent * num_iter)
    num_iter_upd = num_iter - num_annealing_iter
    
    x = [0, np.ceil(num_iter_upd/2.0), num_iter_upd, num_iter]
    y = [momen_max, momen_0, momen_max, momen_max]
    tck = scipy.interpolate.splrep(x, y, k=1, s=0)
    
    xnew = np.arange(0, num_iter)
    momentum_vals = scipy.interpolate.splev(xnew, tck, der=0)
    return momentum_vals

def build_classification_df(ids, true_class, pred_class, prob_scores, base_pos=None):

    prob_scores_dict = {}
    for i in range (prob_scores.shape[-1]):
        prob_scores_dict[f'prob_score_class{i}'] = prob_scores[:, i]

    if not base_pos:
        df_dict = {
            'id': ids,
            'true_class': true_class,
            'pred_class': pred_class
        }
    else:
        df_dict = {
            'id': ids,
            'base_pos':base_pos,
            'true_class': true_class,
            'pred_class': pred_class
        }
    df_dict.update(prob_scores_dict)
    predictions_df = pd.DataFrame(df_dict)
    predictions_df.set_index('id', inplace=True)
    return predictions_df

def build_probscores_df(ids, prob_scores, base_pos=None):

    prob_scores_dict = {}
    for i in range (prob_scores.shape[-1]):
        prob_scores_dict[f'prob_score_class{i}'] = prob_scores[:, i]

    if not base_pos:
        df_dict = {
            'id': ids
        }
    else:
        df_dict = {
            'id': ids,
            'base_pos':base_pos
        }
    df_dict.update(prob_scores_dict)
    predictions_df = pd.DataFrame(df_dict)
    # predictions_df.set_index('id', inplace=True)
    return predictions_df

def build_predictions_df(inpseqs_ids, outpseqs_ids, true_score, pred_score):
     
    # print(inpseqs_ids)
    seqid_inpseq_df = pd.DataFrame(inpseqs_ids)
    seqid_inpseq_df.columns = ['seq_id','Inp_seq']

    if true_score is not None:
        df_dict = {
            'Outp_seq':outpseqs_ids,
            'true_score': true_score,
            'pred_score': pred_score
        }
    else:
        df_dict = {
            'Outp_seq':outpseqs_ids,
            'pred_score': pred_score
        }     

    predictions_df = pd.concat([seqid_inpseq_df, pd.DataFrame(df_dict)], axis=1)
    return predictions_df

def dump_dict_content(dsettype_content_map, dsettypes, desc, wrk_dir):
    for dsettype in dsettypes:
        path = os.path.join(wrk_dir, '{}_{}.pkl'.format(desc, dsettype))
        ReaderWriter.dump_data(dsettype_content_map[dsettype], path)

class ReaderWriter(object):
    """class for dumping, reading and logging data"""
    def __init__(self):
        pass

    @staticmethod
    def dump_data(data, file_name, mode="wb"):
        """dump data by pickling
           Args:
               data: data to be pickled
               file_name: file path where data will be dumped
               mode: specify writing options i.e. binary or unicode
        """
        with open(file_name, mode) as f:
            #print('we are here saving',file_name )
            pickle.dump(data, f)

    @staticmethod
    def read_data(file_name, mode="rb"):
        """read dumped/pickled data
           Args:
               file_name: file path where data will be dumped
               mode: specify writing options i.e. binary or unicode
        """
        with open(file_name, mode) as f:
            data = pickle.load(f)
        return(data)

    @staticmethod
    def dump_tensor(data, file_name):
        """
        Dump a tensor using PyTorch's custom serialization. Enables re-loading the tensor on a specific gpu later.
        Args:
            data: Tensor
            file_name: file path where data will be dumped
        Returns:
        """
        torch.save(data, file_name)

    @staticmethod
    def read_tensor(file_name, device):
        """read dumped/pickled data
           Args:
               file_name: file path where data will be dumped
               device: the gpu to load the tensor on to
        """
        data = torch.load(file_name, map_location=device)
        return data

    @staticmethod
    def write_log(line, outfile, mode="a"):
        """write data to a file
           Args:
               line: string representing data to be written out
               outfile: file path where data will be written/logged
               mode: specify writing options i.e. append, write
        """
        with open(outfile, mode) as f:
            f.write(line)

    @staticmethod
    def read_log(file_name, mode="r"):
        """write data to a file
           Args:
               line: string representing data to be written out
               outfile: file path where data will be written/logged
               mode: specify writing options i.e. append, write
        """
        with open(file_name, mode) as f:
            for line in f:
                yield line


def create_directory(folder_name, directory="current"):
    """create directory/folder (if it does not exist) and returns the path of the directory
       Args:
           folder_name: string representing the name of the folder to be created
       Keyword Arguments:
           directory: string representing the directory where to create the folder
                      if `current` then the folder will be created in the current directory
    """
    if directory == "current":
        path_current_dir = os.path.dirname(__file__)  # __file__ refers to utilities.py
    else:
        path_current_dir = directory
    path_new_dir = os.path.join(path_current_dir, folder_name)
    if not os.path.exists(path_new_dir):
        os.makedirs(path_new_dir)
    return(path_new_dir)


def get_device(to_gpu, index=0):
    is_cuda = torch.cuda.is_available()
    if(is_cuda and to_gpu):
        target_device = 'cuda:{}'.format(index)
    else:
        target_device = 'cpu'
    return torch.device(target_device)


def report_available_cuda_devices():
    if(torch.cuda.is_available()):
        n_gpu = torch.cuda.device_count()
        print('number of GPUs available:', n_gpu)
        for i in range(n_gpu):
            print("cuda:{}, name:{}".format(i, torch.cuda.get_device_name(i)))
            device = torch.device('cuda', i)
            get_cuda_device_stats(device)
            print()
    else:
        print("no GPU devices available!!")

def get_cuda_device_stats(device):
    print('total memory available:', torch.cuda.get_device_properties(device).total_memory/(1024**3), 'GB')
    print('total memory allocated on device:', torch.cuda.memory_allocated(device)/(1024**3), 'GB')
    print('max memory allocated on device:', torch.cuda.max_memory_allocated(device)/(1024**3), 'GB')
    print('total memory cached on device:', torch.cuda.memory_reserved(device)/(1024**3), 'GB')
    print('max memory cached  on device:', torch.cuda.max_memory_reserved(device)/(1024**3), 'GB')


def compute_harmonic_mean(a, b):
    assert (a >= 0) & (b>=0), 'cannot compute the harmonic mean, one (or both) of the arguments is negative!'
    if a==0 and b==0:
        return 0.
    return 2*a*b/(a+b)
    
def compute_spearman_corr(pred_score, ref_score):
    # return scipy.stats.kendalltau(pred_score, ref_score)
    return scipy.stats.spearmanr(pred_score, ref_score)

def compute_pearson_corr(pred_score, ref_score):
    # return scipy.stats.kendalltau(pred_score, ref_score)
    return scipy.stats.pearsonr(pred_score, ref_score)

def restrict_grad_(mparams, mode, limit):
    """clamp/clip a gradient in-place
    """
    if(mode == 'clip_norm'):
        __, maxl = limit
        torch.nn.utils.clip_grad_norm_(mparams, maxl, norm_type=2) # l2 norm clipping
    elif(mode == 'clamp'): # case of clamping
        minl, maxl = limit
        for param in mparams:
            if param.grad is not None:
                param.grad.data.clamp_(minl, maxl)
def check_na(df):
    assert df.isna().any().sum() == 0


def perfmetric_report_cont(pred_score, ref_score, epoch_loss, epoch, outlog):  
    lsep = "\n"
    report = "Epoch: {}".format(epoch) + lsep
    spearman_corr, pvalue_spc = compute_spearman_corr(pred_score, ref_score)
    pearson_corr, pvalue_prc = compute_pearson_corr(pred_score, ref_score)
    report += f"Spearman correlation score:{spearman_corr}    pvalue:{pvalue_spc}" + lsep
    report += f"Pearson correlation score:{pearson_corr}    pvalue:{pvalue_prc}" + lsep    
    report += f"epoch average batch loss:{epoch_loss}" + lsep
    report += "-"*15 + lsep
    modelscore = ContModelScore(epoch, spearman_corr, pearson_corr)
    ReaderWriter.write_log(report, outlog)
    return modelscore


def plot_loss(epoch_loss_avgbatch, wrk_dir):
    dsettypes = epoch_loss_avgbatch.keys()
    for dsettype in dsettypes:
        plt.figure(figsize=(9, 6))
        plt.plot(epoch_loss_avgbatch[dsettype], 'r')
        plt.xlabel("number of epochs")
        plt.ylabel("negative loglikelihood cost")
        plt.legend(['epoch batch average loss'])
        plt.savefig(os.path.join(wrk_dir, os.path.join(dsettype + ".pdf")))
        plt.close()


def plot_xy(x, y, xlabel, ylabel, legend, fname, wrk_dir):
    plt.figure(figsize=(9, 6))
    plt.plot(x, y, 'r')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if legend:
        plt.legend([legend])
    plt.savefig(os.path.join(wrk_dir, os.path.join(fname + ".pdf")))
    plt.close()

def delete_directory(directory):
    if(os.path.isdir(directory)):
        shutil.rmtree(directory)
        
def transform_genseq_upper(df, columns):
    for colname in columns:
        df[colname] = df[colname].str.upper()
    return df

def visualize_motif_agg(top_motif_df, x_varname, model_name, color, t_class=1, topk=10, fig_dir=None):
    if t_class is not None:
        motif_df = top_motif_df.loc[top_motif_df['true_class']==t_class].copy()
    else:
        motif_df = top_motif_df.copy()
    unique_pos = set(motif_df['base_pos'].unique())
    if model_name == 'AID':
        target_pos = {2,3,4,5} # 3,4,5,6 1-indexing
    else:
        target_pos = {3,4,5,6,7} # {4,5,6,7,8} 1-indexing
    target_pos_lst = list(target_pos.intersection(unique_pos))
    fig, axs = plt.subplots(figsize=(9,11), 
                            nrows=len(target_pos_lst), 
                            constrained_layout=True) # we index these axes from 0 subscript\
    axs = axs.ravel()
    letters = list(string.ascii_uppercase)[:len(target_pos_lst)]

    panel_labels = [letters[i] for i in range(len(target_pos_lst))]
#     ymax = motif_df.loc[motif_df['base_pos'].isin(target_pos_lst), 'proportion'].max()
    for i, ax in enumerate(axs):
        ax.text(-0.05, 1.08, panel_labels[i], transform=ax.transAxes,
                fontsize=14, fontweight='bold', va='top', ha='right')
        pos = target_pos_lst[i]
        tmp_df = motif_df[motif_df['base_pos'] == pos]
        g = sns.barplot(x=x_varname, 
                        y='proportion', 
                        data=tmp_df, 
                        ax=ax, palette=[color]*topk)
        
#         g.legend(bbox_to_anchor=(0., -0.2), loc=2, borderaxespad=0.)
        ax.set_xlabel(f'3-mer motifs at base position {pos+1}', fontsize=12)
#         ax.set_title(model_order[i], fontsize=12)
        ax.set_ylabel('Percent', fontsize=12)
        ax.tick_params(labelsize=12)
#         ymax = tmp_df['proportion'].max()
#         ax.set_ylim([0, int(ymax+0.5)])
#         ax.set_xticklabels(rotation=55, fontsize=13)
    if fig_dir:
        fig.savefig(os.path.join(fig_dir,f'{model_name}_aggmotifs_class{t_class}.pdf'))

def highlight_attn_on_seq(df, indx, model_name, cmap = 'YlOrRd', fig_dir=None):
    fig, ax = plt.subplots(figsize=(11,3), 
                            nrows=1, 
                            constrained_layout=True) # we index these axes from 0 subscript\
    seq_id = df.iloc[indx]['id']
    panel_labels = [df['id']]
    attn_vars = [f'Attn{i}'for i in range(20)]
    letter_vars = [f'L{i}' for i in range(1,21)]
    # y_pred = df.iloc[indx]['true_class']
    prob = df.iloc[indx]['prob_score_class1']
    base_pos = df.iloc[indx]['base_pos'] + 1
    attn_scores  = df.iloc[indx][[f'Attn{i}'for i in range(20)]].values.astype(np.float).reshape(1,-1)
    max_score = df.iloc[indx][[f'Attn{i}'for i in range(20)]].max()
    print(max_score)
    base_letters =  df.iloc[indx][letter_vars].values.reshape(1,-1).tolist()
    print(base_letters)
    cbar_kws={'label': 'Attention score', 'orientation': 'horizontal'}
#     cmap='YlOrRd'
    g = sns.heatmap(attn_scores, cmap=cmap,annot=base_letters,fmt="",linewidths=.5, cbar_kws=cbar_kws, ax=ax)
    
#         g.legend(bbox_to_anchor=(0., -0.2), loc=2, borderaxespad=0.)
    
    ax.set_xticklabels(list(range(1,21)))
    ax.set(xlabel='Base position', ylabel='')
    ax.set_yticklabels([''])
    ax.text(20.5, 0.1 , 'base position = {}'.format(base_pos), bbox={'facecolor': '#4BB3FD', 'alpha': 0.2, 'pad': 10},
         fontsize=12)
    # ax.text(20.5, 0.55 ,r'$y_{pred}=$'+ '{}'.format(y_pred), bbox={'facecolor': 'green', 'alpha': 0.2, 'pad': 10},
    #          fontsize=12)
    ax.text(20.5, 0.65,r'Edit $probability=$'+ '{:.2f}'.format(prob), bbox={'facecolor': '#DD7596', 'alpha': 0.2, 'pad': 10},
             fontsize=12)
    ax.text(0.2, -0.2 ,r'$seqid={}$'.format(seq_id), bbox={'facecolor': 'grey', 'alpha': 0.2, 'pad': 10},
                 fontsize=12)
    if fig_dir:
        fig.savefig(os.path.join(fig_dir,f'{model_name}_seqattn_{seq_id}_basepos_{base_pos}.pdf'))
        plt.close()
    
    return ax