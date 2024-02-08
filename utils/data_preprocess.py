## those are common to both absolute efficiency and proportion model
import pandas as pd
import re
from prettytable import PrettyTable
### drop entries that have nan or missing values
from sklearn.model_selection import train_test_split
import numpy as np
import random
import os
def transform_genseq_upper(df, columns):
    for colname in columns:
        df[colname] = df[colname].str.upper()
    return df

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

def get_train_test_val(seqid_info, run_num = 5, random_state = 42):
     # number of folds
    g_id = seqid_info['group_id'].unique()
    data_partitions = {}
    #torch.manual_seed(random_state)
    np.random.seed(random_state)
    for i in range(run_num):
        n_group = len(seqid_info['group_id'].unique())
        n_test = int(0.20 * n_group)
        test_gindex = random.sample(list(g_id) , n_test)
        train_gindex = [idx for idx in g_id if idx not in test_gindex]
        
        test_index = seqid_info[seqid_info['group_id'].isin(test_gindex)].index.tolist()
        train_index = seqid_info[seqid_info['group_id'].isin(train_gindex)].index.tolist()
        #print(test_gindex)
        test_index, val_index = train_test_split(test_index, test_size=0.5, random_state=random_state)
        
        tr_val = set(train_index).intersection(val_index)
        tr_te = set(train_index).intersection(test_index)
        te_val = set(test_index).intersection(val_index)
        print(len(tr_val))
        print(len(tr_te))
        print(len(te_val))
        data_partitions[i] = {'train': train_index,
                                        'validation': val_index,
                                        'test': test_index}
        print("run_num:", i)
        print('train data:{}/{} = {}'.format(len(train_index), len(seqid_info), len(train_index)/len(seqid_info)))
        print()
        print('validation data:{}/{} = {}'.format(len(val_index), len(seqid_info), len(val_index)/len(seqid_info)))
        print('test data:{}/{} = {}'.format(len(test_index), len(seqid_info), len(test_index)/len(seqid_info)))
    return data_partitions
    
    
def add_absolute_efficiency(df, pbar,prg_counter):
    gdf_clean = df.copy()
    gdf_clean['wild_type']=''
    gdf_clean['absolute_efficiency'] = ''
    for i in range(len(gdf_clean)):
        if len(gdf_clean) == 1:
            if gdf_clean['Reference'].iloc[i] == gdf_clean['Outcome'].iloc[i]:
                gdf_clean['wild_type'].iloc[i] = gdf_clean['Proportion'].iloc[i]
                gdf_clean['absolute_efficiency'].iloc[i] = 0
            else:
                gdf_clean['wild_type'].iloc[i] = 0
                gdf_clean['absolute_efficiency'].iloc[i] = gdf_clean['Proportion'].iloc[i]
          
        if len(gdf_clean) >1:
            #print(gdf_clean)
            #print(gdf_clean[gdf_clean['Reference'] == gdf_clean['Outcome']]['Proportion'])
            if len(gdf_clean[gdf_clean['Reference'].iloc[0] == gdf_clean['Outcome']])>0:
                gdf_clean['wild_type'].iloc[i] = gdf_clean[gdf_clean['Reference'] == gdf_clean['Outcome']]['Proportion'].iloc[0]
                gdf_clean['absolute_efficiency'].iloc[i] = gdf_clean[gdf_clean['Reference'] != gdf_clean['Outcome']]['Proportion'].sum()
            else:
                gdf_clean['wild_type'].iloc[i] = 0
                gdf_clean['absolute_efficiency'].iloc[i] = gdf_clean[gdf_clean['Reference'] != gdf_clean['Outcome']]['Proportion'].sum()  
                 
            
    prg_counter+=1
    pbar.update(prg_counter)
    return gdf_clean

def drop_nan_missing_values(df):
    N1 = len(df)
    df = df.dropna()
    df = df.reset_index(drop=True)
    index_pam = []
    index_left =[]
    index_right =[]
    for i in range(len(df)):
        if len(df.iloc[i]['PAM']) <4:
            print(df.iloc[i]['PAM'])
            index_pam.append(i)


        if len(df.iloc[i]['refseq_leftoverhang'])<5:

            index_left.append(i)

        if len(df.iloc[i]['refseq_rightoverhang'])<5:

            index_right.append(i)


    union_list = list(set(index_left).union(set(index_pam)).union(set(index_right)))
    print(len(union_list))  
    df = df.drop(union_list)
    df = df.reset_index(drop=True)
    print('number of filtered out instances:', N1 -len(df))
    return df


def compare_strings(string1, string2):
    differences = []
    for i in range(len(string1)):
        if string1[i] != string2[i]:
            differences.append(( i))
    return differences

def remove_elements(dictionary):
    for key, values in dictionary.items():
        dictionary[key] = [value for value in values if 1 <= value <= 11]
    return dictionary

from collections import defaultdict
def group_lists(dictionary):
    grouped_lists = defaultdict(list)
    
    for key, values in dictionary.items():
        grouped_lists[tuple(sorted(values))].append(key)
    
    return grouped_lists

def update_proportion_and_keep_first(group):
    group['Proportion'] = group['Proportion'].sum()
    return group.head(1)

def filter_edit_window(df, pbar):
    df_new = df.copy()
    original_edit = {}
    for i in range(len(df_new)):
        #editing_pos = compare_strings(target, df_new['Outcome'].iloc[i])
        original_edit[i]= compare_strings(df.iloc[0]['Reference'], df_new['Outcome'].iloc[i])
        #print(editing_pos)

    ## transfer the single edit outside of editing window to wild type
    edited_pos = original_edit.copy()
    for i in range(len(edited_pos)):
        if len(edited_pos[i])==1 and (edited_pos[i][0]>11 or edited_pos[i][0]<1):
            edited_pos[i] = []

    ## remove the edits outside the editing window
    considered_edit = remove_elements(edited_pos)
    ### groupd the similar edits after droping the outside edit window edits
    grouped_lists = group_lists(considered_edit)
    index = list(grouped_lists.values())
    ## defined grouping index list for the data frame
    for i in range(len(index)):
        index[i] = i*np.ones(len(index[i]))

    ## transform it to list
    one_dimensional_list = []
    [one_dimensional_list.extend(arr) for arr in index]
    integer_list = [int(element) for element in one_dimensional_list]

    ## groupd the df 
    df_new.loc[:, 'GroupIndex'] = integer_list
    grouped_df = df_new.groupby('GroupIndex')

    new_df = grouped_df.apply(update_proportion_and_keep_first)
    new_df = new_df.reset_index(drop=True)
    pbar.close()
    return new_df

def drop_wilde_type(df,pbar):
    df_new = df.copy()
    df_new = df_new.drop(df_new[df_new["Reference"] == df_new["Outcome"]].index)
    return df_new

def renormalize(df):
    df_new = df.copy()
    df_new['Proportion']= df['Proportion']/df['Proportion'].sum()
    return df_new

def get_char(seq):
    """split string int sequence of chars returned in pandas.Series"""
    chars = list(seq)
    return pd.Series(chars)

def validate_df(df):
    print('number of NA:', df.isna().any().sum())
    
class VizInpOutp_Haplotype(object):

    html_colors = {'blue':' #aed6f1',
                   'red':' #f5b7b1',
                   'green':' #a3e4d7',
                   'yellow':' #f9e79f',
                   'violet':'#d7bde2'}
    codes = {'A':'@', 'C':'!', 'T':'#', 'G':'_', 'conv':'~', 'prob':'%'}
    nucl_colrmap = {'A':'red',
                   'C':'yellow',
                   'T':'blue',
                   'G':'green',
                   'prob':'violet'}
    
    def __init__(self):
        pass
 

    @classmethod
    def viz_align_haplotype(clss, df, seqid, outcome_colname, seqconfig, conv_nl, predscore_thr=0., return_type='html'):
        """
        Args:
            df: processed dataframe using HaplotypeSeqProcessor.process_inp_outp_df
            seqid: string, sequence id column name
            outcome_colname: string or None, the ground truth outcome proportion column name
            seqconfig: instance of SeqProcessConfig class
            conv_nl: tuple of (target nucleotide, transition nucleotide)
            predscore_thr: float, probability threshold 
            return_type: string, default `html`
        
        """
        seq_len = seqconfig.seq_len
        seq_st, seq_end = seqconfig.seq_st, seqconfig.seq_end
        ewindow_st, ewindow_end = seqconfig.ewindow_st, seqconfig.ewindow_end
        offset_st, offset_end = seqconfig.offset_st, seqconfig.offset_end
        
        tb_nucl, cb_nucl = conv_nl
        codes = clss.codes
        
        tb = PrettyTable()
        tb.field_names = ['Desc.'] + [f'{i}' for i in range(1, seq_len+1)]
        
        cond = df['seq_id'] == seqid
        if 'pred_score' in df:
            cond_thr = df['pred_score'] >= predscore_thr
            cond_combined = (cond) & (cond_thr)
        else:
            cond_combined = cond
        df = df.loc[cond_combined].copy()
        # sort df by outcome probability
        if outcome_colname is not None:
            df.sort_values(by=[outcome_colname], ascending=False, inplace=True)
        else:
            df.sort_values(by=['pred_score'], ascending=False, inplace=True)

        # get the input 
        #print("seq_len:", seq_len)
  
        inp_nucl = df.iloc[0][[f'Inp_L{i}' for i in range(1,seq_len+1)]].values
        inp_str_lst = ['Input sequence'] + [f'{codes[nucl]}{nucl}' for nucl in inp_nucl]
        tb.add_row(inp_str_lst)

        n_rows = df.shape[0]
        # generate outcome (haplotype) rows
        for rcounter in range(n_rows):
            row = df.iloc[rcounter]
            outp_nucl = row[[f'Outp_L{i}' for i in range(1,seq_len+1)]].values
            if outcome_colname is not None:
                outp_str_lst = ['{}Output sequence\n Prob.={:.4f}'.format(codes['prob'], row[outcome_colname])]
            else:
                outp_str_lst = ['{}Output sequence'.format(codes['prob'])]

            cl_lst = []
            for pos, nucl in enumerate(outp_nucl):
                if row[f'conv{tb_nucl}{cb_nucl}_{pos+1}']:
                    cl_lst += [f"{codes['conv']}{nucl}"]
                else:
                    cl_lst += [f'{nucl}']
            outp_str_lst += cl_lst
            tb.add_row(outp_str_lst)

        pos_str_lst = ['Position numbering']+[str(elm) for elm in range(offset_st, offset_end+1)]
        tb.add_row(pos_str_lst)

        ewindow_str_lst = ['Editable window (*)'] + \
                      [' ' for elm in range(0, ewindow_st)]+ \
                      ['*' for elm in range(ewindow_st, ewindow_end+1)]+ \
                      [' ' for elm in range(ewindow_end+1, seq_len)]
        tb.add_row(ewindow_str_lst)

        seqwindow_str_lst = ['Sequence window (+)'] + \
                      [' ' for elm in range(0, seq_st)]+ \
                      ['+' for elm in range(seq_st, seq_end+1)]+ \
                      [' ' for elm in range(seq_end+1, seq_len)]
        tb.add_row(seqwindow_str_lst)

        if return_type == 'html':
            return clss._format_html_table(tb.get_html_string(), conv_nl)
        else: # default string
            return tb.get_string()
    @classmethod
    def _format_html_table(clss, html_str, conv_nl):
        tb_nucl, cb_nucl = conv_nl
        html_colors = clss.html_colors
        codes = clss.codes
        nucl_colrmap = clss.nucl_colrmap
        for nucl in codes:
            if nucl == 'conv':
                ctext = codes[nucl]
                color = html_colors[nucl_colrmap[cb_nucl]]
            else:
                ctext = codes[nucl]
                color = html_colors[nucl_colrmap[nucl]]
            html_str = re.sub(f'<td>{ctext}', '<td bgcolor="{}">'.format(color), html_str)
        return html_str

class HaplotypeVizFile():
    def __init__(self, resc_pth):
        # resc_pth: viz resources folder path
        # it contains 'header.txt',  'jupcellstyle.css', 'begin.txt', and 'end.txt'
        self.resc_pth = resc_pth
    def create(self, tablehtml, dest_pth, fname):
        resc_pth = self.resc_pth
        ls = []
        for ftname in ('header.txt',  'jupcellstyle.css', 'begin.txt'):
            with open(os.path.join(resc_pth, ftname), mode='r') as f:
                ls.extend(f.readlines())
        ls.append(tablehtml)
        with open(os.path.join(resc_pth, 'end.txt'), mode='r') as f:
            ls.extend(f.readlines())
        content = "".join(ls)
        with open(os.path.join(dest_pth, f'{fname}.html'), mode='w') as f:
            f.write(content)