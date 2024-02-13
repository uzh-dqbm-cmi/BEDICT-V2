import os
import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
import argparse
from configparser import ConfigParser
import itertools
woking_path = '../'
print(woking_path )
import sys
sys.path.append(woking_path)
from utils.utilities import compute_spearman_corr,compute_pearson_corr



def get_overall_prob(pdf, edf):
    df_new = pdf.copy()
    ID=pdf.iloc[0]['seq_id']
    #print(ID)
    pred_y = edf[edf['id']== ID]['pred_class'].tolist()
    #print(pred_y)
    true_y = edf[edf['id']== ID]['true_class'].tolist()
    #print(true_y)
    x_pred = pdf[pdf['seq_id']==ID]['pred_score']
    x_pred = np.array(x_pred)
    x_true = pdf[pdf['seq_id']==ID]['true_score']
    x_true = np.array(x_true)
    df_new['overall_true'] = x_true*true_y[0]
    df_new['overall_pred'] = x_pred*pred_y[0]
    df_new=df_new.drop(df_new.columns[0], axis=1)
    ## adding back the wild type
    #if ID == 'NM_000520.6(HEXA)c.91CtoT(p.Gln31Ter)Position6':
        #print(pred_y)
    new_row = pd.Series([ID, pdf.iloc[0]['Inp_seq'], pdf.iloc[0]['Inp_seq'], 1, 1 ,1-true_y[0], 1-pred_y[0]], index=df_new.columns)
    df_new.loc[len(df_new)] = new_row
    return df_new


def peformance_report(data_name,saved_model, exp_name,num_runs, in_vivo, screen_type, transfer_learning,model_cell_type, data_cell_type):
    model_name = 'CNN'
    version = 2
    exp_name = 'protospacer_PAM'
    print('result for',exp_name)
    print(transfer_learning)
    if not transfer_learning:   
        if not in_vivo:
            proportion = os.path.join(woking_path , 'proportion_model','output', 'experiment_run_proportions_encenc_two_model', f'{data_name}_proportions_encenc_two_model' ,exp_name,'exp_version_0/test/')
            #proportion = os.path.join(woking_path , 'proportion_model','output', 'experiment_run_DeepBE', f'{data_name}_DeepBE' ,exp_name,'exp_version_0/test/')
            absolute_efficiency = os.path.join(woking_path , 'absolute_efficiency_model', 'output' ,f'{model_name}_v{version}',data_name, exp_name, 'test/')
            #absolute_efficiency = os.path.join(woking_path , 'absolute_efficiency_model', 'output' ,f'{model_name}_v{version}',f'{data_name}_DeepBE', exp_name, 'test/')
            print(proportion)
            print(absolute_efficiency)
        else:
            
            proportion = os.path.join(woking_path , 'proportion_model','output', 'experiment_run_proportions_encenc_two_model',screen_type,  f'{data_name}_proportions_encenc_two_model' ,exp_name,'exp_version_0/test/')
            absolute_efficiency = os.path.join(woking_path , 'absolute_efficiency_model', 'output' ,f'{model_name}_v{version}','invivo',screen_type,  data_name, exp_name, 'test/')
    else:
        
        if model_cell_type is not None:
            print('we are running transfer learning within invivo')
            proportion = os.path.join(woking_path , 'proportion_model','output', 'experiment_run_proportions_encenc_two_model','transfer_learning',f'model_{model_cell_type}_data_{data_cell_type}', data_name ,exp_name,'test/')
            print(proportion)
            absolute_efficiency = os.path.join(woking_path , 'absolute_efficiency_model', 'output' ,f'{model_name}_v{version}','transfer_learning',f'model_{model_cell_type}_data_{data_cell_type}', data_name,exp_name)
            print(absolute_efficiency)
        else:
            if data_name == saved_model:
                print('we are running transfer learning from vitro to vivo')
                proportion = os.path.join(woking_path , 'proportion_model','output', 'experiment_run_proportions_encenc_two_model','transfer_learning','vitro_model_vivo_data',screen_type, data_name ,exp_name,'test')
                print(proportion)
                absolute_efficiency = os.path.join(woking_path , 'absolute_efficiency_model', 'output' ,f'{model_name}_v{version}','transfer_learning','vitro_model_vivo_data', screen_type, data_name, exp_name)
                print(absolute_efficiency)
            else: 
                print('we are running transfer learning within invivo/invitro')
                proportion = os.path.join(woking_path , 'proportion_model','output', 'experiment_run_proportions_encenc_two_model','transfer_learning',f'model_{saved_model}_data_{data_name}',screen_type, data_name ,exp_name,'test/')
                print(proportion)
                absolute_efficiency = os.path.join(woking_path , 'absolute_efficiency_model', 'output' ,f'{model_name}_v{version}','transfer_learning',f'model_{saved_model}_data_{data_name}', screen_type,data_name, exp_name)
                print(absolute_efficiency)


    spearman_overall = []
    pearson_overall = []
    spearman_proportion = []
    pearson_proportion = []
    spearman_proportion_denorm = []
    pearson_proportion_denorm = []
    for i in range(num_runs):
        print('run number:', i)
        proportion_df = pd.read_csv(os.path.join(proportion,'run_'f'{i}', 'predictions_test.csv'), )
        efficiency_df = pd.read_csv(os.path.join(absolute_efficiency,'run_'f'{i}','predictions_test.csv'), )
        #print(len(efficiency_df))
        #print(efficiency_df)
        dfg = proportion_df.groupby('seq_id')
        final_df = dfg.apply(get_overall_prob,efficiency_df )
        final_df.reset_index(inplace=True, drop=True)
        non_wild_df = final_df.copy()
        non_wild_df = non_wild_df.drop(non_wild_df[non_wild_df['Inp_seq'] == non_wild_df['Outp_seq']].index)
        
        wild_df = final_df.copy()
        wild_df = wild_df[wild_df['Inp_seq'] == wild_df['Outp_seq']]
        
        spearman_corr, pvalue_spc = compute_spearman_corr(final_df['overall_pred'], final_df['overall_true'])
        pearson_corr, pvalue_prc = compute_pearson_corr(final_df['overall_pred'], final_df['overall_true'])
        spearman_overall.append(spearman_corr)
        pearson_overall.append(pearson_corr)
        
        spearman_proportion.append(compute_spearman_corr(proportion_df['pred_score'], proportion_df['true_score'])[0])
        pearson_proportion.append(compute_pearson_corr(proportion_df['pred_score'], proportion_df['true_score'])[0])
        
        
        spearman_proportion_denorm.append(compute_spearman_corr(non_wild_df['overall_pred'], non_wild_df['overall_true'])[0])
        pearson_proportion_denorm.append(compute_pearson_corr(non_wild_df['overall_pred'], non_wild_df['overall_true'])[0])
        
    spearman_overall.append(np.array(spearman_overall).mean(axis = 0))
    pearson_overall.append(np.array(pearson_overall).mean(axis = 0))
    spearman_overall.append(np.array(spearman_overall).std(axis = 0))
    pearson_overall.append(np.array(pearson_overall).std(axis = 0))
    print('for over all model',exp_name,data_name  )
    print('spearman correlation',spearman_overall)
    print('pearson correlation',pearson_overall)
    print('')
    
   
    spearman_proportion.append(np.array(spearman_proportion).mean(axis = 0))
    pearson_proportion.append(np.array(pearson_proportion).mean(axis = 0))
    spearman_proportion.append(np.array(spearman_proportion).std(axis = 0))
    pearson_proportion.append(np.array(pearson_proportion).std(axis = 0))
    print('proportion model on the renormalized space',exp_name,data_name  )
    print('spearman correlation',spearman_proportion)
    print('pearson correlation',pearson_proportion)
    print('')
    
    
    spearman_proportion_denorm.append(np.array(spearman_proportion_denorm).mean(axis = 0))
    pearson_proportion_denorm.append(np.array(pearson_proportion_denorm).mean(axis = 0))
    spearman_proportion_denorm.append(np.array(spearman_proportion_denorm).std(axis = 0))
    pearson_proportion_denorm.append(np.array(pearson_proportion_denorm).std(axis = 0))
    print('proportion model on the original space',exp_name,data_name  )
    print('spearman correlation',spearman_proportion_denorm)
    print('pearson correlation',pearson_proportion_denorm)
    print('')
    
    
    return final_df, proportion_df,efficiency_df, non_wild_df, wild_df

def scatter_plot(df, pred, true, data_name, over_all, in_vivo ,screen_name):
    spearman_corr, pvalue_spc = compute_spearman_corr(df[true], df[pred])
    pearson_corr, pvalue_prc = compute_pearson_corr(df[true], df[pred])
    print('spearmand correlation:', spearman_corr)
    print('pearson correlation:', pearson_corr)

    plt.scatter(df[true],df[pred],s=2, color='black')
    # Add a line representing the correlation
    #m, b = np.polyfit(df[true], df[pred], deg=1)
    #plt.axline(xy1=(0, b), slope=m, color='grey',label=f'Pearson correlation {round(spearman_corr,3)}')
    #plt.xlim(0, 1)
    #plt.ylim(0, 1)
    m, b = np.polyfit(df[true], df[pred], deg=1)
    
    # Clip the line so it doesn't go below zero on the y-axis
    x_values = np.array(plt.xlim())
    y_values = m * x_values + b
    y_values = np.maximum(y_values, 0)
    x_values = np.maximum(x_values, 0)
    #y_values = np.clip(y_values, 0, None)  # Clip to zero or positive values
    
    plt.plot(x_values, y_values, color='grey', label=f'Pearson correlation {round(spearman_corr, 3)}')
    custom_legend = [plt.Line2D([], [], color='grey', linestyle='-', label=f'Pearson correlation {round(pearson_corr,3)}'),
                     plt.Line2D([], [], color='grey', linestyle='', label=f'Spearman correlation {round(spearman_corr,3)}')]
    
    plt.legend(handles=custom_legend, loc="lower right")
    #legend.texts[0].set_text(f'Spearman correlation {round(pearson_corr,3)}')
    plt.title(f'{data_name} Testset (on {over_all} type)')
    plt.xlabel('True probability')
    plt.ylabel('Predicted probability')
    #plt.xlim(0, max(df[true].max(), df[pred].max())+0.05 )
    #plt.ylim(0, max(df[true].max(), df[pred].max())+0.05 )
    if in_vivo:
        folder_name = f'./in_vivo/{screen_name}'
    else:
        folder_name = f'./in_vitro'

    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        plt.savefig(f'{folder_name}/scatter_plot_{data_name}_{over_all}.svg')
    else: 
        plt.savefig(f'./in_vitro/scatter_plot_{data_name}_{over_all}.svg')



cmd_opt = argparse.ArgumentParser(description='Argparser for data')
cmd_opt.add_argument('-transfer_learning',type=str,default ='',help = ' transfer learning')
cmd_opt.add_argument('-screen_name',  type=str,default = '', help = '')
cmd_opt.add_argument('-model_cell_type',type=str,default = '',help = '')
cmd_opt.add_argument('-data_cell_type',type=str,default = '',help = '')
cmd_opt.add_argument('-data_name',type=str,default = '',help = '')
cmd_opt.add_argument('-model_name',type=str,default = '',help = 'model_name')
cmd_opt.add_argument('-in_vivo',type=str,default ='',help = '')
cmd_opt.add_argument('-config_file', type=str, default='config_file.ini', help='')


args, _ = cmd_opt.parse_known_args()


def main(args):

    config = ConfigParser()
    config.read(args.config_file)

    if 'Inference_overall' in config:
        params = config['Inference_overall']
        
        args.in_vivo = config.getboolean('Inference_overall', 'in_vivo')
        args.transfer_learning = config.getboolean('Inference_overall', 'transfer_learning')
        args.screen_name = params['screen_name']
        args.model_cell_type = params['model_cell_type']
        args.data_cell_type = params['data_cell_type']
        args.data_name = params['data_name']
        args.model_name = params['model_name']

        print('we are printing out args', args)

        transfer_learning = args.transfer_learning
        print('is it transfer learning:', transfer_learning)

        # Rest of the code...

    else:
        raise ValueError("The 'InferenceConfig' section is missing in the configuration file.")
    


    ### fixed parameters
    #model_name = 'CNN'
    #screen_name = ''
    #in_vivo = True
  
    num_runs = 3
    in_vivo = args.in_vivo
    transfer_learning = args.transfer_learning
    screen_name = args.screen_name
    model_cell_type = args.model_cell_type
    data_cell_type = args.data_cell_type
    data_name = args.data_name
    model_name =args.model_name

    if not in_vivo:
        final_df, proportion_df,efficiency_df,non_wild_df, wild_df = peformance_report(data_name, data_name, 'protospacer_PAM',num_runs, in_vivo,screen_name,transfer_learning, model_cell_type, data_cell_type)
    else:
        if screen_name == 'Liver_LentiAAV':
            if data_name in ['ABEmax-SpRY', 'ABE8e-SpRY']:
                final_df, proportion_df,efficiency_df,non_wild_df, wild_df= peformance_report(data_name, data_name, 'protospacer_PAM',num_runs, True,screen_name,transfer_learning, model_cell_type, data_cell_type)

        if screen_name == 'Liver_LentiLNP':
            if data_name in ['ABEmax-SpRY','ABE8e-NG', 'ABE8e-SpRY','ABE8e-SpCas9' ]:
                final_df, proportion_df,efficiency_df,non_wild_df, wild_df= peformance_report(data_name, data_name,'protospacer_PAM',num_runs, True,screen_name,transfer_learning,model_cell_type, data_cell_type)

        if screen_name == 'Liver_SBApproach':
            if data_name in ['ABEmax-SpRY' ]:
                final_df, proportion_df,efficiency_df,non_wild_df, wild_df= peformance_report(data_name,data_name, 'protospacer_PAM',num_runs, True,screen_name,transfer_learning,model_cell_type, data_cell_type)

        else:
            print('wrong cell line name')

    scatter_plot(final_df, 'overall_pred', 'overall_true', data_name, 'all', True,screen_name)
    scatter_plot(wild_df, 'overall_pred', 'overall_true', data_name, 'wild', in_vivo, screen_name)
    scatter_plot(proportion_df, 'true_score', 'pred_score', data_name, 'non-wild_renormalized',in_vivo, screen_name)



if __name__ == "__main__":
    args, _ = cmd_opt.parse_known_args()
    res_desc = main(args)
    
