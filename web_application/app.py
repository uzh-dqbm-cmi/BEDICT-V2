### this is the main inference file used for website

## preprocessing
from flask import Flask, render_template, request, send_file
#%load_ext autoreload
#%autoreload 2
import os
import random
import numpy as np
import pandas as pd
import torch
import sys
import io
import matplotlib as plt
import time
from tqdm import tqdm
current_pth = os.path.abspath('../')
#print(current_pth)
sys.path.append(current_pth)
from utils.sequence_process import SeqProcessConfig, HaplotypeSeqProcessor,validate_df
from utils.data_preprocess import  drop_nan_missing_values,drop_wilde_type,renormalize
from utils.data_preprocess import transform_genseq_upper
from utils.utilities import ReaderWriter, report_available_cuda_devices,get_device
from utils.data_preprocess import create_directory
from proportion_model.src.predict_model import BEDICT_EncEnc_HaplotypeModel

## for absolute efficiency model
working_path = '../absolute_efficiency_model'
sys.path.append(working_path)
from src.utils import one_hot_encode,compute_eval_results_df
from models.dataset import ProtospacerDataset
from models.data_process import generate_partition_datatensor,get_datatensor_partitions_for_inference,prepare_sample_data
from models.trainval_workflow import run_inference
import re
from flask import jsonify
def get_overall_prob(pdf, edf, num_runs):
    
    df_new = pdf.copy()
    #print(df_new.columns)
    wild_type = []
    for i in range(num_runs):
        ID=pdf.iloc[0]['seq_id']
        pred_y = edf[edf['seq_id']== ID][f'pred_class_run_{i}'].tolist()
        x_pred = pdf[pdf['seq_id']==ID][f'score_run_{i}']
        x_pred = np.array(x_pred)
        df_new[f'overall_pred_run_{i}'] = x_pred*pred_y[0]
        ## adding back the wild type
        wild_type.append(1- pred_y[0])
    
    list_1 = [ID, pdf.iloc[0]['Inp_seq'], pdf.iloc[0]['Inp_seq']]
    list_1.extend(wild_type)
    list_1.extend(wild_type)
    new_row = pd.Series(list_1, index=df_new.columns)
    #print(new_row)
    df_new.loc[len(df_new)] = new_row
    return df_new


def main(df,editor_name, in_vitro,lib_name, editing_window):
    
    #editor_name = 'ABE8e-NG'
    #in_vitro = True
    #lib_name =''
    
    input_type ='protospacer_PAM'
    num_runs = 3
    data_name = 'KM_SpCas9-ABEmax_v2'

    model_name= 'CNN'
    version = 2
    gpu_index = 0
    print(df)
    #data_pth = os.path.abspath('../../../crispr_private')
    #data_dir = create_directory(os.path.join(data_pth, 'dataset', 'final_dataset'))
    #df = pd.read_excel(os.path.join(data_dir,f'{data_name}.xlsx'),header=0)
    #df.rename(columns={'rname':'ID'}, inplace=True)
    #df.rename(columns={'refSeq':'Reference'}, inplace=True)
    #df.rename(columns={'seq':'Outcome'}, inplace=True)
    #df.rename(columns={'refseq_pam':'PAM'}, inplace=True)
    #df = transform_genseq_upper(df, ['refSeq_full','refseq_leftoverhang', 'Reference','PAM', 'refseq_rightoverhang' ])
    df = transform_genseq_upper(df, ['protospacer_PAM'])
    df['ID'] = df['protospacer_PAM']
    #protospacer_PAM =pd.DataFrame(df['Reference'] + df['PAM'])
    #protospacer_PAM.columns = ['protospacer_PAM']
    #extended_df = pd.concat([df,protospacer_PAM],axis=1)
    tseq_col = 'protospacer_PAM'
    #df = extended_df.copy()
    report_available_cuda_devices()
    device = get_device(True, 0)

    
    target_conv_nucl = {'ABEmax-NG':('A', 'G'),'ABE8e-NG':('A', 'G'), 'ABE8e-SpCas9':('A', 'G'), 'ABE8e-SpRY':('A', 'G'), 'ABEmax-SpCas9':('A','G'),'ABEmax-SpRY':('A', 'G'),}
    seqconfig = SeqProcessConfig(24, (1,24), (editing_window[0],editing_window[1]), 1)
    seq_processor = HaplotypeSeqProcessor(editor_name, target_conv_nucl[editor_name], seqconfig)
    bedict = BEDICT_EncEnc_HaplotypeModel(seq_processor, seqconfig, device)
    
    ## select the proportion model
    if in_vitro:
        model_path = current_pth +  f'/proportion_model/output/experiment_run_proportions_encenc_two_model/{editor_name}_proportions_encenc_two_model/'+ input_type + '/exp_version_0/train_val/'
        #proportion = []
    else:
        model_path = current_pth +  '/proportion_model/output/experiment_run_proportions_encenc_two_model/'+lib_name+ f'/{editor_name}_proportions_encenc_two_model/'+ input_type + '/exp_version_0/train_val/'

    print('we are loding the model from:',model_path)    
    proportion_df = pd.DataFrame(columns=['seq_id', 'Inp_seq', 'Outp_seq'])
    for i in range(num_runs):
    
        model_dir = model_path + f'run_{i}'
        start_test = time.time()
        #print(len(df))
        pred_df = bedict.predict_from_dataframe(df, ['ID','protospacer_PAM'] ,model_dir, outpseq_col=None, outcome_col=None, renormalize=True, batch_size=5)
        elapsed_time_test = time.time() - start_test
        #print(f"inference for {len(df)}: Time used = {elapsed_time_test:.4f} seconds")
        proportion_df['seq_id']=  pred_df['seq_id']
        proportion_df['Inp_seq']=  pred_df['Inp_seq']   
        proportion_df['Outp_seq']=  pred_df['Outp_seq'] 
        proportion_df[f'score_run_{i}']=  pred_df['pred_score'] 

    pd.set_option('display.float_format', '{:.6f}'.format)

    
    #absolute efficiency model prediciton
    x_protospacer, y,ID, x_non_protos_f = prepare_sample_data(df,input_type,'proportion')
    if in_vitro:
        model_path = os.path.join(current_pth, 'absolute_efficiency_model',
                                          'output', 
                                          f'{model_name}_v{version}',editor_name, 
                                         input_type)
    else:
        model_path = os.path.join(current_pth, 'absolute_efficiency_model',
                                          'output', 
                                          f'{model_name}_v{version}','invivo', lib_name, editor_name, 
                                         input_type)

    print('we are loading the model from:',model_path)
    data_partitions = {}
    for num_run in range(num_runs):
        data_partitions[num_run] = {'train_index': None, 'test_index':np.arange(x_protospacer.shape[0]) }

    if model_name == 'CNN':
        proc_x_protospacer = one_hot_encode(x_protospacer)
        proc_x_protospacer = proc_x_protospacer.reshape(proc_x_protospacer.shape[0], -1)

    dpartitions, datatensor_partitions = get_datatensor_partitions_for_inference(data_partitions,
                                                                   model_name,
                                                                   proc_x_protospacer,
                                                                   y,ID,
                                                                   x_non_protos_f,
                                                                   fdtype=torch.float32,
                                                                   train_size=0.0,
                                                                   random_state=42)

    
    train_val_path = os.path.join(model_path, 'train_val')
    test_path = os.path.join(model_path, 'sample_test', data_name)
    print(f'Running model: {model_name}, exp_name: {input_type}, saved at {train_val_path}')
    a, b = run_inference(datatensor_partitions, 
                                 train_val_path, 
                                 test_path, 
                                 gpu_index, 
                                 to_gpu=True)
                                 #num_runs=num_runs)
    print('='*15)
    
    #efficiency_df = pd.DataFrame(columns=['seq_id', 'Inp_seq', 'Outp_seq', 'true_class'])
    for i in range(num_runs):
        if i == 0:
            efficiency_df = pd.read_csv(os.path.join(test_path,'run_'f'{i}', 'predictions_test.csv'), )
            efficiency_df.rename(columns={'pred_class':f'pred_class_run_{i}'}, inplace=True)
            efficiency_df.rename(columns={'id':'seq_id'}, inplace=True)


        else:
            temp_df= pd.read_csv(os.path.join(test_path,'run_'f'{i}', 'predictions_test.csv'), )
            pred = temp_df['pred_class']
            efficiency_df[f'pred_class_run_{i}'] = pred


    
    dfg = proportion_df.groupby('seq_id')
    final_df = dfg.apply(get_overall_prob, efficiency_df,num_runs)
    final_df.reset_index(inplace=True, drop=True)
    columns_to_remove = [f'score_run_{i}' for i in range(num_runs)]
    final_df = final_df.drop(columns=columns_to_remove)
    selected_columns = [f'overall_pred_run_{i}' for i in range(num_runs)]
    final_df['average_overal_pred'] = final_df[selected_columns].mean(axis=1)
    final_df['std_overal_prd'] = final_df[selected_columns].std(axis=1)


    ranked_df = final_df.copy()
    ranked_df['rank'] = ranked_df.groupby('seq_id')['average_overal_pred'].rank(ascending=False)
    ranked_df['rank'] = ranked_df['rank'].astype(int)
    df_sorted = ranked_df.sort_values(by=['seq_id', 'rank'])
    ## rank the output
    
    ### display the result
    tseqids = df_sorted['seq_id'][0]
    tseqids = [tseqids]
    df_small = df_sorted[df_sorted['seq_id']==tseqids[0]].copy()
    
    res_html = bedict.visualize_haplotype(df= df_small, seqsids_lst= tseqids, 
                                            inpseq_cols = ['seq_id','Inp_seq'], 
                                            outpseq_col = 'Outp_seq', 
                                            outcome_col='average_overal_pred', 
                                            predscore_thr=0.)
    
    return final_df, df_sorted, res_html


app = Flask(__name__)
## models load

@app.route('/')
def home():
    # Assume editors is a list of available editor names
    #'ABEmax-SpRY', 'ABEmax-SpCas9', 'ABEmax-NG', 'ABE8e-NG', 'ABE8e-SpRY', 'ABE8e-SpCas9'
    editors = ['ABEmax-SpRY', 'ABEmax-SpCas9', 'ABEmax-NG', 'ABE8e-NG', 'ABE8e-SpRY', 'ABE8e-SpCas9']  # Add more as needed
    return render_template('index.html', editors=editors)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/help')
def help():
    return render_template('help.html')

@app.route("/", methods=["POST"])
def TnpB():
    print("Received POST request")
    data = request.json.get('input_data')
    #editing_starts = request.json.get('input1')
    #print('editng start', editing_starts)
    #editing_end = request.json.get('input2')
    #print('editing edns', editing_end)
    editing_window = [1, 20]
    
    editor_name = request.json.get('editor_name')  # Add this line to get the editor name
    #in_vitro = request.json.get('screening_method')
    in_vitro = request.json.get('prediction_type')  #prediction_type
    print('is it in', in_vitro)
    cell_type = request.json.get('cell_type') 
    print('cell type', cell_type)    
    print('editor name', editor_name)
    if data is None or editor_name is None:
        return jsonify({'error': 'No input data or editor name provided'})
    strings = [seq.strip() for seq in data.split('\n') if seq.strip()]
    print(strings)

    not_dna_pattern = re.compile(r"[^ACGTacgt]")
    linecounter = 0
    for line in strings:
        linecounter += 1
        linelength = len(line)
        if  linelength != 24:
            print("wrong length")
            error_payload = {"error": "Input is wrong - wrong line length - should be 24 per line, it is: " + str(linelength) + " in line "+str(linecounter)}
            return jsonify(error_payload)
        found_not_dna_pattern = re.search(not_dna_pattern,line)
        if found_not_dna_pattern != None:
            error_payload = {"error": "One line of input contains wrong characters in line "+str(linecounter)}
            return jsonify(error_payload)
    df = pd.DataFrame({'protospacer_PAM': strings})
    print('the input data frame is', df)
    if in_vitro == 'InVitro':
        in_vitro = True
    else:
        in_vitro = False
      
    final_df, df_sorted,res_html = main(df, editor_name, in_vitro, cell_type, editing_window)
    print('printing html')
    print(res_html)
    print('printing html end')


    predictions_list = []
    for i, (ref_seq, output_seq, score) in enumerate(zip(df_sorted['Inp_seq'], df_sorted['Outp_seq'], df_sorted['average_overal_pred'])):
        prediction_dict = {
            'index': i,
            'Ref_seq':ref_seq, 
            'Output_seq': output_seq,
            'score': score
        }
        predictions_list.append(prediction_dict)
        print(prediction_dict)

    # Create a downloadable Excel file
    # excel_output = io.BytesIO()
    # df_sorted.to_excel(excel_output, index=False, sheet_name='predictions')
    # excel_output.seek(0)

    # # Save the Excel file to the server
    # excel_file_path = 'predictions.xlsx'
    # excel_output.seek(0)
    
    # with open(excel_file_path, 'wb') as f:
    #      f.write(excel_output.read())

    # # Return the Excel file as a downloadable attachment
    # return send_file(
    #     excel_file_path,
    #     mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    #     as_attachment=True,
    #     download_name='predictions.xlsx'
    # )

    #return jsonify(predictions_list)
        return res_html
    # return render_template("index.html", res_html=res_html)

if __name__== "__main__":
    app.run(debug=True, port=8888)

