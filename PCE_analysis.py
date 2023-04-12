from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer
from matminer.featurizers.conversions import StrToComposition, CompositionToOxidComposition
from matminer.featurizers.composition import ElementProperty, OxidationStates
import numpy as np
import pandas as pd
import datetime
from sklearn.decomposition import PCA      
import os
import csv

# from ax import *
from ax.core.metric import Metric
from ax.metrics.noisy_function import NoisyFunctionMetric
from ax.service.utils.report_utils import exp_to_df
from ax.runners.synthetic import SyntheticRunner
import torch
# from ax.service.ax_client import AxClient
# from ax.service.utils.instantiation import ObjectiveProperties


from utils.parser_ import get_args
from utils.utils_ import *
from plot import plt_true_vs_pred, plot_Xy_relation, plot_desc_distribution, plot_CycleTrain
# from train import cross_train_validation, cycle_train, elem1_train_and_plot
from train import *
from sklearn.model_selection import train_test_split


def Preprocessing(path, col_labels, data_path):
    df = get_data(path, col_labels) #
    if 'OER' in data_path:
        df = add_formula_col_OER(df)
    elif 'PCE' in data_path:
        df = add_formula_col_PCE(df)
    else:
        raise ValueError('data_path should contain PCE or OER')
    df_cleaned = sort_clean_df(df)
    df = add_comp_col(df_cleaned)
    return df

def get_data(path, col_labels=None):
    '''read data'''
    if col_labels==None:
        df_pec_data = pd.read_excel(path)
    else:
        df_pec_data = pd.read_excel(path, header = 0)
        df_pec_data.columns = eval(col_labels)
    df_pec_data['material'] = df_pec_data['material'].ffill()
    # df_pec_data = df_pec_data.sort_values(['Sample'], ignore_index = True)
    df_pec_data.dropna(axis=0, how='all', inplace=True)
    df_pec_data = df_pec_data.reset_index(drop=True)
    return df_pec_data

def add_formula_col_PCE(daf):
    daf['formula'] = daf['Element']                           #为pd数据格式加了一列formula数据
    return daf

def add_formula_col_OER(daf):
    formula_list = []
    spt_proportions = [i.split(':') for i in daf['Elemental proportions'][1:]]          #remove the first row
    spt_materials = [i.split(':') for i in daf['material'][1:]]
    for i in range(len(spt_proportions)):
        formula = ''
        for j in range(len(spt_proportions[i])):
            spt_proportions[i][j] = str(round(int(spt_proportions[i][j])*0.1, 2))            
            formula += spt_materials[i][j] + spt_proportions[i][j]
        formula_list.append(formula)
    daf['formula'] = [daf['material'][0]] + formula_list   #为pd数据格式加了一列formula数据

    return daf


def sort_clean_df(daf):
    # df_pec_data_cleaned = df_pec_data_sorted        #不去除同组成的data
    # df_pec_data_cleaned = df_pec_data_sorted.drop_duplicates('formula', keep='last',ignore_index=True)  #pd格式下按相同的formula过滤，并只留下最后一个P最大的
    # df_pec_data_sorted = df_pec_data.sort_values(by=['formula', P_cal_form], ascending = True) #升序排列
    return daf

def add_comp_col(daf):
    df_pec = StrToComposition().featurize_dataframe(daf, "formula") #似乎是按前几列的colum label整理target col的数据，输出含有前几列的label，并作为新的col添加到pd中
    return df_pec

def Add_extract_descriptors(df_pec, use_concentration):
    ep_feat = ElementProperty.from_preset(preset_name="magpie")
    df_pec_magpie = ep_feat.featurize_dataframe(df_pec, col_id="composition")  #这两行是matminer的固定操作，用于加入描述符col
    _ = df_pec_magpie.shape[1] - 132           # changed param1 

    if use_concentration:
        #在df的列名中判断是否有Concentration这一列
        assert 'Concentration' in df_pec_magpie.columns
        desc = pd.concat([ df_pec_magpie['Concentration'], df_pec_magpie.iloc[:, _:] ], axis=1)
    else:
        desc = df_pec_magpie.iloc[:, _:]
    return desc

      
def norm_PCA_norm(X_compo, y_pmax, selected_method, n_dims, dataset_name, seed):
    std_scalerX = MinMaxScaler()            #用于进行col数据的归一化（norm1）到[0,1]之间，是按列进行norm（将数据的每一个属性值减去其最小值，然后除以其极差）
       #是一个用来对数据进行归一化和标准化的类norm2（利用var std等（那么在预测的时候， 也要对数据做同样的标准化处理，即也要用上面的scaler中的均值和方差来对预测时候的特征进行标准化
    std_scalerX_afpca = MinMaxScaler()

    X = np.array(X_compo)
    #X_log = np.log(X.astype('float'))   
    y = np.array(y_pmax.reshape(-1, y_pmax.shape[1]))   
    # plot_Xy_relation(X, y)

    pca = PCA(n_components=PCA_dim_select(selected_method, n_dims), random_state=seed)                  #使用：则会被降到5维
    X_norm = std_scalerX.fit_transform(X)             #对X进行归一化 norm3
    X_pca = pca.fit_transform(X_norm)                    #PCA之前是否需要StandardScaler norm一下（和原论文中顺序不同）
    X_pca_norm = std_scalerX_afpca.fit_transform(X_pca)
    # y_norm =  std_scalery.fit_transform(y)
    # fn_dict = {'fn_norm_bfPCA': std_scalerX, 'fn_pca': pca, 'fn_norm_afPCA': std_scalerX_afpca}
    fn_dict = {}
    fn_dict['fn_input'] = fn_comb([std_scalerX, pca, std_scalerX_afpca])

    if 'OER' in dataset_name:
        assert y.shape[1] == 2 
        std_scaler_y0 = StandardScaler()
        std_scaler_y1 = StandardScaler()
        y[:, 0] = std_scaler_y0.fit_transform(y[:, 0].reshape(-1, 1))[:, -1]
        y[:, 1] = std_scaler_y1.fit_transform(y[:, 1].reshape(-1, 1))[:, -1]
        assert '''y1 is y['slope relative to Ru']'''
        y[:, 1] = - y[:, 1]
        fn_dict['std_scaler_y0'] = std_scaler_y0
        fn_dict['std_scaler_y1'] = std_scaler_y1

    return X_pca_norm, y, fn_dict

def PCA_dim_select(selected_method, n_dims):
    if selected_method == 'auto':
        assert type(n_dims) == float
        selected_dim = n_dims
    elif selected_method == 'assigned':
        # assert type(n_dims) == int
        selected_dim = int(n_dims)
    return selected_dim

def select_train_elems():
    return X_inp_list[0][elem1_indx_random], y_outp_list[0][elem1_indx_random] #只对num_elements==3的数据进行训练 #(并且只用其中随机抽取的20个元素)
                                                

def Main(args):
    # 1. Import Data and Preprocessing 
    df = Preprocessing(args.data_path, args.col_labels, args.data_path)

    # 2 .Build composition descriptors (from `matminer`)
    descs = Add_extract_descriptors(df, args.use_concentration)
    X_compo = descs.values              # all descriptors
    y_pmax = df[args.model].values      #P（Eff max）

    # 3. Build regression model with composition descriptors 
    ## 3.1. norm and PCA input:
    # plot_Xy_relation(X_compo, y_pmax, descs.columns.values)
    X, y, fn_dict = norm_PCA_norm(X_compo, y_pmax, args.PCA_dim_select_method, args.PCA_dim, args.data_path, args.seed)
    printc.blue('PCA dimensions:', X.shape[1])
    # plot_desc_distribution(X, screen_dims=8)
    ## 3.2 split data into train and test, and train model
    if 'PCE' in args.data_path:
        # cross_train_validation(X, y, args.Kfold, args.num_restarts,
        #                        args.ker_lengthscale_upper, args.ker_var_upper, save_file_instance)
        elem1_train_and_plot(X, y, args.num_restarts, args.ker_lengthscale_upper, args.ker_var_upper, save_file_instance)
    elif 'OER' in args.data_path:
        # 1:
        # cross_train_validation(X, y, args.Kfold, args.num_restarts,
        #                        args.ker_lengthscale_upper, args.ker_var_upper, save_file_instance)
        # 2：
        # elem1_train_and_plot(X, y, args.num_restarts, args.ker_lengthscale_upper, args.ker_var_upper, save_file_instance)
        # 3:
        # X_list, y_list = select_train_elems()     #first try without split elem data
        if args.split_ratio != 0:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.split_ratio)
        else:
            X_train, y_train = X, y
        if args.only_use_elem2:
            X_train, y_train = X_train[1:, :], y_train[1:, :]

        MOBO_one_batch(X_train, y_train, args.num_restarts, 
                       args.ref_point, args.bs, args.num_raw_samples, save_file_instance, fn_dict)

        # log_values = cycle_train([X_train, y_train], [X_test, y_test], args.num_restarts, args.ker_lengthscale_upper, args.ker_var_upper)
        # plot_CycleTrain(y_list_descr, X_train, X_test)
        
    else:
        raise ValueError('Unknow dataset')


def save_logfile(save_name, model_dir, args):       #TODO: rewrite it to a class
    '''send me 'saveType, savename, value' and a value, and I will save it to a file '''
    os.makedirs(pjoin(model_dir, save_name), exist_ok=True)
    while True:
        saveType, savename, value = yield
        if saveType == 'setup':
            with open(pjoin(model_dir, save_name, 'setup.txt'), 'a') as f:
                print('{}:\n{}\n'.format(savename, str(value)), file=f)
                f.write('\n\n')
        elif saveType == 'result':
            if savename == '': savename = args.id
            with open(pjoin(model_dir, save_name, 'result.txt'), 'a') as f:
                print('{}:\n{}\n'.format(savename, str(value)), file=f)
                f.write('\n\n')
            write_dict_to_csv(value, pjoin(model_dir, save_name, savename+'.csv'))
            write_dict_to_csv(value, pjoin('tempdata', savename+'.csv'))
        elif saveType == 'model':
            if savename == '': savename = args.id
            paths = pjoin(model_dir, save_name, savename)    #
            value.save_model(paths)
        else:
            print('saveType must be args, result, or model')

if __name__ == '__main__':
    current_time = datetime.datetime.now()
    current_time = current_time.strftime("%m%d-%H_%M_%S")
    with measure_time():
        args = get_args()
        tkwargs = {
            "dtype": torch.double,
            "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        }

        # SMOKE_TEST = os.environ.get("SMOKE_TEST")
        become_deterministic(args.seed)
        
        save_file_instance = save_logfile(args.save_name, args.model_dir, args)
        next(save_file_instance)
        save_file_instance.send(('setup', 'args:', args))

        printc.blue( '\nsave_name:', args.save_name, '\n')

        Main(args)


    printc.green('--------------- Training finished ---------------')
    print('model_name:', args.save_name)
    # writer.close()


