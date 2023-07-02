from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer
from matminer.featurizers.conversions import StrToComposition, CompositionToOxidComposition
from matminer.featurizers.composition import ElementProperty, OxidationStates
import numpy as np
import pandas as pd
import datetime
from sklearn.decomposition import PCA      
from sklearn.feature_selection import mutual_info_regression
import os
import csv

# from ax import *
# from ax.core.metric import Metric
# from ax.metrics.noisy_function import NoisyFunctionMetric
# from ax.service.utils.report_utils import exp_to_df
# from ax.runners.synthetic import SyntheticRunner
# import torch
# from ax.service.ax_client import AxClient
# from ax.service.utils.instantiation import ObjectiveProperties

from sklearn.model_selection import train_test_split
from utils.parser_ import get_args
from utils.utils_ import *
from plot import plt_true_vs_pred, plot_Xy_relation, plot_desc_distribution, plot_CycleTrain, plot_PCA_vis, plot_PCA_matminer_heatmap
# from train import cross_train_validation, cycle_train, elem1_train_and_plot
from train import *
from validate import *
from sklearn.model_selection import train_test_split


def Preprocessing(data_path, col_labels=None):
    df = get_data(data_path, col_labels) #
    if 'OER' in data_path:
        df = clean_df_OER(df)
        df = add_formula_col_OER(df)
    elif 'PCE' in data_path:
        df = clean_df_PCE(df)
        df = add_formula_col_PCE(df)
    else:
        raise ValueError('data_path should contain PCE or OER')
    df_cleaned = sort_clean_df(df)
    df = add_comp_col(df_cleaned)
    return df

def clean_df_OER(df_pec_data):
    df_pec_data['material'] = df_pec_data['material'].ffill()
    df_pec_data.dropna(axis=0, how='all', inplace=True)
    df_pec_data = df_pec_data.reset_index(drop=True)
    return df_pec_data

def clean_df_PCE(df_pec_data):
    # df_pec_data = df_pec_data.sort_values(['Sample'], ignore_index = True)
    df_pec_data.dropna(axis=0, how='all', inplace=True)
    df_pec_data = df_pec_data.reset_index(drop=True)
    return df_pec_data

def get_data(path, col_labels=None):
    '''read data'''
    if col_labels==None:
        df_pec_data = pd.read_excel(path)
    else:
        df_pec_data = pd.read_excel(path, header = 0)
        df_pec_data.columns = eval(col_labels)
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
            spt_proportions[i][j] = str(round(float(spt_proportions[i][j])*0.1, 4))
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

def select_train_elems():
    return X_inp_list[0][elem1_indx_random], y_outp_list[0][elem1_indx_random] #只对num_elements==3的数据进行训练 #(并且只用其中随机抽取的20个元素)


def filter_byMI(X, y, thr=0.0001):
    mi = mutual_info_regression(X, y)   #[616, 132] VS [616, 1]
    mi /= np.max(mi)                    #还是norm3
    idx = np.nonzero(mi>thr)
    # printc.BLUE(idx)        #32
    return idx[0]

def filter_byIdx(idx_union):
    def fn(X):
        assert len(X.shape) == 2
        return X[:, idx_union]
    return fn

class PCA_preprocessor:
    def __init__(self, 
                use_MI_filter,
                use_y_norm,
                is_MOBO,
                use_Xnorm_afterPCA,
                PCA_dim_select_method,
                PCA_dim,
                **kwargs):
        
        self.use_MI_filter = use_MI_filter
        self.use_y_norm = use_y_norm
        self.is_MOBO = is_MOBO
        self.use_Xnorm_afterPCA = use_Xnorm_afterPCA
        self.PCA_dim_select_method = PCA_dim_select_method
        self.PCA_dim = PCA_dim
        self.pre_fndict = None
        for key, value in kwargs.items():
            setattr(self, key, value)

    def MI_filtering_X(self, X, y):
        idx_list_beforeMerge = []
        for i in range(y.shape[1]):
            idx = filter_byMI(X, y[:, i])
            idx_list_beforeMerge.append(idx)
    
        idx_union = np.unique(np.concatenate(idx_list_beforeMerge))  # Find the union
        X = X[:, idx_union]
        print('X desc shape after MI filtering:', X.shape[1])
        return X, filter_byIdx(idx_union), idx_union

    def norm_y(self, y, fn_dict):
        for i in range(y.shape[1]):
            std_scaler_y = StandardScaler()
            y[:, i] = std_scaler_y.fit_transform(y[:, i].reshape(-1, 1))[:, -1]
            # assert '''y1 is y['slope relative to Ru']'''
            if i == 1 and self.is_MOBO:
                y[:, i] = - y[:, i]
            fn_dict['std_scaler_y'+str(i)] = std_scaler_y

        def fn_for_y(y_current, inverse_transform=False):
            assert y_current.shape[1] == y.shape[1]
            for i in range(y_current.shape[1]):
                if inverse_transform:
                    y_current[:, i] = fn_dict['std_scaler_y'+str(i)].inverse_transform(y_current[:, i].reshape(-1, 1))[:, -1]
                else:
                    y_current[:, i] = fn_dict['std_scaler_y'+str(i)].transform(y_current[:, i].reshape(-1, 1))[:, -1]
                if i == 1 and self.is_MOBO:
                    y_current[:, i] = - y_current[:, i]
            return y_current

        fn_dict['fn_for_y'] = fn_for_y
        return y

    def PCA_dim_select(self):
        if self.PCA_dim_select_method == 'auto':
            assert type(self.PCA_dim) == float
            selected_dim = self.PCA_dim
        elif self.PCA_dim_select_method == 'assigned':
            # assert type(PCA_dim) == int
            selected_dim = int(self.PCA_dim)
        return selected_dim

    def transform_fn_PCA(self, X_compo, y_pmax):
        fn_dict = {}
        methods_tobe_combined = []
        X = np.array(X_compo)   #shape=(160, 132)
        #X_log = np.log(X.astype('float'))
        y = np.array(y_pmax.reshape(-1, y_pmax.shape[1]))    #shape=(160, 2)
        
        # 1. MI filtering:
        if self.use_MI_filter:
            X, filter_method, idx_union = self.MI_filtering_X(X, y)
            methods_tobe_combined.append(filter_method)

        #2. X norm before PCA
        std_scalerX = StandardScaler()            
        X_norm = std_scalerX.fit_transform(X)     
        methods_tobe_combined.append(std_scalerX.transform)

        #3. PCA
        pca = PCA(n_components=self.PCA_dim_select())
        X_pca = pca.fit_transform(X_norm)
        methods_tobe_combined.append(pca.transform)
        # plot_PCA_vis(X_pca, y)
        # plot_PCA_matminer_heatmap(np.array(X_compo), X_pca, matminer_colnames)
        # plot_PCA_matminer_heatmap(np.array(X_compo)[:, idx_union], X_pca, matminer_colnames[idx_union])

        #4. X norm after PCA
        if self.use_Xnorm_afterPCA:
            std_scalerX_afpca = StandardScaler()
            X_pca = std_scalerX_afpca.fit_transform(X_pca)
            methods_tobe_combined.append(std_scalerX_afpca.transform)

        #5. y norm
        if self.use_y_norm:
            y = self.norm_y(y, fn_dict)

        fn_dict['fn_input'] = fn_comb(kwargs=methods_tobe_combined)
        self.pre_fndict = fn_dict
        printc.blue('PCA dimensions:', X_pca.shape[1])
        return X_pca, y
 

def Main(args, args_general, args_pre, args_BO):
    # 1. Import Data and Preprocessing 
    df = Preprocessing(args.data_path, args.col_labels)

    # 2 .Build composition descriptors (from `matminer`)
    descs = Add_extract_descriptors(df, args.use_concentration)
    X_compo = descs.values              # all descriptors
    y_pmax = df[args.model].values      #P（Eff max）

    # 3. Build regression model with composition descriptors 
    ## 3.1. norm and PCA input:
    # plot_Xy_relation(X_compo, y_pmax, descs.columns.values)

    kwargs_pre = vars(args_pre)
    preprocessor = PCA_preprocessor(**kwargs_pre)

    if args.split_ratio != 0:       
        X_init, X_remain, y_init, y_remain = train_test_split(X_compo, y_pmax,      
                                                              test_size=args.split_ratio)
    else:
        X_init, X_remain, y_init, y_remain = X_compo, None, y_pmax, None

    # plot_desc_distribution(X, screen_dims=8)
    ## 3.2 split data into train and test, and train model
    
    kwargs_BO = vars(args_BO)
    kwargs_BO.update({'PCA_preprocessor': preprocessor})
    kwargs_BO.update({'y_original_seq': y_pmax})
    if X_remain is not None and y_remain is not None:
        kwargs_BO.update({'X_remain': X_remain})
        kwargs_BO.update({'y_remain': y_remain})

    Model = MLModel(X_train=X_init, y_train=y_init,
                    save_file_instance=save_file_instance,
                    df_space_path=args.data_search_space,
                    **kwargs_BO)
    if args.is_SOBO:
        assert 'PCE' in args.data_path
        # cross_train_validation(X, y, args.Kfold, args.num_restarts,
        #                        args.ker_lengthscale_upper, args.ker_var_upper, save_file_instance)
        # elem1_train_and_plot(X, y, args.num_restarts, args.ker_lengthscale_upper,
        #                      args.ker_var_upper, save_file_instance,
        #                      args.split_ratio)
        Model.SOBO_one_batch()

    elif args.is_MOBO:
        assert 'OER' in args.data_path
        if args.only_use_elem2:
            X_init, y_init = X_init[1:, :], y_init[1:, :]
        # 1:
        # cross_train_validation(X, y, args.Kfold, args.num_restarts,
        #                        args.ker_lengthscale_upper, args.ker_var_upper, save_file_instance)
        # 2：
        # elem1_train_and_plot(X, y, args.num_restarts, args.ker_lengthscale_upper,
        #                      args.ker_var_upper, save_file_instance,
        #                      args.split_ratio)
        # 3：
        # Model.MOBO_one_batch()
        Model.MOBO_batches(mode="qEHVI")
        
        # log_values = cycle_train([X, y], [X_test, y_test], args.num_restarts, args.ker_lengthscale_upper, args.ker_var_upper)
        # plot_CycleTrain(y_list_descr, X, X_test)
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
        args, args_general, args_pre, args_BO = get_args()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args_BO.device = device
        
        become_deterministic(args.seed)
        
        save_file_instance = save_logfile(args.save_name, args.model_dir, args)
        next(save_file_instance)
        save_file_instance.send(('setup', 'args:', args))

        printc.blue( '\nsave_name:', args.save_name, '\n')

        Main(args, args_general, args_pre, args_BO)


    printc.green('--------------- Training finished ---------------')
    print('model_name:', args.save_name)
    # writer.close()


