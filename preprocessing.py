from matminer.featurizers.conversions import StrToComposition, CompositionToOxidComposition
from matminer.featurizers.composition import ElementProperty, OxidationStates
import pandas as pd

def Preprocessing(data_path):
    df = get_data(data_path) #
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

def Add_extract_descriptors(df_pec, use_concentration=False):
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



