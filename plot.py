from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_absolute_percentage_error as mape
from scipy.stats import spearmanr, pearsonr
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFE
from sklearn.feature_selection import mutual_info_regression

def y_inverse_transform(y_list):
    #y_orignal = np.exp(std_scalery.inverse_transform(y_list.reshape(-1,1)))
    #y_orignal = std_scalery.inverse_transform(y_list.reshape(-1,1))
    y_orignal = y_list.reshape(-1,1)
    return y_orignal 


def plot_CrossVal_avg():
    return None

def plot_CycleTrain(y_list_descr, X_train, X_test):
    fig, axes = plt.subplots(1, 1, figsize=(5.5*1.5, 4.5), sharey = False)
    fs = 20
    ax = axes
    ymax_acc_list = []
    for i in np.arange(len(y_list_descr)):
        ymax_acc = np.maximum.accumulate(y_list_descr[i])
        ax.plot(np.arange(len(y_list_descr[i]))-len(np.concatenate([X_train, X_test])),     #画max线。len(y_list_descr[i])应该是相同的
                ymax_acc, c = 'green', alpha = 0.3)
        ymax_acc_list.append(ymax_acc)
    ax.plot(np.arange(len(y_list_descr[0]))-len(np.concatenate([X_train, X_test])),
            np.mean(ymax_acc_list, axis = 0), '--', c = 'blue', alpha = 0.8)                 #repeat的10次的mean Pmax值
    #ax.scatter(np.arange(len(y_init))-len(np.concatenate([X_train, X_test])),y_init, c = 'green',alpha = 0.2)

    ax.plot(np.zeros(10),np.arange(10)/2-1, '--',c = 'black')
    ax.set_ylabel('Current Best Efficiency', fontsize = 20)
    ax.set_xlabel('Materials Composition', fontsize = 20)

    ax.set_ylim(0.2, 3.5)
    #axes[0].set_xlim(-1, 105)
    #axes[0].set_xticks(np.arange(0,105,10))
    ax.legend(fontsize = fs*0.7)
    ax.tick_params(direction='in', length=5, width=1, labelsize = fs*.8, grid_alpha = 0.5)
    ax.grid(True, linestyle='-.')

    plt.show()

def plt_true_vs_pred(y_true_list, y_pred_list, y_uncer_list, title_str_list, color_list,
                      only_value=False,
                      criterion='correlation'):
    plt.rcParams.update({'font.family': 'Palatino Linotype'})   #'fontname':'Times New Roman'
    fig, axes = plt.subplots(1, 2, figsize=(5.5*2, 4.5))
    fs = 20
    scores, cal_methods = [], []
    for i in np.arange(len(axes)):
        ## inverse transform
        y_true = y_true_list[i][:, -1]
        y_pred = y_pred_list[i][:, -1]
        y_uncer = np.sqrt(y_uncer_list[i][:,-1])

        if 'correlation' in criterion:
            spearman = spearmanr(y_true, y_pred) [0]
            pearson = pearsonr(y_true, y_pred) [0]
            scores.append([spearman, pearson])
            cal_methods.append(['sp_r', 'p_r'])
        elif'value' in criterion:
            rmse_value = np.sqrt(mse(y_true, y_pred))
            # mae_value = mae(y_true, y_pred)
            mape_value = mape(y_true, y_pred)
            scores.append([rmse_value, mape_value])
            cal_methods.append(['rmse', 'mape'])
        else:
            raise TypeError("Invalid input. OPT: 'correlation', 'value' ")

        if not only_value:
            lims1 = (0*0.9, 3*1.1)
            axes[i].scatter(y_true, y_pred, alpha = 0.3, c = color_list[i])
            axes[i].errorbar(y_true, y_pred, yerr = y_uncer, ms = 0,
                            ls = '', capsize = 2, alpha = 0.6,
                            color = 'gray', zorder = 0)
            axes[i].plot(lims1, lims1, 'k--', alpha=0.75, zorder=0)

            title = title_str_list[i] + " ({}={}, {}={})".format(cal_methods[i][0], np.round(scores[i][0],2),
                                                                 cal_methods[i][1], np.round(scores[i][1],2))
            axes[i].set_xlabel('Ground Truth', fontsize = fs)
            axes[i].set_ylabel('Prediction', fontsize = fs)
            axes[i].set_title(title, fontsize = fs)
            axes[i].set_xlim(0.5 , 1.5)
            axes[i].set_ylim(0.5 , 1.5)
            axes[i].tick_params(direction='in', length=5, width=1, labelsize = fs*.8, grid_alpha = 0.5)
            axes[i].grid(True, linestyle='-.')

    if not only_value:
        plt.subplots_adjust(wspace = 0.35)
        plt.show()

    def transfer_lst2dict(lstkey: list, lstvalue: list) -> dict:
        dict1 = {}
        for i in range(len(lstkey)):
            if i % 2 == 0:
                strfix = '-train'
            else:
                strfix = '-test'
            for j in range(len(lstkey[i])):
                dict1[lstkey[i][j] + strfix] = lstvalue[i][j]
        return dict1

    assert len(cal_methods) == len(scores)
    return transfer_lst2dict(cal_methods, scores)

def plot_Xy_relation(X, y, col_names):
    mi = mutual_info_regression(X, y)   #[616, 132] VS [616, 1]
    mi /= np.max(mi)                    #还是norm3
    # ax.hist(mi, orientation='horizontal', color='blue')        # hist is distribution figure, while bar is 柱形图
    # plt.bar(np.arange(len(mi)), mi)     #mi 为132项， 画图是为了看mat desc中哪项对于Pmax的关联性最大
    idx_sort = np.argsort(mi)
    mi_sort = mi[idx_sort][::-1]
    idx_sort = idx_sort[::-1]                     #reverse sort
    filtered_names = col_names[idx_sort][mi_sort > 0.6]
    y_lenlist = np.arange(len(filtered_names))
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.barh(y_lenlist, mi_sort[:len(y_lenlist)], align='center')
    # ax.set_yticks(y_len, labels=filtered_names)
    ax.set_yticks(y_lenlist)
    ax.set_yticklabels(filtered_names)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('MI correlation')
    # ax.set_title('')
    plt.show()
    pass

def plot_desc_distribution(X_pca, screen_dims=5):
    n_plots=screen_dims
    fs = 20
    # set the font name for a font family
    plt.rcParams.update({'font.family': 'Palatino Linotype'})
    fig, axes = plt.subplots(1, n_plots, figsize=(5.5*n_plots, 4))

    for i in np.arange(n_plots):
        axes[i].hist(np.array(X_pca[:,i]), bins =20,                                                #画hist是为了查看X某个维度(20个里面找一个）的数据分布情况
                    width = 0.04*(np.max(np.array(X_pca[:,i]))-np.min(np.array(X_pca[:,i]))),          #X_pca
                    alpha = 0.7)
        axes[i].tick_params(direction='in', length=5, width=1, labelsize = fs*.8, grid_alpha = 0.5)
        axes[i].grid(True, linestyle='-.')
        axes[i].set_xlabel('Values', fontsize = fs)
        axes[i].set_ylabel('Counts', fontsize = fs)
        axes[i].set_title('PCA X'+str(i+1), fontsize = fs)
    # set the spacing between subplots
    plt.subplots_adjust(left=0.1,
                        bottom=0.1, 
                        right=0.9, 
                        top=0.9, 
                        wspace=0.35, 
                        hspace=0.35)
    plt.show()