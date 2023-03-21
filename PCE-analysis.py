from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_absolute_percentage_error as mape
from scipy.stats import spearmanr, pearsonr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def y_inverse_transform(y_list):
    #y_orignal = np.exp(std_scalery.inverse_transform(y_list.reshape(-1,1)))
    #y_orignal = std_scalery.inverse_transform(y_list.reshape(-1,1))
    y_orignal = y_list.reshape(-1,1)
    return y_orignal

def plt_true_vs_pred(y_true_list, y_pred_list, y_uncer_list, title_str_list, color_list,
                      only_value=False,
                      criterion='correlation'):
    fig, axes = plt.subplots(1, 2, figsize=(5.5*2, 4.5))
    fs = 20
    scores, cal_methods = [], []
    for i in np.arange(len(axes)):
        ## inverse transform
        y_true = y_inverse_transform(y_true_list[i])
        y_pred = y_inverse_transform(y_pred_list[i])
        y_uncer = np.sqrt(y_uncer_list[i][:,-1])

        if 'correlation' in criterion:
            spearman = spearmanr(y_true, y_pred) [0]
            pearson = pearsonr(y_true, y_pred) [0]
            scores.append([spearman, pearson])
            cal_methods.append(['spearmanr', 'pearson'])
        elif'value' in criterion:
            rmse_value = np.sqrt(mse(y_true, y_pred))
            # mae_value = mae(y_true, y_pred)
            mape_value = mape(y_true, y_pred)
            scores.append([rmse_value, mape_value])
            cal_methods.append(['rmse_value', 'mape_value'])
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

    assert len(cal_methods) == len(scores)
    return cal_methods, scores

y_train, y_test = np.arange(1, 5), np.arange(2, 6)
y_pred_train, y_pred_test = np.arange(1, 5), np.arange(2, 6)
y_uncer_train, y_uncer_test = 0.1*np.arange(1, 5), 0.1*np.arange(2, 6)

plt_true_vs_pred([y_train, y_test],
                 [y_pred_train, y_pred_test],[y_uncer_train, y_uncer_test],
                 ['GP-Mat52 - Train','GP-Mat52 - Test'],
                 ['blue', 'darkorange'], criterion='correlation') 