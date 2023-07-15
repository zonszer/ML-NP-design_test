from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_absolute_percentage_error as mape
from scipy.stats import spearmanr, pearsonr
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFE
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler

from ax.utils.notebook.plotting import render, init_notebook_plotting
from ax.plot.pareto_utils import compute_posterior_pareto_frontier
from ax.plot.pareto_frontier import plot_pareto_frontier

from botorch.utils.sampling import draw_sobol_samples


def plot_frontier():
    init_notebook_plotting()

    return None

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
        scale_min = np.min(y_true) - np.mean(y_true)*0.32
        scale_max = np.max(y_true) + np.mean(y_true)*0.32

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
            lims1 = (0, scale_max*2)
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
            axes[i].set_xlim(max(scale_min, 0), scale_max)
            axes[i].set_ylim(max(scale_min, 0), scale_max)
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
    for i in range(len(col_names)):
        col_names[i] = col_names[i].replace("MagpieData", "")

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

def plot_PCA_vis(X, y):
    from sklearn.preprocessing import StandardScaler
    # Assuming X and y are already defined as numpy arrays or lists
    assert np.array(X).shape[1] == 3

    # Data preprocessing
    std_scalerX_afpca = StandardScaler()
    X_pca_norm = std_scalerX_afpca.fit_transform(X)

    # Step 4: Plot the 3D visualization
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(X_pca_norm[:, 0], X_pca_norm[:, 1], X_pca_norm[:, 2], c=y, cmap=plt.cm.get_cmap('copper'),
                    edgecolor='k', s=45)

    cbar = plt.colorbar(sc)
    cbar.set_label('Average Ratio over Control')
    ax.set_title("PCE Dataset Clustering Analysis After PCA Dimensionality Reduction")
    ax.set_xlabel("1st Eigenvector")
    ax.set_ylabel("2nd Eigenvector")
    ax.set_zlabel("3rd Eigenvector")
    ax.view_init(elev=30, azim=45)
    plt.show()

def plot_PCA_matminer_heatmap(X_matminer, X_pca, matminer_colnames):
    from sklearn.preprocessing import StandardScaler
    import seaborn as sns
    import pandas as pd

    for i in range(len(matminer_colnames)):
        matminer_colnames[i] = matminer_colnames[i].replace("MagpieData", "")
    # # Data preprocessing (Scale the data)
    # scaler = StandardScaler()
    # X_scaled = scaler.fit_transform(X)

    # Compute the correlations between the original features and PCA components
    correlations = np.corrcoef(X_matminer.T, X_pca.T)

    # Extract correlations between original features and PCA components
    correlations = correlations[:X_matminer.shape[1], -X_pca.shape[1]:]
    correlations_df = pd.DataFrame(correlations, index=matminer_colnames, columns=[f"PCA{i+1}" for i in range(X_pca.shape[1])])
    correlations_df.dropna(axis=0, how='any', inplace=True)
    correlations_df = correlations_df.sort_values('PCA1', ascending=False)

    # Visualize the heatmap
    # plt.figure(figsize=(5, 15))
    # fig, ax = plt.subplots()
    # im, cbar = sns.heatmap(correlations_df, annot=True, ax=ax,
    #                 cmap="YlGn", cbarlabel="Correlation")
    # fig.tight_layout()
    # plt.title("Correlations between original features and PCA components")
    # plt.show()

    # Visualize the heatmap
    plt.figure(figsize=(8, 55))
    sns.heatmap(correlations_df, annot=True, cmap="RdBu_r")
    plt.title("Correlations between original features and PCA components")
    plt.show()
 
#%% 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np


def get_regret(sub_df ):
    # Scalar (normalize) the target columns, y1 and y2, and add them 
    std_scaler_y1 = StandardScaler()
    std_scaler_y2 = StandardScaler()
    y1 = std_scaler_y1.fit_transform(sub_df["y1"].values.reshape(-1, 1))[:, -1]
    y2 = -std_scaler_y2.fit_transform(sub_df["y2"].values.reshape(-1, 1))[:, -1]

    sub_df["y1y2_hv"] = (y1 - y1.min()) * (y2 - y2.min())
    #Identify the global optimum
    glob_opt_hv, glob_opt_idx = sub_df["y1y2_hv"].max(), sub_df["y1y2_hv"].idxmax()    # idxmax provides the index of max value 

    #Calculate the regret for each row and add as a new column
    sub_df["regret"] = glob_opt_hv - sub_df["y1y2_hv"]
    sub_df["hv_rank"] = sub_df["y1y2_hv"].rank(ascending=False)
    return sub_df, glob_opt_idx

def plot_overallRegret(df, method, ax, iter_budget=5, with_stderr=True, color='blue'):
    # Filter dataframe to include only rows with method == method
    df = df.dropna()
    seeds = df['seed'].unique()
    trace_list = []
    glob_opt_idx_list = []
    overall_regret_list = []
    sum_hvRank_list = []
    for seed in seeds:
        sub_df = df[df['seed']==seed]
        sub_df = sub_df[sub_df["iter"] > 0]
        sub_df, glob_opt_idx = get_regret(sub_df) 
        sum_overall_regret = sub_df["regret"][sub_df["iter"] <= iter_budget].sum()
        sum_hvRank = sub_df["hv_rank"][sub_df["iter"] <= iter_budget].sum()

        overall_regret_list.append(sum_overall_regret)
        glob_opt_idx_list.append(sub_df.loc[glob_opt_idx])
        trace_list.append(sub_df['regret'][:50].values)
        sum_hvRank_list.append(sum_hvRank)

    # Plot the overall regret and other data here:
    trace_list = np.array(trace_list)
    mean = np.nanmean(trace_list, axis=0)
    stderr = np.nanstd(trace_list, axis=0, ddof=1)/ np.sqrt(np.shape(trace_list)[0]-1)

    x = np.arange(np.shape(trace_list)[1])+1

    # Plot the overall regret versus iter num
    ax.plot(x, mean, color=color, linewidth=1.5, 
            label=f"{method} glob_opt_idx" #{[int(glob_opt_idx_list[i]['original index in excel']) for i in range(len(glob_opt_idx_list))]},"
                f"found in iter:{[int(glob_opt_idx_list[i]['iter']) for i in range(len(glob_opt_idx_list))]}"
                f"\nSum overall_regret on first {int(iter_budget)} iters: {int(np.array(overall_regret_list).mean())}"
                f"\nSum hvRank on first {int(iter_budget)} iters: {int(np.array(sum_hvRank_list).mean())}")

    if with_stderr: 
        ax.fill_between(x, y1=mean-1.96*stderr, y2=mean+1.96*stderr, alpha=0.2, color=color)
        ax.plot(x, mean-1.96*stderr, color=color, linewidth=1, alpha=0.5)
        ax.plot(x, mean+1.96*stderr, color=color, linewidth=1, alpha=0.5)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Regret")
    ax.set_title("Bayesian Regret(all smaller better)")
    ax.legend(loc='lower right', fancybox=True, ncol=2)
    ax.grid(True)

#=============================
fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharex=False)
# Read the CSV file into the DataFrame
df = pd.read_csv('data/explored_sequence/MOBO_batches-qNEHVI.csv', index_col=0)
plot_overallRegret(df, method='qNEHVI', ax=axes[0], iter_budget=5, with_stderr=True, color='blue')

df = pd.read_csv('data/explored_sequence/MOBO_batches-random.csv', index_col=0)
plot_overallRegret(df, method='Random', ax=axes[1], iter_budget=5, with_stderr=True, color='blue')

plt.tight_layout()
