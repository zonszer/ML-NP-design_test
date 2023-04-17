from sklearn.model_selection import KFold
import GPy
from GPy.models import GPRegression
from scipy.stats import spearmanr, pearsonr
import numpy as np
from utils.utils_ import *
from plot import plot_CrossVal_avg
from sklearn.model_selection import train_test_split
from plot import plt_true_vs_pred, plot_Xy_relation, plot_desc_distribution, plot_CycleTrain
import pandas as pd

import torch
from botorch import fit_gpytorch_model
from botorch.acquisition.monte_carlo import qExpectedImprovement, qNoisyExpectedImprovement
from botorch.sampling import SobolQMCNormalSampler
from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.utils.multi_objective.hypervolume import Hypervolume

from botorch.optim.optimize import optimize_acqf, optimize_acqf_list
from botorch.utils.multi_objective.box_decompositions import NondominatedPartitioning
from botorch.acquisition.multi_objective.monte_carlo import qExpectedHypervolumeImprovement
from botorch.utils.sampling import sample_simplex
from botorch.optim import optimize_acqf_discrete

from botorch.models.gp_regression import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch.utils.transforms import unnormalize
from botorch.sampling.normal import NormalMCSampler

tkwargs = {
    # "dtype": torch.double,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}


def generate_bounds(X, y, dim_X, num_objectives, scale=(0, 1)):
    bounds = np.zeros((2, dim_X))
    for i in range(dim_X):
        bounds[0][i] = min(scale)  # min of bound
        bounds[1][i] = max(scale)  # max of bound
    return bounds


def generate_initial_data(X, y, ref_point):
    '''generate training data'''
    train_x = torch.DoubleTensor(X)
    train_obj = torch.DoubleTensor(y)
    if ref_point is None:
        rp = torch.min(train_obj, axis=0)[0] - torch.abs(torch.min(train_obj, axis=0)[0]*0.1)
    else:
        rp = torch.DoubleTensor(ref_point)

    if tkwargs["device"].type == 'cuda':
        train_x = train_x.cuda()
        train_obj = train_obj.cuda()
        rp = rp.cuda()
    return train_x, train_obj, rp

def init_experiment_input(X, y, ref_point):
    X_dim = len(X[1])
    num_objectives = len(y[1])
    bounds = generate_bounds(X, y, X_dim, num_objectives, scale=(0, 1))
    bounds = torch.FloatTensor(bounds).cuda()
    ref_point = eval(ref_point) if isinstance(ref_point, type('')) else None
    X, y, ref_point_ = generate_initial_data(X=X, y=y, ref_point=ref_point)
    return X, y, bounds, ref_point_


def optimize_qehvi_and_get_observation(model, train_obj, sampler, num_restarts, bs, bounds, raw_samples, ref_point_, all_descs ):
    """Optimizes the qEHVI acquisition function, and returns a new candidate and observation."""
    partitioning = NondominatedPartitioning(ref_point=ref_point_, Y=train_obj)
    acq_func = qExpectedHypervolumeImprovement(
        model=model,
        ref_point=ref_point_.tolist(),
        partitioning=partitioning,
        sampler=sampler,
        # maximize=True,
    )
    candidates, _ = optimize_acqf_discrete(
        acq_function=acq_func,
        bounds=bounds,
        q=bs,
        choices=all_descs,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
        options={"batch_limit": 5, "maxiter": 200, "nonnegative": True},
        sequential=False,
    )
    new_x = unnormalize(candidates.detach(), bounds=bounds)
    # new_obj = torch.FloatTensor([[  -6.7064,   -5.8886],        #？为啥不用根据new_x去计算new_obj # not used for now:date4.7
    #     [ -51.7423,   -6.8102],
    #     [ -38.3063,   -6.8469],
    #     [ -13.4827,   -9.0434],
    #     [ -10.3850,  -10.6817],
    #     [ -27.7399,   -6.6023],
    #     [ -64.7528,   -2.1669],
    #     [-168.0079,   -4.3890],
    #     [ -17.1416,  -10.4511],
    #     [  -7.0856,   -5.5974]])        #.shape = [10,2]

    return new_x


def initialize_model(train_x, train_obj):
    # define models for objective and constraint
    model = SingleTaskGP(train_x, train_obj, outcome_transform=Standardize(m=train_obj.shape[-1]))
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    return mll, model


class SearchSpace_Sampler(NormalMCSampler):
    def __init__(self, fn_dict, data_df, MC_SAMPLES='all'):
        self.fn_input = fn_dict['fn_input']
        self.df = data_df
        self.MC_SAMPLES = self.get_sample_shape(MC_SAMPLES)
        super().__init__(self.MC_SAMPLES)
        # self.bounds = bounds

    def get_sample_shape(self, num):
        if num == 'all':
            MC_SAMPLES = self.df.shape[0]
        else:
            MC_SAMPLES = num
        return MC_SAMPLES

    def forward(self, posterior):
        samples = self._construct_base_samples()
        return samples.unsqueeze(1)

    def _construct_base_samples(self):       #TODO: posterior for what
        # assert size == self.MC_SAMPLES
        daf = self.df.sample(n=self.MC_SAMPLES) 
        samples_desc = self.PCA(daf)
        samples = torch.DoubleTensor(samples_desc).cuda()
        self.base_samples = samples
        return samples

    def PCA(self, df):
        _ = df.shape[1] - 132  # changed param1
        desc = df.iloc[:, _:].values
        # X = np.array(desc)
        return self.fn_input(desc)


def MOBO_one_batch(X_train, y_train, num_restarts,
                   ref_point, bs, post_mc_samples, save_file_instance, fn_dict,
                   df_space):
    N_TRIALS = 1
    N_BATCH = 1
    MC_SAMPLES = post_mc_samples
    verbose = True
    df_space = pd.read_pickle(df_space)  #在这里改search space
    df_space.reset_index(drop=True, inplace=True)

    hvs_qehvi_all = []
    X, y, bounds, ref_point = init_experiment_input(X=X_train, y=y_train, ref_point=ref_point)
    hv = Hypervolume(ref_point=ref_point)

    # average over multiple trials
    for trial in range(1, N_TRIALS + 1):
        print(f"\nTrial {trial:>2} of {N_TRIALS} ", end="\n")
        hvs_qehvi = []
        train_x_qehvi, train_obj_qehvi = X, y
        mll_qehvi, model_qehvi = initialize_model(train_x_qehvi, train_obj_qehvi)

        pareto_mask = is_non_dominated(train_obj_qehvi)
        pareto_y = train_obj_qehvi[pareto_mask]

        volume = hv.compute(pareto_y)
        hvs_qehvi.append(volume)

        print("Hypervolume is ", volume)
        # run N_BATCH rounds of BayesOpt after the initial random batch
        for iteration in range(1, N_BATCH + 1):
            fit_gpytorch_model(mll_qehvi)
            new_sampler = SearchSpace_Sampler(fn_dict, df_space, MC_SAMPLES)        #SearchSpace_Sampler 这class没啥用，就用了其中一个PCA，代码还没改
            all_descs = torch.DoubleTensor(new_sampler.PCA(df_space)).cuda()
            new_sampler = SobolQMCNormalSampler(MC_SAMPLES)

            new_x_qehvi = optimize_qehvi_and_get_observation(
                model=model_qehvi, train_obj=train_obj_qehvi, sampler=new_sampler,
                num_restarts=num_restarts, bs=bs, bounds=bounds, raw_samples=MC_SAMPLES,
                ref_point_=ref_point, all_descs=all_descs)

            # update training points
            train_x_qehvi = torch.cat([train_x_qehvi, new_x_qehvi])
            # train_obj_qehvi = torch.cat([train_obj_qehvi, new_obj_qehvi])         #not used for now:date4.7
            print("New Samples--------------------------------------------")  # nsga-2
            recommend_descs = train_x_qehvi[-bs:]
            # print(recommend_descs)

            # save_logfile.send(('result', 'true VS pred:', dict2))

            torch.cuda.empty_cache()
            distmin_idx = compute_L2dist(recommend_descs, all_descs)
            save_recommend_comp(distmin_idx, df_space, recommend_descs)

def save_recommend_comp(idx, df_space, recommend_descs, all_descs=None):
    df_space.iloc[idx , :].to_csv("recommend_comp4.13.csv", index=True, header=True)
    print(df_space.iloc[idx , 0:4])
    df = pd.DataFrame(recommend_descs.cpu().numpy())
    df.to_csv("recommend_descs4.13.csv", index=True, header=False)
    if all_descs is not None:
        df_desc = pd.DataFrame(all_descs.cpu().numpy())
        df_desc.to_csv("all_PCAdescs4.13.csv", index=True, header=False)

def compute_L2dist(target_obj, space):
    dm = torch.cdist(target_obj, space)
    dist_min, distmin_idx = dm.min(dim=1)
    if dist_min.min() > 1e-4:
        printc.red("Warning: the distance between the recommended and the actual is too large, please check it!")
    return distmin_idx.cpu().numpy()


# ================================   以下是单变量的部分   ===================================
def elem1_train_and_plot(X, y, num_restarts, ker_lengthscale_upper, ker_var_upper, save_logfile):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    ker = GPy.kern.Matern52(input_dim=X_train.shape[1], ARD=True)  # Matern52有啥讲究吗？
    ker.lengthscale.constrain_bounded(1e-2, ker_lengthscale_upper)  # 超参数？（好像是posterior 得到的）
    ker.variance.constrain_bounded(1e-2, ker_var_upper)
    gpy_regr = GPRegression(X_train, y_train, ker)  #
    # gpy_regr.Gaussian_noise.variance = (0.01)**2       #这个一般需要怎么调整呢？（好像是posterior 得到的）
    # gpy_regr.Gaussian_noise.variance.fix()
    gpy_regr.randomize()
    gpy_regr.optimize_restarts(num_restarts=num_restarts, verbose=False, messages=False)

    dict1 = {'ker-lengthscale': ker.lengthscale.values, 'ker-variance': ker.variance.values,
             'Gaussian_noise_var': gpy_regr.Gaussian_noise.variance.values}
    save_logfile.send(('result', 'model_params:', dict1))

    y_pred_train, y_uncer_train = gpy_regr.predict(X_train)
    y_pred_test, y_uncer_test = gpy_regr.predict(X_test)
    dict2 = plt_true_vs_pred([y_train, y_test],
                             [y_pred_train, y_pred_test], [y_uncer_train, y_uncer_test],
                             ['Mat52-Train', 'Mat52-Test'],
                             ['blue', 'darkorange'], criterion='correlation')
    save_logfile.send(('result', 'true VS pred:', dict2))

    save_logfile.send(('model', '', gpy_regr))


def cross_train_validation(X_norm, y, Kfold, num_restarts, ker_lengthscale_upper, ker_var_upper, save_logfile):
    # when use K fold not use train_test split:
    X_train = X_norm;
    y_train = y[:, -1]
    # 创建一个用于得到不同训练集和测试集样本的索引的StratifiedKFold实例，折数为5
    strtfdKFold = KFold(n_splits=Kfold, shuffle=True)
    # 把特征和标签传递给StratifiedKFold实例
    kfold = strtfdKFold.split(X_train, y_train)
    # 循环迭代，（K-1）份用于训练，1份用于验证，把每次模型的性能记录下来。
    scores_train, scores_test = [], []
    uncer_train, uncer_test = [], []

    for k, (train, test) in enumerate(kfold):
        X_train_fold, y_train_fold = X_train[train], y_train[train]
        X_test_fold, y_test_fold = X_train[test], y_train[test]

        ker = GPy.kern.Matern52(input_dim=len(X_train_fold[0]), ARD=True)  # Matern52有啥讲究吗？
        ker.lengthscale.constrain_bounded(1e-2, ker_lengthscale_upper)  # 超参数？（好像是posterior 得到的）
        ker.variance.constrain_bounded(1e-2, ker_var_upper)
        gpy_regr = GPRegression(X_train_fold, y_train_fold.reshape(-1, 1), ker)  #
        # gpy_regr.Gaussian_noise.variance = (0.01)**2       #这个一般需要怎么调整呢？（好像是posterior 得到的）
        # gpy_regr.Gaussian_noise.variance.fix()
        gpy_regr.randomize()
        gpy_regr.optimize_restarts(num_restarts=num_restarts, verbose=False, messages=False)

        y_pred_train, y_uncer_train = gpy_regr.predict(X_train_fold)
        y_pred_test, y_uncer_test = gpy_regr.predict(X_test_fold)

        y_pred_train, y_uncer_train = y_pred_train[:, -1], y_uncer_train[:, -1]
        y_pred_test, y_uncer_test = y_pred_test[:, -1], y_uncer_test[:, -1]

        score_train = spearmanr(y_train_fold, y_pred_train)[0];
        scores_train.append(score_train)  # spearmanr or pearsonr
        score_test = spearmanr(y_test_fold, y_pred_test)[0];
        scores_test.append(score_test)
        # train_score = spearmanr(y_train_fold, y_pred_train) [0]
        uncer_train.append(y_uncer_train.mean())
        uncer_test.append(y_uncer_test.mean())

    dict1 = {'score_train': np.array(scores_train).mean(), 'score_train_std': np.array(scores_train).std(),
             'score_test': np.array(scores_test).mean(), 'score_test_std': np.array(scores_test).std()}
    printc.blue('\nTRAIN: Cross-Validation score: %.3f +/- %.3f' % (np.array(scores_train).mean(),
                                                                    np.array(scores_train).std()))
    printc.yellow('TEST: Cross-Validation score: %.3f +/- %.3f' % (np.array(scores_test).mean(),
                                                                   np.array(scores_test).std()))
    printc.blue('TRAIN: Cross-Validation uncertainty: %.3f +/- %.3f' % (np.array(uncer_train).mean(),
                                                                        np.array(uncer_train).std()))
    printc.yellow('TEST: Cross-Validation uncertainty: %.3f +/- %.3f' % (np.array(uncer_test).mean(),
                                                                         np.array(uncer_test).std()))

    # plot_CrossVal_avg(A, B ,C, D)
    save_logfile.send(('result', '', dict1))
    save_logfile.send(('model', '', gpy_regr))
    return dict1


# ================================   以下是2 elem预测3 elem的部分   ===================================
def cycle_train(train_data, test_data, num_restarts, ker_lengthscale_upper, ker_var_upper):
    y_list_descr = []  # 此cell为重复之前的思路的总结版
    for rep in np.arange(10):  # 10次cycle，这里相当于repeat 10次

        # def select_for_next():
        elem4_indx_random = [np.random.randint(len(X_val)) for i in np.arange(10)]
        X_val_init = X_val[elem4_indx_random]
        y_val_init = y_val[elem4_indx_random]
        X_init = np.concatenate([X_train, X_test, X_val_init])  # 这里的X_init训练集包括所有的elem3以及5个elem4 dp
        y_init = np.concatenate([y_train, y_test, y_val_init])
        X_remain = np.delete(X_val, elem4_indx_random, 0)
        y_remain = np.delete(y_val, elem4_indx_random, 0)

        for it in np.arange(20):  # 每次cycle中进行20次迭代（最终训练集size = init_X_size + 20*10)
            print("highest power so far: ", np.max(y_init))  # 源码中似乎是每次进行一次X_init的更新后进行一次pca（每个batch后）

            ker = GPy.kern.Matern52(input_dim=len(X_init[0]), ARD=True)  #
            ker.lengthscale.constrain_bounded(1e-2, ker_lengthscale_upper)
            ker.variance.constrain_bounded(1e-2, ker_var_upper)

            gpy_regr = GPRegression(X_init, y_init, ker)  #
            # gpy_regr.Gaussian_noise.variance = (0.01)**2
            # gpy_regr.Gaussian_noise.variance.fix()
            gpy_regr.randomize()
            gpy_regr.optimize_restarts(num_restarts=num_restarts, verbose=True, messages=False)
            print(ker.lengthscale)
            print(ker.variance)
            print(gpy_regr.Gaussian_noise)

            y_pred_init, y_uncer_init = gpy_regr.predict(X_init)
            y_pred_remain, y_uncer_remain = gpy_regr.predict(X_remain)
            plt_true_vs_pred([y_init, y_remain],
                             [y_pred_init, y_pred_remain], [y_uncer_init, y_uncer_remain],
                             ['GP-Mat52 - Init', 'GP-Mat52 - Remain'],
                             ['darkorange', 'darkred'])
            ucb = np.sqrt(y_uncer_remain) + y_pred_remain  # 对计算标准有一点疑问？+ 最终model的预测准确度不做评判吗？+还是只能输入一组特定材料输出Pmax？
            top_ucb_indx = np.argsort(ucb[:, -1])[-10:]
            X_new = X_remain[top_ucb_indx]  # 加入训练集的的是每次预测X_remain中的P最大的前10个（相当于选择性的将测试集中的数据加入训练）
            y_new = y_remain[top_ucb_indx]
            '''抽样策略的评价标准很重要'''

            X_init = np.concatenate([X_init, X_new])
            y_init = np.concatenate([y_init, y_new])
            X_remain = np.delete(X_remain, top_ucb_indx, 0)
            y_remain = np.delete(y_remain, top_ucb_indx, 0)
            print(len(X_init))
        y_list_descr.append(y_init)  # 记录repeat 10次中每次repeat的最后参与训练的的数据集
    return y_list_descr
