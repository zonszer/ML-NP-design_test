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
import time

import torch
from botorch import fit_gpytorch_mll
from botorch.acquisition.monte_carlo import qExpectedImprovement, qNoisyExpectedImprovement
from botorch.sampling import SobolQMCNormalSampler
from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.utils.multi_objective.hypervolume import Hypervolume

from botorch.optim.optimize import optimize_acqf, optimize_acqf_list
from botorch.utils.multi_objective.box_decompositions import NondominatedPartitioning
from botorch.acquisition.multi_objective.monte_carlo import qExpectedHypervolumeImprovement, qNoisyExpectedHypervolumeImprovement
from botorch.utils.sampling import sample_simplex
from botorch.optim import optimize_acqf_discrete

from botorch.models.gp_regression import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch.utils.transforms import unnormalize, normalize
from botorch.sampling.normal import NormalMCSampler
from botorch.utils.multi_objective.box_decompositions.dominated import (
    DominatedPartitioning,
)
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    FastNondominatedPartitioning,
)
from botorch.exceptions import BadInitialCandidatesWarning, InputDataWarning
import warnings
from botorch.utils.multi_objective.scalarization import get_chebyshev_scalarization
from botorch.acquisition.objective import GenericMCObjective
from botorch.utils.sampling import draw_sobol_samples

tkwargs = {
    # "dtype": torch.double,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}
warnings.filterwarnings("ignore", category=BadInitialCandidatesWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
# warnings.filterwarnings("ignore", category=InputDataWarning)

def generate_bounds(X, y, dim_X, num_objectives, scale=(0, 1)):
    bounds = np.zeros((2, dim_X))
    for i in range(dim_X):
        bounds[0][i] = X[:, i].min()  # min of bound
        bounds[1][i] = X[:, i].max()  # max of bound
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


def optimize_qehvi_and_get_observation(model, train_X, train_obj, sampler, num_restarts, 
                                       q_num, bounds, raw_samples,
                                       ref_point_, all_descs, max_batch_size, 
                                       validate=False, all_y=None):
    """Optimizes the qEHVI acquisition function, and returns a new candidate and observation."""
    with torch.no_grad():
        pred = model.posterior(normalize(train_X, bounds)).mean
    partitioning = FastNondominatedPartitioning(ref_point=ref_point_, Y=pred)  #    Y=pred, Y=train_obj
    # partitioning = NondominatedPartitioning(ref_point=ref_point_, Y=train_obj)
    acq_func = qExpectedHypervolumeImprovement(
        model=model,
        ref_point=ref_point_.tolist(),
        partitioning=partitioning,
        sampler=sampler,
    )
    candidates, _ = optimize_acqf_discrete(
        acq_function=acq_func,
        q=q_num,
        choices=normalize(all_descs, bounds),
        max_batch_size=max_batch_size,
        unique=False,                   #TODO: if train changed to True
        # num_restarts=num_restarts,
        # raw_samples=raw_samples,
        # options={"batch_limit": 5, "maxiter": 200, "nonnegative": False},
        # sequential=False,
    )
    new_x = unnormalize(candidates.detach(), bounds=bounds)
    # new_obj = new_obj_true + torch.randn_like(new_obj_true) * NOISE_SE
    if validate and all_y is not None:
        new_obj, new_obj_idx = get_idx_and_corObj(new_x, all_descs, all_y=all_y)
        new_obj = new_obj['all_y']
        print('idx are:', new_obj_idx)
    else:
        new_obj = None

    return new_x, new_obj

def optimize_qnehvi_and_get_observation(model, train_X, train_obj, sampler, num_restarts, 
                                       q_num, bounds, raw_samples,
                                       ref_point_, all_descs, max_batch_size, 
                                       validate=False, all_y=None):
    """Optimizes the qEHVI acquisition function, and returns a new candidate and observation."""
    acq_func = qNoisyExpectedHypervolumeImprovement(
        model=model,
        ref_point=ref_point_.tolist(),  # use known reference point
        X_baseline=normalize(train_X, bounds),
        prune_baseline=True,  # prune baseline points that have estimated zero probability of being Pareto optimal
        sampler=sampler,
    )
    # optimize
    candidates, _ = optimize_acqf_discrete(
        acq_function=acq_func,
        q=q_num,
        choices=normalize(all_descs, bounds),
        max_batch_size=max_batch_size,
        unique=False,                   #TODO: if train changed to True
        # num_restarts=num_restarts,
        # raw_samples=raw_samples,
        # options={"batch_limit": 5, "maxiter": 200, "nonnegative": False},
        # sequential=False,
    )
    # observe new values
    new_x = unnormalize(candidates.detach(), bounds=bounds)
    # new_obj = new_obj_true + torch.randn_like(new_obj_true) * NOISE_SE
    if validate and all_y is not None:
        new_obj, new_obj_idx = get_idx_and_corObj(new_x, all_descs, all_y=all_y)
        new_obj = new_obj['all_y']
        print('idx are:', new_obj_idx)
    else:
        new_obj = None

    return new_x, new_obj

def optimize_qnparego_and_get_observation(model, train_x, train_obj, sampler, num_restarts, 
                                       q_num, bounds, raw_samples,
                                       ref_point_, all_descs, max_batch_size, 
                                       validate=False, all_y=None):
    """Samples a set of random weights for each candidate in the batch, performs sequential greedy optimization
    of the qNParEGO acquisition function, and returns a new candidate and observation."""
    with torch.no_grad():
        pred = model.posterior(train_x).mean
    acq_func_list = []
    for _ in range(max_batch_size):
        weights = sample_simplex(q_num, **tkwargs).squeeze()
        objective = GenericMCObjective(
            get_chebyshev_scalarization(weights=weights, Y=pred)
        )
        acq_func = qNoisyExpectedImprovement(  # pyre-ignore: [28]
            model=model,
            objective=objective,
            X_baseline=train_x,
            sampler=sampler,
            prune_baseline=True,
        )
        acq_func_list.append(acq_func)
    # optimize
    candidates, _ = optimize_acqf_list(
        acq_function_list=acq_func_list,
        bounds=standard_bounds,
        num_restarts=num_restarts,
        raw_samples=raw_samples,  # used for intialization heuristic
        options={"batch_limit": 5, "maxiter": 200},
    )
    # observe new values
    new_x = unnormalize(candidates.detach(), bounds=problem.bounds)
    new_obj_true = problem(new_x)
    return new_x, new_obj


def get_idx_and_corObj(new_x, all_descs, **kwargs):
    '''generate new_obj from all_y and idx of new_x'''
    distmin_idx = compute_L2dist(new_x, all_descs)
    for key in kwargs.keys():
        kwargs[key] = kwargs[key][distmin_idx]
    return kwargs, distmin_idx


def initialize_model(train_x, train_obj, bounds):
    # define models for objective and constraint
    train_x = normalize(train_x, bounds)
    model = SingleTaskGP(train_x, train_obj, outcome_transform=Standardize(m=train_obj.shape[-1]))  #TODO: maybe occur bug in outcome_transform
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


def split_for_val(X, y, ini_size=0.2):
    data_num = X.shape[0]
    ini_num = int(data_num * ini_size)
    random_elements = np.random.choice(range(data_num), ini_num, replace=False)
    return X[torch.tensor(random_elements).cuda(), :], y[torch.tensor(random_elements).cuda(), :]

def optimize_and_get_random_observation(X, y, X_now, n):
    _, X_now_idx = get_idx_and_corObj(X_now, X)
    mask = np.ones_like(X.cpu().numpy(), dtype=bool); mask_y = np.ones_like(y.cpu().numpy(), dtype=bool)
    mask[X_now_idx] = False; mask_y[X_now_idx] = False
    return generate_sobol_data(X[mask].reshape(-1, X.shape[1]), y[mask_y].reshape(-1, y.shape[1]), n=n)

def generate_sobol_data(X_r, y_r, n):
    '''generate random data of new_x'''
    data_num = X_r.shape[0]
    random_elements = np.random.choice(range(data_num), n, replace=False)
    return X_r[torch.tensor(random_elements).cuda(), :], y_r[torch.tensor(random_elements).cuda(), :]

def compute_hv(hv, train_obj_qehvi):
    pareto_mask = is_non_dominated(train_obj_qehvi)
    pareto_y = train_obj_qehvi[pareto_mask]
    volume = hv.compute(pareto_y)
    return volume

def MOBO_batches(X_train, y_train, num_restarts,
                ref_point, q_num, bs, post_mc_samples, 
                save_file_instance, fn_dict,
                df_space=None):
    N_TRIALS = 1
    N_BATCH = 25
    MC_SAMPLES = post_mc_samples
    verbose = True

    hvs_qehvi_all = []
    X, y, bounds, ref_point = init_experiment_input(X=X_train, y=y_train, ref_point=ref_point)
    hv = Hypervolume(ref_point=ref_point)

    # average over multiple trials
    for trial in range(1, N_TRIALS + 1):
        print(f"\nTrial {trial:>2} of {N_TRIALS} ", end="\n")
        hvs_qehvi, hvs_random = [], []
        
        # split data for valideation
        train_x_qehvi, train_obj_qehvi = split_for_val(X, y, ini_size=0.2)
        # train_x_qparego, train_obj_x_qparego = train_x_qehvi, train_obj_qehvi
        train_x_random, train_obj_random = train_x_qehvi, train_obj_qehvi

        mll_qehvi, model_qehvi = initialize_model(train_x_qehvi, train_obj_qehvi, bounds)
        # mll_qparego, model_qparego = initialize_model(train_x_qehvi, train_obj_qehvi)

        init_volume = compute_hv(hv, train_obj_qehvi)
        hvs_qehvi.append(init_volume)
        hvs_random.append(init_volume)
        print("init Hypervolume is ", init_volume)
        # run N_BATCH rounds of BayesOpt after the initial random batch
        for iteration in range(1, N_BATCH + 1):
            t0 = time.monotonic()

            fit_gpytorch_mll(mll_qehvi)
            # fit_gpytorch_mll(mll_qparego)

            new_sampler = SobolQMCNormalSampler(sample_shape=torch.Size([MC_SAMPLES]))
            # new_sampler_qnparego = SobolQMCNormalSampler(sample_shape=torch.Size([MC_SAMPLES]))

            new_x, new_obj = optimize_qehvi_and_get_observation(
                model=model_qehvi, train_X=train_x_qehvi, train_obj=train_obj_qehvi, sampler=new_sampler, num_restarts=num_restarts, 
                q_num=q_num, bounds=bounds, raw_samples=MC_SAMPLES,
                ref_point_=ref_point, all_descs=X, max_batch_size=bs,
                all_y=y, validate=True
            )
            #new_x, new_obj = optimize_qnparego_and_get_observation(
            #    model=model_qehvi, train_obj=train_obj_qehvi, sampler=new_sampler_random, num_restarts=num_restarts, 
            #    q_num=q_num, bounds=bounds, raw_samples=MC_SAMPLES,
            #    ref_point_=ref_point, all_descs=X, max_batch_size=bs,
            #    all_y=y, validate=True, train_x=train_x_qparego
            #)
            new_x_random, new_obj_random = optimize_and_get_random_observation(X, y, X_now=train_x_random,
                                                                                n=q_num)

            # update training points
            train_x_qehvi = torch.cat([train_x_qehvi, new_x])
            train_obj_qehvi = torch.cat([train_obj_qehvi, new_obj])
            train_x_random = torch.cat([train_x_random, new_x_random])
            train_obj_random = torch.cat([train_obj_random, new_obj_random])
            
            print("--------------------------------------------")
            recommend_descs = train_x_qehvi[-q_num:]
            # print(recommend_descs)
            # update progress
            ## compute hypervolume
            hvs_qehvi.append(compute_hv(hv, train_obj_qehvi))
            hvs_random.append(compute_hv(hv, train_obj_random))

            mll_qehvi, model_qehvi = initialize_model(train_x_qehvi, train_obj_qehvi, bounds)

            t1 = time.monotonic()

            if verbose:
                print(
                    f"\nBatch {iteration:>2}: Hypervolume (random, qNParEGO, qEHVI, qNEHVI) = "
                    f"({hvs_random[-1]:>4.2f}, {hvs_qehvi[-1]:>4.2f}) "
                    f" time = {t1-t0:>4.2f}.",
                    end="",
                )
            else:
                print(".", end="")


def MOBO_one_batch(X_train, y_train, num_restarts,
                   ref_point, q_num, bs, post_mc_samples, 
                   save_file_instance, fn_dict,
                   df_space_path):
    df_space = pd.read_pickle(df_space_path)
    df_space.reset_index(drop=True, inplace=True)

    hvs_qehvi_all = []
    X, y, bounds, ref_point = init_experiment_input(X=X_train, y=y_train, ref_point=ref_point)
    hv = Hypervolume(ref_point=ref_point)

    # average over multiple trials
    for trial in range(1, 2):
        print(f"\nTrial {trial:>2}", end="\n")
        hvs_qehvi = []
        train_x_qehvi, train_obj_qehvi = X, y
        mll_qehvi, model_qehvi = initialize_model(train_x_qehvi, train_obj_qehvi, bounds)

        pareto_mask = is_non_dominated(train_obj_qehvi)
        pareto_y = train_obj_qehvi[pareto_mask]
        volume = hv.compute(pareto_y)
        hvs_qehvi.append(volume)

        print("Hypervolume is ", volume)
        # run N_BATCH rounds of BayesOpt after the initial random batch
        for iteration in range(1, 2):
            fit_gpytorch_mll(mll_qehvi)
            new_sampler = SearchSpace_Sampler(fn_dict, df_space, post_mc_samples)        #SearchSpace_Sampler 这class没啥用，就用了其中一个PCA，代码还没改
            all_descs = torch.DoubleTensor(new_sampler.PCA(df_space)).cuda()
            new_sampler = SobolQMCNormalSampler(sample_shape=torch.Size([post_mc_samples]))

            new_x_qehvi, _ = optimize_qehvi_and_get_observation(    #or use optimize_qnehvi_and_get_observation
                model=model_qehvi, train_X=train_x_qehvi, train_obj=train_obj_qehvi,
                sampler=new_sampler, num_restarts=num_restarts,
                q_num=q_num, bounds=bounds, raw_samples=post_mc_samples,
                ref_point_=ref_point, all_descs=all_descs, max_batch_size=bs
            )

            # update training points
            train_x_qehvi = torch.cat([train_x_qehvi, new_x_qehvi])
            # train_obj_qehvi = torch.cat([train_obj_qehvi, new_obj_qehvi])         #not used for now:date4.7
            print("New Samples--------------------------------------------")  # nsga-2
            recommend_descs = train_x_qehvi[-q_num:]
            # print(recommend_descs)

            # save_logfile.send(('result', 'true VS pred:', dict2))

            torch.cuda.empty_cache()
            distmin_idx = compute_L2dist(recommend_descs, all_descs)
            save_recommend_comp(distmin_idx, df_space, recommend_descs, all_descs, df_space_path)      #opt: all_descs

def save_recommend_comp(idx, df_space, recommend_descs, all_descs=None, df_space_path='RuRu'):
    str1 = get_str_after_substring(df_space_path, 'Ru')
    df_space.iloc[idx , :].to_csv("recommend_comp{}.csv".format(str1), index=True, header=True)
    print(df_space.iloc[idx , 0:4])
    df = pd.DataFrame(recommend_descs.cpu().numpy())
    df.to_csv("recommend_descs{}.csv".format(str1), index=True, header=False)
    if all_descs is not None:
        df_desc = pd.DataFrame(all_descs.cpu().numpy())
        df_desc.to_csv("all_PCAdescs{}.csv".format(str1), index=True, header=False)

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
