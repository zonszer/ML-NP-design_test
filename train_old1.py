from utils.utils_ import *
import pandas as pd
import time

import torch
from botorch import fit_gpytorch_mll
from botorch.acquisition.monte_carlo import qExpectedImprovement, qNoisyExpectedImprovement, qUpperConfidenceBound
# from botorch.acquisition import qExpectedImprovement, qNoisyExpectedImprovement
from botorch.sampling import SobolQMCNormalSampler
from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.utils.multi_objective.hypervolume import Hypervolume

from botorch.optim.optimize import optimize_acqf, optimize_acqf_list
from botorch.utils.multi_objective.box_decompositions import NondominatedPartitioning
from botorch.acquisition.multi_objective.monte_carlo import qExpectedHypervolumeImprovement, qNoisyExpectedHypervolumeImprovement
from botorch.acquisition import UpperConfidenceBound
from botorch.utils.sampling import sample_simplex
from botorch.optim import optimize_acqf_discrete

from botorch.models.gp_regression import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from gpytorch.priors import UniformPrior
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.constraints import Interval
from botorch.utils.transforms import unnormalize, normalize
from botorch.sampling.normal import NormalMCSampler
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
        bounds[0][i] = X[:, i].min() * 1. # min of bound   #TODO: can not deine the bound of the problem determailtically
        bounds[1][i] = X[:, i].max() * 1.  # max of bound
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
                                       fn_dict,
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
        unique=True,                   #TODO: if train changed to True
        # num_restarts=num_restarts,
        # raw_samples=raw_samples,
        # options={"batch_limit": 5, "maxiter": 200, "nonnegative": False},
        # sequential=False,
    )
    get_candidates_pred(model, candidates, fn_dict)
    new_x = unnormalize(candidates.detach(), bounds=bounds)
    # new_obj = new_obj_true + torch.randn_like(new_obj_true) * NOISE_SE
    if validate and all_y is not None:
        new_obj, new_obj_idx = get_idx_and_corObj(new_x, all_descs, all_y=all_y)
        new_obj = new_obj['all_y']
        print('idx are:', new_obj_idx)
    else:
        new_obj = None
    return new_x, new_obj

def get_candidates_pred(model, candidates, fn_dict):
    pred_mean = model.posterior(candidates).mean.detach().cpu().numpy()
    pred_var = model.posterior(candidates).variance.detach().cpu().numpy()
    for i in range(pred_mean.shape[-1]):
        # assert '''y1 is y['slope relative to Ru']'''
        if i == 1:
            pred_mean[:, i] = -pred_mean[:, i]
        if 'std_scaler_y'+str(i) in fn_dict:
            pred_mean[:, i] = fn_dict['std_scaler_y'+str(i)].inverse_transform(pred_mean[:, i].reshape(-1, 1))[:, -1]
    np.savetxt("pred_meanORE.csv", pred_mean, delimiter=",")
    return pred_mean


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


def initialize_model(train_x, train_obj, bounds, lengthscale, state_dict=None):
    # define models for objective and constraint
    train_x = normalize(train_x, bounds)
    ker = MaternKernel(nu=2.5, ard_num_dims=train_x.shape[-1], lengthscale_constraint=lengthscale).cuda()
    # ker.lengthscale_constraint = lengthscale    #TODO: lengthscale_constraint is seemingly useless
    ker = ScaleKernel(ker)
    model = SingleTaskGP(train_x, train_obj, covar_module=ker,
                         outcome_transform=Standardize(m=train_obj.shape[-1]))  #TODO: maybe occur bug in outcome_transform
    model_parameters = model.state_dict()
    # Load state_dict if it is provided
    if state_dict is not None:
        model.load_state_dict(state_dict)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    return mll, model


class SearchSpace_Sampler(NormalMCSampler):
    def __init__(self, fn_dict, data_df, MC_SAMPLES='all'):
        self.fn_input = fn_dict['fn_input']
        self.df = data_df
        # self.bounds = bounds

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


def MOBO_one_batch(X_train, y_train, num_restarts,
                   ref_point, q_num, bs, post_mc_samples, 
                   save_file_instance, fn_dict,
                   df_space_path, 
                   ker_lengthscale_upper):
    df_space = pd.read_pickle(df_space_path)
    df_space.reset_index(drop=True, inplace=True)

    hvs_qehvi_all = []
    train_x_qehvi, train_obj_qehvi, bounds, ref_point = init_experiment_input(X=X_train, y=y_train, ref_point=ref_point)
    hv = Hypervolume(ref_point=ref_point)

    # average over multiple trials
    for trial in range(1, 2):
        print(f"\nTrial {trial:>2}", end="\n")
        hvs_qehvi = []
        mll_qehvi, model_qehvi = initialize_model(train_x_qehvi, train_obj_qehvi, bounds, lengthscale=Interval(0.01, ker_lengthscale_upper))

        pareto_mask = is_non_dominated(train_obj_qehvi)
        pareto_y = train_obj_qehvi[pareto_mask]
        volume = hv.compute(pareto_y)
        hvs_qehvi.append(volume)

        print("Hypervolume is ", volume)
        # run N_BATCH rounds of BayesOpt after the initial random batch
        for __ in range(1, 2):
            fit_gpytorch_mll(mll_qehvi)
            # file_path = "output.txt"  # The path to the output file
            # with open(file_path, 'a') as file:
            #     file.write(str(model_qehvi.covar_module.base_kernel.lengthscale))
            new_sampler = SearchSpace_Sampler(fn_dict, df_space, post_mc_samples)        #SearchSpace_Sampler 这class没啥用，就用了其中一个PCA，代码还没改
            all_descs = torch.DoubleTensor(new_sampler.PCA(df_space)).cuda()
            new_sampler = SobolQMCNormalSampler(sample_shape=torch.Size([post_mc_samples]))

            new_x_qehvi, _ = optimize_qehvi_and_get_observation(    #or use optimize_qnehvi_and_get_observation
                model=model_qehvi, train_X=train_x_qehvi, train_obj=train_obj_qehvi,
                sampler=new_sampler, num_restarts=num_restarts,
                q_num=q_num, bounds=bounds, raw_samples=post_mc_samples,
                ref_point_=ref_point, all_descs=all_descs, max_batch_size=bs,
                fn_dict=fn_dict,
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
            save_recommend_comp(distmin_idx, df_space, recommend_descs,
                                df_space_path=df_space_path, iter="1iters")      #opt: all_descs


def MOBO_batches(X_train, y_train, 
                save_file_instance, 
                fn_dict,
                ref_point, q_num, bs, mc_samples_num, 
                num_restarts,
                split_ratio, ker_lengthscale_upper, 
                df_space=None,
                **kwargs):
    N_TRIALS = 1
    N_BATCH = 100
    MC_SAMPLES = mc_samples_num
    verbose = True

    hvs_qehvi_all = []
    X, y, bounds, ref_point = init_experiment_input(X=X_train, y=y_train, ref_point=ref_point)
    hv = Hypervolume(ref_point=ref_point)

    # average over multiple trials
    for trial in range(1, N_TRIALS + 1):
        print(f"\nTrial {trial:>2} of {N_TRIALS} ", end="\n")
        hvs_qehvi, hvs_random = [], []
        
        # split data for valideation
        if split_ratio != 0:
            train_x_qehvi, train_obj_qehvi = split_for_val(X, y, ini_size=split_ratio)
        else:
            train_x_qehvi, train_obj_qehvi = X, y
        # train_x_qparego, train_obj_x_qparego = train_x_qehvi, train_obj_qehvi
        train_x_random, train_obj_random = train_x_qehvi, train_obj_qehvi

        mll_qehvi, model_qehvi = initialize_model(train_x_qehvi, 
                                                  train_obj_qehvi, 
                                                  bounds, 
                                                  lengthscale=Interval(0.01, ker_lengthscale_upper))
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
                all_y=y, validate=True,
                fn_dict=fn_dict,
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

            mll_qehvi, model_qehvi = initialize_model(train_x_qehvi, train_obj_qehvi, bounds, lengthscale=Interval(0.01, ker_lengthscale_upper))

            t1 = time.monotonic()

            if verbose:
                print(
                    f"\nBatch {iteration:>2}: Hypervolume (random, qEHVI) = "   #qNEHVI
                    f"({hvs_random[-1]:>4.2f}, {hvs_qehvi[-1]:>4.2f}) "
                    f" time = {t1-t0:>4.2f}.",
                    end="",
                )
            else:
                print(".", end="")


def SOBO_one_batch(X_train, y_train, num_restarts,
                   ref_point, q_num, bs, post_mc_samples, 
                   save_file_instance, fn_dict,
                   df_space_path, 
                   ker_lengthscale_upper,
                   beta):
    df_space = pd.read_pickle(df_space_path)
    df_space.reset_index(drop=True, inplace=True)

    train_x_ucb, train_obj_ucb, bounds, _ = init_experiment_input(X=X_train, y=y_train, ref_point=ref_point)

    for trial in range(1, 2):
        print(f"\nTrial {trial:>2}", end="\n")
        mll_ucb, model_ucb = initialize_model(train_x_ucb, train_obj_ucb, bounds, 
                                                  lengthscale=Interval(0.01, ker_lengthscale_upper))
        # ucb = UpperConfidenceBound(model_ucb, beta=beta)
        ucb_qkg_acqf = qUpperConfidenceBound(model_ucb, beta=beta)

        for __ in range(1, 2):
            fit_gpytorch_mll(mll_ucb)
            new_sampler = SearchSpace_Sampler(fn_dict, df_space, post_mc_samples)        #SearchSpace_Sampler 这class没啥用，就用了其中一个PCA，代码还没改
            all_descs = torch.DoubleTensor(new_sampler.PCA(df_space)).cuda()

            candidates, _ = optimize_acqf_discrete(
                acq_function=ucb_qkg_acqf,
                q=q_num,
                choices=normalize(all_descs, bounds),
                max_batch_size=bs,
                unique=True,                   #TODO: if train changed to True
                # num_restarts=num_restarts,
                # raw_samples=raw_samples,
                # options={"batch_limit": 5, "maxiter": 200, "nonnegative": False},
                # sequential=False,
            )
            get_candidates_pred(model_ucb, candidates, fn_dict)
            new_x_ucb = unnormalize(candidates.detach(), bounds=bounds)
            # update training points
            train_x_ucb = torch.cat([train_x_ucb, new_x_ucb])
            # train_obj_ucb = torch.cat([train_obj_ucb, new_obj_qehvi])         #not used for now:date4.7
            print("New Samples--------------------------------------------")  # nsga-2
            recommend_descs = train_x_ucb[-q_num:]
            # print(recommend_descs)
            # save_logfile.send(('result', 'true VS pred:', dict2))

            torch.cuda.empty_cache()
            distmin_idx = compute_L2dist(recommend_descs, all_descs)
            save_recommend_comp(distmin_idx, df_space, recommend_descs, 
                                df_space_path=df_space_path)      #opt: all_descs


def save_recommend_comp(idx, df_space, recommend_descs, all_descs=None, df_space_path=None, iter=None):
    str1 = get_str_after_substring(df_space_path, 'Ru')
    df_space.iloc[idx , :].to_csv(f"recommend_comp{str1}-{iter}.csv", index=True, header=True)
    print(df_space.iloc[idx , 0:4])
    df = pd.DataFrame(recommend_descs.cpu().numpy())
    df.to_csv(f"recommend_descs{str1}-{iter}.csv", index=True, header=False)
    if all_descs is not None:
        df_desc = pd.DataFrame(all_descs.cpu().numpy())
        df_desc.to_csv(f"all_PCAdescs{str1}-{iter}.csv", index=True, header=False)

def compute_L2dist(target_obj, space):
    dm = torch.cdist(target_obj, space)
    dist_min, distmin_idx = dm.min(dim=1)
    if dist_min.min() > 1e-4:
        printc.red("Warning: the distance between the recommended and the actual is too large, please check it!")
    return distmin_idx.cpu().numpy()
