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


class MLModel:
    def __init__(self, 
                X_train, y_train, 
                fn_dict,
                ref_point, 
                q_num, bs, 
                mc_samples_num, 
                ker_lengthscale_upper, 
                df_space_path=None,
                **kwargs):
        
        self.X_train = X_train
        self.y_train = y_train
        self.ref_point = ref_point
        self.q_num = q_num
        self.bs = bs
        self.mc_samples_num = mc_samples_num
        self.ker_lengthscale_upper = ker_lengthscale_upper
        self.df_space_path = df_space_path
        self.fn_dict = fn_dict
        if df_space_path is not None:
            self.df_space = pd.read_pickle(df_space_path)
            self.df_space.reset_index(drop=True, inplace=True)
        for key, value in kwargs.items():
            setattr(self, key, value)
            # self.beta = kwargs['beta'] 'split_ratio'
            # self.save_file_instance = kwargs['save_file_instance']
            # self.num_restarts = kwargs['num_restarts']

    def generate_bounds(self, X, y, dim_X, num_objectives, scale=(0, 1)):
        bounds = np.zeros((2, dim_X))
        for i in range(dim_X):
            bounds[0][i] = X[:, i].min() * 1. # min of bound   #TODO: can not define the bound of the problem deterministically
            bounds[1][i] = X[:, i].max() * 1.  # max of bound
        return bounds

    def generate_initial_data(self, X, y, ref_point):
        '''generate training data'''
        train_x = torch.DoubleTensor(X).to(self.device)
        train_obj = torch.DoubleTensor(y).to(self.device)
        if ref_point is None:
            rp = torch.min(train_obj, axis=0)[0] - torch.abs(torch.min(train_obj, axis=0)[0]*0.1)
        else:
            rp = torch.DoubleTensor(ref_point).to(self.device)
        return train_x, train_obj, rp

    def init_experiment_input(self, X, y, ref_point):
        X_dim = len(X[1])
        num_objectives = len(y[1])
        bounds = self.generate_bounds(X, y, X_dim, num_objectives, scale=(0, 1))
        bounds = torch.FloatTensor(bounds).cuda()
        ref_point = eval(ref_point) if isinstance(ref_point, type('')) else None
        X, y, ref_point_ = self.generate_initial_data(X=X, y=y, ref_point=ref_point)
        return X, y, bounds, ref_point_

    def optimize_qehvi_and_get_observation(self, model, train_X, train_obj, sampler, num_restarts, 
                                           q_num, bounds, raw_samples, ref_point_, 
                                           all_descs, max_batch_size, fn_dict, 
                                           validate=False, all_y=None):
        """Optimizes the qEHVI acquisition function, and returns a new candidate and observation."""
        with torch.no_grad():
            pred = model.posterior(normalize(train_X, bounds)).mean
        partitioning = FastNondominatedPartitioning(ref_point=ref_point_, Y=pred)
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
            unique=True,
        )
        self.get_candidates_pred(model, candidates, fn_dict)
        new_x = unnormalize(candidates.detach(), bounds=bounds)
        if validate and all_y is not None:
            new_obj, new_obj_idx = self.get_idx_and_corObj(new_x, all_descs, all_y=all_y)
            new_obj = new_obj['all_y']
            print('idx are:', new_obj_idx)
        else:
            new_obj = None
        return new_x, new_obj

    def optimize_and_get_random_observation(self, X, y, X_now, n):
        _, X_now_idx = self.get_idx_and_corObj(X_now, X)
        mask = np.ones_like(X.cpu().numpy(), dtype=bool);
        mask_y = np.ones_like(y.cpu().numpy(), dtype=bool)
        mask[X_now_idx] = False;
        mask_y[X_now_idx] = False
        return self.generate_sobol_data(X[mask].reshape(-1, X.shape[1]), y[mask_y].reshape(-1, y.shape[1]), n=n)

    def generate_sobol_data(self, X_r, y_r, n):
        '''generate random data of new_x'''
        data_num = X_r.shape[0]
        random_elements = np.random.choice(range(data_num), n, replace=False)
        return X_r[torch.tensor(random_elements).cuda(), :], y_r[torch.tensor(random_elements).cuda(), :]

    def get_candidates_pred(self, model, candidates, fn_dict):
        pred_mean = model.posterior(candidates).mean.detach().cpu().numpy()
        pred_var = model.posterior(candidates).variance.detach().cpu().numpy()
        for i in range(pred_mean.shape[-1]):
            if i == 1:
                pred_mean[:, i] = -pred_mean[:, i]
            if 'std_scaler_y'+str(i) in fn_dict:
                pred_mean[:, i] = fn_dict['std_scaler_y'+str(i)].inverse_transform(pred_mean[:, i].reshape(-1, 1))[:, -1]
        np.savetxt("pred_meanORE.csv", pred_mean, delimiter=",")
        return pred_mean

    def optimize_qnehvi_and_get_observation(self, model, train_X, train_obj, sampler, num_restarts, q_num, bounds, raw_samples, ref_point_, all_descs, max_batch_size, validate=False, all_y=None):
        """Optimizes the qEHVI acquisition function, and returns a new candidate and observation."""
        acq_func = qNoisyExpectedHypervolumeImprovement(
            model=model,
            ref_point=ref_point_.tolist(),
            X_baseline=normalize(train_X, bounds),
            prune_baseline=True,
            sampler=sampler,
        )
        candidates, _ = optimize_acqf_discrete(
            acq_function=acq_func,
            q=q_num,
            choices=normalize(all_descs, bounds),
            max_batch_size=max_batch_size,
            unique=False,
        )
        new_x = unnormalize(candidates.detach(), bounds=bounds)
        if validate and all_y is not None:
            new_obj, new_obj_idx = self.get_idx_and_corObj(new_x, all_descs, all_y=all_y)
            new_obj = new_obj['all_y']
            print('idx are:', new_obj_idx)
        else:
            new_obj = None
        return new_x, new_obj

    def get_idx_and_corObj(self, new_x, all_descs, **kwargs):
        '''generate new_obj from all_y and idx of new_x'''
        distmin_idx = self.compute_L2dist(new_x, all_descs)
        for key in kwargs.keys():
            kwargs[key] = kwargs[key][distmin_idx]
        return kwargs, distmin_idx
    
    def compute_hv(self, hv, train_obj_qehvi):
        pareto_mask = is_non_dominated(train_obj_qehvi)
        pareto_y = train_obj_qehvi[pareto_mask]
        volume = hv.compute(pareto_y)
        return volume

    def split_for_val(self, X, y, ini_size=0.2):
        data_num = X.shape[0]
        ini_num = int(data_num * ini_size)
        random_elements = np.random.choice(range(data_num), ini_num, replace=False)
        return X[torch.tensor(random_elements).cuda(), :], y[torch.tensor(random_elements).cuda(), :]

    def transform_fn_PCA(self, df=None):
        if df is None:
            df = self.df_space
        _ = df.shape[1] - 132  # changed param1
        desc = df.iloc[:, _:].values
        # X = np.array(desc)
        return self.fn_dict['fn_input'](desc)

    def initialize_model(self, train_x, train_obj, bounds, lengthscale, state_dict=None):
        train_x = normalize(train_x, bounds)
        ker = MaternKernel(nu=2.5, ard_num_dims=train_x.shape[-1], lengthscale_constraint=lengthscale).cuda()
        ker = ScaleKernel(ker)
        model = SingleTaskGP(train_x, train_obj, covar_module=ker, outcome_transform=Standardize(m=train_obj.shape[-1]))
        model_parameters = model.state_dict()
        if state_dict is not None:
            model.load_state_dict(state_dict)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        return mll, model

    def MOBO_one_batch(self):
        hvs_qehvi_all = []
        train_x_qehvi, train_obj_qehvi, bounds, ref_point = self.init_experiment_input(X=self.X_train, 
                                                                                       y=self.y_train, 
                                                                                       ref_point=self.ref_point)
        hv = Hypervolume(ref_point=ref_point)

        for trial in range(1, 2):
            print(f"\nTrial {trial:>2}", end="\n")
            hvs_qehvi = []
            mll_qehvi, model_qehvi = self.initialize_model(train_x_qehvi, train_obj_qehvi, bounds, 
                                                           lengthscale=Interval(0.01, self.ker_lengthscale_upper))

            pareto_mask = is_non_dominated(train_obj_qehvi)
            pareto_y = train_obj_qehvi[pareto_mask]
            volume = hv.compute(pareto_y)
            hvs_qehvi.append(volume)
            print("Hypervolume is ", volume)

            for __ in range(1, 2):
                fit_gpytorch_mll(mll_qehvi)
                all_descs = torch.DoubleTensor(self.transform_fn_PCA(df=self.df_space)).cuda()
                new_sampler = SobolQMCNormalSampler(sample_shape=torch.Size([self.mc_samples_num]))

                new_x_qehvi, _ = self.optimize_qehvi_and_get_observation(
                    model=model_qehvi, train_X=train_x_qehvi, train_obj=train_obj_qehvi,
                    sampler=new_sampler, num_restarts=self.num_restarts,
                    q_num=self.q_num, bounds=bounds, raw_samples=self.mc_samples_num,
                    ref_point_=ref_point, all_descs=all_descs, max_batch_size=self.bs,
                    fn_dict=self.fn_dict,
                )

                train_x_qehvi = torch.cat([train_x_qehvi, new_x_qehvi])
                print("New Samples--------------------------------------------")
                recommend_descs = train_x_qehvi[-self.q_num:]
                torch.cuda.empty_cache()
                distmin_idx = self.compute_L2dist(recommend_descs, all_descs)
                self.save_recommend_comp(distmin_idx, self.df_space, recommend_descs, iter="1iters")
    
    def MOBO_batches(self):
        N_TRIALS = 1
        N_BATCH = 100
        verbose = True

        hvs_qehvi_all = []
        X, y, bounds, ref_point = self.init_experiment_input(X=self.X_train, y=self.y_train, ref_point=self.ref_point)
        hv = Hypervolume(ref_point=ref_point)

        for trial in range(1, N_TRIALS + 1):
            print(f"\nTrial {trial:>2} of {N_TRIALS} ", end="\n")
            hvs_qehvi, hvs_random = [], []

            if self.split_ratio != 0:
                train_x_qehvi, train_obj_qehvi = self.split_for_val(X, y, ini_size=self.split_ratio)
            else:
                train_x_qehvi, train_obj_qehvi = X, y
            train_x_random, train_obj_random = train_x_qehvi, train_obj_qehvi

            mll_qehvi, model_qehvi = self.initialize_model(train_x_qehvi, 
                                                           train_obj_qehvi, 
                                                           bounds, 
                                                           lengthscale=Interval(0.01, self.ker_lengthscale_upper))

            init_volume = self.compute_hv(hv, train_obj_qehvi)
            hvs_qehvi.append(init_volume)
            hvs_random.append(init_volume)
            print("init Hypervolume is ", init_volume)

            for iteration in range(1, N_BATCH + 1):
                t0 = time.monotonic()

                fit_gpytorch_mll(mll_qehvi)

                new_sampler = SobolQMCNormalSampler(sample_shape=torch.Size([self.mc_samples_num]))

                new_x, new_obj = self.optimize_qehvi_and_get_observation(
                    model=model_qehvi, train_X=train_x_qehvi, 
                    train_obj=train_obj_qehvi, sampler=new_sampler, 
                    num_restarts=self.num_restarts,
                    q_num=self.q_num, bounds=bounds, 
                    raw_samples=self.mc_samples_num,
                    ref_point_=ref_point, all_descs=X, max_batch_size=self.bs,
                    all_y=y, validate=True,
                    fn_dict=self.fn_dict,
                )

                new_x_random, new_obj_random = self.optimize_and_get_random_observation(X, y, X_now=train_x_random, n=self.q_num)

                train_x_qehvi = torch.cat([train_x_qehvi, new_x])
                train_obj_qehvi = torch.cat([train_obj_qehvi, new_obj])
                train_x_random = torch.cat([train_x_random, new_x_random])
                train_obj_random = torch.cat([train_obj_random, new_obj_random])

                print("--------------------------------------------")
                recommend_descs = train_x_qehvi[-self.q_num:]
                hvs_qehvi.append(self.compute_hv(hv, train_obj_qehvi))
                hvs_random.append(self.compute_hv(hv, train_obj_random))

                mll_qehvi, model_qehvi = self.initialize_model(train_x_qehvi, train_obj_qehvi, bounds, 
                                                               lengthscale=Interval(0.01, self.ker_lengthscale_upper))

                t1 = time.monotonic()

                if verbose:
                    print(
                        f"\nBatch {iteration:>2}: Hypervolume (random, qEHVI) = "
                        f"({hvs_random[-1]:>4.2f}, {hvs_qehvi[-1]:>4.2f}) "
                        f" time = {t1-t0:>4.2f}.",
                        end="",
                    )
                else:
                    print(".", end="")

    def SOBO_one_batch(self):
        train_x_ucb, train_obj_ucb, bounds, _ = self.init_experiment_input(X=self.X_train, y=self.y_train, 
                                                                           ref_point=self.ref_point)

        for trial in range(1, 2):
            print(f"\nTrial {trial:>2}", end="\n")
            mll_ucb, model_ucb = self.initialize_model(train_x_ucb, train_obj_ucb, bounds, 
                                                       lengthscale=Interval(0.01, self.ker_lengthscale_upper))
            ucb_qkg_acqf = qUpperConfidenceBound(model_ucb, beta=self.beta)

            for __ in range(1, 2):
                fit_gpytorch_mll(mll_ucb)
                all_descs = torch.DoubleTensor(self.transform_fn_PCA(df=self.df_space)).cuda()

                candidates, _ = optimize_acqf_discrete(
                    acq_function=ucb_qkg_acqf,
                    q=self.q_num,
                    choices=normalize(all_descs, bounds),
                    max_batch_size=self.bs,
                    unique=True,
                )
                self.get_candidates_pred(model_ucb, candidates, self.fn_dict)
                new_x_ucb = unnormalize(candidates.detach(), bounds=bounds)

                train_x_ucb = torch.cat([train_x_ucb, new_x_ucb])
                print("New Samples--------------------------------------------")
                recommend_descs = train_x_ucb[-self.q_num:]
                torch.cuda.empty_cache()
                distmin_idx = self.compute_L2dist(recommend_descs, all_descs)
                self.save_recommend_comp(distmin_idx, self.df_space, recommend_descs, 
                                         df_space_path=self.df_space_path)

    def save_recommend_comp(self, idx, df_space, recommend_descs, all_descs=None, iter=None):
        str1 = get_str_after_substring(self.df_space_path, 'Ru')
        df_space.iloc[idx , :].to_csv(f"recommend_comp{str1}-{iter}.csv", index=True, header=True)
        print(df_space.iloc[idx , 0:4])
        df = pd.DataFrame(recommend_descs.cpu().numpy())
        df.to_csv(f"recommend_descs{str1}-{iter}.csv", index=True, header=False)
        if all_descs is not None:
            df_desc = pd.DataFrame(all_descs.cpu().numpy())
            df_desc.to_csv(f"all_PCAdescs{str1}-{iter}.csv", index=True, header=False)

    def compute_L2dist(self, target_obj, space):
        dm = torch.cdist(target_obj, space)
        dist_min, distmin_idx = dm.min(dim=1)
        if dist_min.min() > 1e-4:
            printc.red("Warning: the distance between the recommended and the actual is too large, please check it!")
        return distmin_idx.cpu().numpy()
