To refactor the code into a class form, you can create a class called "MLModel" that contains all the functions as methods. Here's an example of how you can organize the code into a class:

```python
import pandas as pd
import time
import torch
from botorch import fit_gpytorch_mll
from botorch.acquisition.monte_carlo import qExpectedImprovement, qNoisyExpectedImprovement, qUpperConfidenceBound
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
from botorch.utils.multi_objective.box_decompositions.non_dominated import FastNondominatedPartitioning
from botorch.exceptions import BadInitialCandidatesWarning, InputDataWarning
import warnings
from botorch.utils.multi_objective.scalarization import get_chebyshev_scalarization
from botorch.acquisition.objective import GenericMCObjective
from botorch.utils.sampling import draw_sobol_samples

class MLModel:
    def __init__(self, X_train, y_train, ref_point, q_num, bs, mc_samples_num, num_restarts, split_ratio, ker_lengthscale_upper, df_space_path, beta):
        self.X_train = X_train
        self.y_train = y_train
        self.ref_point = ref_point
        self.q_num = q_num
        self.bs = bs
        self.mc_samples_num = mc_samples_num
        self.num_restarts = num_restarts
        self.split_ratio = split_ratio
        self.ker_lengthscale_upper = ker_lengthscale_upper
        self.df_space_path = df_space_path
        self.beta = beta
        self.bounds = None
        self.hv = None

    def generate_bounds(self, X, y, dim_X, num_objectives, scale=(0, 1)):
        bounds = np.zeros((2, dim_X))
        for i in range(dim_X):
            bounds[0][i] = X[:, i].min() * 1. # min of bound   #TODO: can not define the bound of the problem deterministically
            bounds[1][i] = X[:, i].max() * 1.  # max of bound
        return bounds

    def generate_initial_data(self, X, y, ref_point):
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

    def init_experiment_input(self, X, y, ref_point):
        X_dim = len(X[1])
        num_objectives = len(y[1])
        bounds = self.generate_bounds(X, y, X_dim, num_objectives, scale=(0, 1))
        bounds = torch.FloatTensor(bounds).cuda()
        ref_point = eval(ref_point) if isinstance(ref_point, type('')) else None
        X, y, ref_point_ = self.generate_initial_data(X=X, y=y, ref_point=ref_point)
        return X, y, bounds, ref_point_

    def optimize_qehvi_and_get_observation(self, model, train_X, train_obj, sampler, num_restarts, q_num, bounds, raw_samples, ref_point_, all_descs, max_batch_size, fn_dict, validate=False, all_y=None):
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
        get_candidates_pred(model, candidates, fn_dict)
        new_x = unnormalize(candidates.detach(), bounds=bounds)
        if validate and all_y is not None:
            new_obj, new_obj_idx = get_idx_and_corObj(new_x, all_descs, all_y=all_y)
            new_obj = new_obj['all_y']
            print('idx are:', new_obj_idx)
        else:
            new_obj = None
        return new_x, new_obj

    def compute_hv(self, hv, train_obj_qehvi):
        pareto_mask = is_non_dominated(train_obj_qehvi)
        pareto_y = train_obj_qehvi[pareto_mask]
        volume = hv.compute(pareto_y)
        return volume

    def MOBO_one_batch(self):
        df_space = pd.read_pickle(self.df_space_path)
        df_space.reset_index(drop=True, inplace=True)

        hvs_qehvi_all = []
        train_x_qehvi, train_obj_qehvi, bounds, ref_point = self.init_experiment_input(X=self.X_train, y=self.y_train, ref_point=self.ref_point)
        hv = Hypervolume(ref_point=ref_point)

        for trial in range(1, 2):
            print(f"\nTrial {trial:>2}", end="\n")
            hvs_qehvi = []
            mll_qehvi, model_qehvi = initialize_model(train_x_qehvi, train_obj_qehvi, bounds, lengthscale=Interval(0.01, self.ker_lengthscale_upper))

            pareto_mask = is_non_dominated(train_obj_qehvi)
            pareto_y = train_obj_qehvi[pareto_mask]
            volume = hv.compute(pareto_y)
            hvs_qehvi.append(volume)

            print("Hypervolume is ", volume)
            for __ in range(1, 2):
                fit_gpytorch_mll(mll_qehvi)
                new_sampler = SearchSpace_Sampler(fn_dict, df_space, self.mc_samples_num)
                all_descs = torch.DoubleTensor(new_sampler.PCA(df_space)).cuda()
                new_sampler = SobolQMCNormalSampler(sample_shape=torch.Size([self.mc_samples_num]))

                new_x_qehvi, _ = self.optimize_qehvi_and_get_observation(
                    model=model_qehvi, train_X=train_x_qehvi, train_obj=train_obj_qehvi,
                    sampler=new_sampler, num_restarts=self.num_restarts,
                    q_num=self.q_num, bounds=bounds, raw_samples=self.mc_samples_num,
                    ref_point_=ref_point, all_descs=all_descs, max_batch_size=self.bs,
                    fn_dict=fn_dict,
                )

                train_x_qehvi = torch.cat([train_x_qehvi, new_x_qehvi])
                print("New Samples--------------------------------------------")
                recommend_descs = train_x_qehvi[-self.q_num:]
                torch.cuda.empty_cache()
                distmin_idx = compute_L2dist(recommend_descs, all_descs)
                save_recommend_comp(distmin_idx, df_space, recommend_descs,
                                    df_space_path=self.df_space_path, iter="1iters")

    def SOBO_one_batch(self):
        df_space = pd.read_pickle(self.df_space_path)
        df_space.reset_index(drop=True, inplace=True)

        train_x_ucb, train_obj_ucb, bounds, _ = self.init_experiment_input(X=self.X_train, y=self.y_train, ref_point=self.ref_point)

        for trial in range(1, 2):
            print(f"\nTrial {trial:>2}", end="\n")
            mll_ucb, model_ucb = initialize_model(train_x_ucb, train_obj_ucb, bounds, 
                                                      lengthscale=Interval(0.01, self.ker_lengthscale_upper))
            ucb_qkg_acqf = qUpperConfidenceBound(model_ucb, beta=self.beta)

            for __ in range(1, 2):
                fit_gpytorch_mll(mll_ucb)
                new_sampler = SearchSpace_Sampler(fn_dict, df_space, self.mc_samples_num)
                all_descs = torch.DoubleTensor(new_sampler.PCA(df_space)).cuda()

                candidates, _ = optimize_acqf_discrete(
                    acq_function=ucb_qkg_acqf,
                    q=self.q_num,
                    choices=normalize(all_descs, bounds),
                    max_batch_size=self.bs,
                    unique=True,
                )
                get_candidates_pred(model_ucb, candidates, fn_dict)
                new_x_ucb = unnormalize(candidates.detach(), bounds=bounds)

                train_x_ucb = torch.cat([train_x_ucb, new_x_ucb])
                print("New Samples--------------------------------------------")
                recommend_descs = train_x_ucb[-self.q_num]
                torch.cuda.empty_cache()
                distmin_idx = compute_L2dist(recommend_descs, all_descs)
                save_recommend_comp(distmin_idx, df_space, recommend_descs, 
                                    df_space_path=self.df_space_path)

    def run_MOBO_batches(self):
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
                train_x_qehvi, train_obj_qehvi = split_for_val(X, y, ini_size=self.split_ratio)
            else:
                train_x_qehvi, train_obj_qehvi = X, y
            train_x_random, train_obj_random = train_x_qehvi, train_obj_qehvi

            mll_qehvi, model_qehvi = initialize_model(train_x_qehvi, 
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
                    model=model_qehvi, train_X=train_x_qehvi, train_obj=train_obj_qehvi, sampler=new_sampler, num_restarts=self.num_restarts, 
                    q_num=self.q_num, bounds=bounds, raw_samples=self.mc_samples_num,
                    ref_point_=ref_point, all_descs=X, max_batch_size=self.bs,
                    all_y=y, validate=True,
                    fn_dict=fn_dict,
                )

                new_x_random, new_obj_random = optimize_and_get_random_observation(X, y, X_now=train_x_random,
                                                                                    n=self.q_num)

                train_x_qehvi = torch.cat([train_x_qehvi, new_x])
                train_obj_qehvi = torch.cat([train_obj_qehvi, new_obj])
                train_x_random = torch.cat([train_x_random, new_x_random])
                train_obj_random = torch.cat([train_obj_random, new_obj_random])
                
                print("--------------------------------------------")
                recommend_descs = train_x_qehvi[-self.q_num:]
                hvs_qehvi.append(self.compute_hv(hv, train_obj_qehvi))
                hvs_random.append(self.compute_hv(hv, train_obj_random))

                mll_qehvi, model_qehvi = initialize_model(train_x_qehvi, train_obj_qehvi, bounds, lengthscale=Interval(0.01, self.ker_lengthscale_upper))

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

    def run_SOBO_batches(self):
        df_space = pd.read_pickle(self.df_space_path)
        df_space.reset_index(drop=True, inplace=True)

        train_x_ucb, train_obj_ucb, bounds, _ = self.init_experiment_input(X=self.X_train, y=self.y_train, ref_point=self.ref_point)

        for trial in range(1, 2):
            print(f"\nTrial {trial:>2}", end="\n")
            mll_ucb, model_ucb = initialize_model(train_x_ucb, train_obj_ucb, bounds, 
                                                      lengthscale=Interval(0.01, self.ker_lengthscale_upper))
            ucb_qkg_acqf = qUpperConfidenceBound(model_ucb, beta=self.beta)

            for __ in range(1, 2):
                fit_gpytorch_mll(mll_ucb)
                new_sampler = SearchSpace_Sampler(fn_dict, df_space, self.mc_samples_num)
                all_descs = torch.DoubleTensor(new_sampler.PCA(df_space)).cuda()

                candidates, _ = optimize_acqf_discrete(
                    acq_function=ucb_qkg_acqf,
                    q=self.q_num,
                    choices=normalize(all_descs, bounds),
                    max_batch_size=self.bs,
                    unique=True,
                )
                get_candidates_pred(model_ucb, candidates, fn_dict)
                new_x_ucb = unnormalize(candidates.detach(), bounds=bounds)

                train_x_ucb = torch.cat([train_x_ucb, new_x_ucb])
                print("New Samples--------------------------------------------")
                recommend_descs = train_x_ucb[-self.q_num]
                torch.cuda.empty_cache()
                distmin_idx = compute_L2dist(recommend_descs, all_descs)
                save_recommend_comp(distmin_idx, df_space, recommend_descs, 
                                    df_space_path=self.df_space_path)


if __name__ == '__main__':
    # Example usage:
    model = MLModel(X_train, y_train, ref_point, q_num, bs, 
                    mc_samples_num, num_restarts, split_ratio, ker_lengthscale_upper, df_space_path, beta)
    model.MOBO_one_batch()
    model.SOBO_one_batch()
    model.run_MOBO_batches()
    model.run_SOBO_batches()
