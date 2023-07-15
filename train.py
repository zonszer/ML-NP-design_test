from utils.utils_ import *
import pandas as pd
import time
import glob
from PCE_analysis import logger
from preprocessing import Preprocessing, Add_extract_descriptors

import torch
from botorch import fit_gpytorch_mll
from botorch.acquisition.monte_carlo import qExpectedImprovement, qNoisyExpectedImprovement, qUpperConfidenceBound
# from botorch.acquisition import qExpectedImprovement, qNoisyExpectedImprovement
from botorch.sampling import SobolQMCNormalSampler
from botorch.utils.multi_objective.hypervolume import Hypervolume
from botorch.utils.multi_objective.box_decompositions.dominated import DominatedPartitioning

from botorch.optim.optimize import optimize_acqf, optimize_acqf_list
from botorch.acquisition.multi_objective.monte_carlo import qExpectedHypervolumeImprovement, \
    qNoisyExpectedHypervolumeImprovement
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
                 X_train_o, y_train_o,
                 ref_point,
                 q_num, bs,
                 mc_samples_num,
                 ker_lengthscale_upper,
                 PCA_preprocessor,
                 df_space_path=None,
                 **kwargs):

        self.X_train_o = X_train_o
        self.y_train_o = y_train_o
        self.ref_point = ref_point
        self.q_num = q_num
        self.bs = bs
        self.mc_samples_num = mc_samples_num
        self.ker_lengthscale_upper = ker_lengthscale_upper
        self.df_space_path = df_space_path
        self.PCA_preprocessor = PCA_preprocessor
        if df_space_path is not None:
            self.df_space = pd.read_pickle(df_space_path)
            self.df_space.reset_index(drop=True, inplace=True)
        for key, value in kwargs.items():
            setattr(self, key, value)
            # self.beta = kwargs['beta'] 'split_ratio'
            # self.save_file_instance = kwargs['save_file_instance']
            # self.num_restarts = kwargs['num_restarts']

    def generate_bounds(self, X, scale=(0, 1)):
        bounds = np.zeros((2, X.shape[1]))
        for i in range(X.shape[1]):
            bounds[0][i] = X[:, i].min() * 1. # min of bound   #TODO: can not define the bound of the problem deterministically
            bounds[1][i] = X[:, i].max() * 1.  # max of bound
        bounds = torch.DoubleTensor(bounds).to(self.device)
        return bounds

    def generate_initial_data(self, X, y):
        '''generate training data'''
        X_trans, y_trans = self.PCA_preprocessor.transform_fn_PCA(X, y)
        train_x = torch.DoubleTensor(X_trans).to(self.device)
        train_obj = torch.DoubleTensor(y_trans).to(self.device)
        return train_x, train_obj

    def generate_ref_point(self, y):
        '''generate reference point from denoted str or data'''
        if isinstance(self.ref_point, type('')):
            rp = eval(self.ref_point) if isinstance(self.ref_point, type('')) else None
            rp = torch.DoubleTensor(rp).to(self.device)
        else:
            if not isinstance(y, type(torch.Tensor(0))):
                y = torch.DoubleTensor(y).to(self.device)
            rp = torch.min(y, axis=0)[0]
        return rp

    def generate_search_space(self, data):
        assert isinstance(data, pd.DataFrame)   
        idx = data.shape[1] - 132  # changed param1
        desc = data.iloc[:, idx:].values
        return desc, None

    def init_experiment(self, X, y, ref_point):
        """Initialize experiment (adaptive generate ref_point and bounds of the problem)."""
        X_trans, y_trans = self.PCA_preprocessor.transform_fn_PCA(X, y)
        bounds = self.generate_bounds(X_trans, scale=(0, 1))
        ref_point_ = self.generate_ref_point(y_trans)
        bd = DominatedPartitioning(ref_point=ref_point_, Y=torch.DoubleTensor(y_trans).to(self.device))
        volume = bd.compute_hypervolume().item()
        logger.log(logging.INFO, f"Dataset Hypervolume is {volume}", color='GREEN')
        return bounds, ref_point_

    def optimize_qehvi_and_get_observation(self, model, train_X, train_obj, sampler, num_restarts,
                                           q_num, bounds, raw_samples, ref_point_,
                                           all_descs, max_batch_size, iter_num,
                                           validate=False, all_y=None):
        """Optimizes the qEHVI acquisition function, and returns a new candidate and observation."""
        with torch.no_grad():  # TODO: check if grad is removed
            pred = model.posterior(normalize(train_X, bounds)).mean
        partitioning = FastNondominatedPartitioning(ref_point=ref_point_, Y=pred)
        acq_func = qExpectedHypervolumeImprovement(
            model=model,
            ref_point=ref_point_.tolist(),
            partitioning=partitioning,
            sampler=sampler,
        )
        candidates, acq_value = optimize_acqf_discrete(
            acq_function=acq_func,
            q=q_num,
            choices=normalize(all_descs, bounds),
            max_batch_size=max_batch_size,
            unique=True,
        )
        self.y_pred, self.y_predVar = self.get_y_pred(model, candidates.detach(), iter_num)

        new_x = unnormalize(candidates.detach(), bounds=bounds)
        if validate:
            assert all_y is not None
            new_item_dict, new_item_idx = self.get_idx_and_corObj(new_x, all_descs, all_y=all_y)
            new_y = new_item_dict['all_y']
        else:
            assert all_y is None
            new_item_dict, new_item_idx = self.get_idx_and_corObj(new_x, all_descs)
            new_y = None
        logger.log(logging.INFO, f'new item idxs are: {new_item_idx}')
        return new_x, new_y, new_item_idx

    def optimize_qnehvi_and_get_observation(self, model, train_X, train_obj, sampler, num_restarts,
                                           q_num, bounds, raw_samples, ref_point_,
                                           all_descs, max_batch_size, iter_num,
                                           validate=False, all_y=None):
        """Optimizes the qEHVI acquisition function, and returns a new candidate and observation."""
        acq_func = qNoisyExpectedHypervolumeImprovement(
            model=model,
            ref_point=ref_point_.tolist(),
            X_baseline=normalize(train_X, bounds),
            prune_baseline=True,
            sampler=sampler,
        )
        candidates, acq_value = optimize_acqf_discrete(
            acq_function=acq_func,
            q=q_num,
            choices=normalize(all_descs, bounds),
            max_batch_size=max_batch_size,
            unique=True,
        )
        self.y_pred, self.y_predVar = self.get_y_pred(model, candidates.detach(), iter_num)

        new_x = unnormalize(candidates.detach(), bounds=bounds)
        if validate:
            assert all_y is not None
            new_item_dict, new_item_idx = self.get_idx_and_corObj(new_x, all_descs, all_y=all_y)
            new_y = new_item_dict['all_y']
        else:
            assert all_y is None
            new_item_dict, new_item_idx = self.get_idx_and_corObj(new_x, all_descs)
            new_y = None
        logger.log(logging.INFO, f'new item idxs are: {new_item_idx}')
        return new_x, new_y, new_item_idx

    def generate_y_fromFile(self, desc_ofNewX, iter):
        if self.iter_files != []:
            file_path = self.iter_files[iter]
            df = Preprocessing(file_path)
            descs = Add_extract_descriptors(df)
            file_descs = descs.values              
            new_y_train = df[self.y_col_names].values  
            new_item_dict, new_item_idx = self.get_idx_and_corObj(desc_ofNewX, file_descs, new_y_train=new_y_train)   #  TODO: check
            new_y_train_ = new_item_dict['new_y_train']
            logger.log(logging.INFO, f'new item idxs in iter_data files are: {new_item_idx}')
        else:
            new_y_train_ = np.full((self.q_num, len(self.y_col_names)), np.nan, dtype=np.float64)
        return new_y_train_

    def update_Xy(self, new_item_idx, is_validate, iteration):  # new_item_idx are relative idxs with the current X_remain
        '''update X and y by removing new_item_idx in X_remain and add it to X_train'''
        maskRow_X = np.ones(self.X_remain.shape[0], dtype=bool)
        maskRow_X[new_item_idx] = False
        # indices_X = np.nonzero(mask_X)  #get the index of non zero (True) elements of the mask
        # indices_y = np.nonzero(mask_y)
        new_X_train = np.concatenate((self.X_train, self.X_remain[new_item_idx]), axis=0)
        if is_validate:
            new_y_train = np.concatenate((self.y_train, self.y_remain[new_item_idx]), axis=0)
            new_X_remain, new_y_remain = self.X_remain[maskRow_X], self.y_remain[maskRow_X]
        else:
            new_y = self.generate_y_fromFile(desc_ofNewX=self.X_remain[new_item_idx], iter=iteration)
            new_y_train = np.concatenate((self.y_train, new_y), axis=0)
            new_X_remain, new_y_remain = self.X_remain[maskRow_X], None
        # new_X_remain = np.ma.array(self.X_remain, mask=~mask_X).filled(fill_value=np.NaN) #still the original shape
        # new_y_remain = np.ma.array(self.y_remain, mask=~mask_y).filled(fill_value=np.NaN)
        return new_X_remain, new_y_remain, new_X_train, new_y_train

    def optimize_and_get_random_observation(self, X_r, y_r, q_num, model, iter_num):
        '''generate random data from X_r and y_r'''
        # _, X_now_idx = self.get_idx_and_corObj(X_now, X)
        # mask = np.ones_like(X.cpu().numpy(), dtype=bool);
        # mask_y = np.ones_like(y.cpu().numpy(), dtype=bool)
        # mask[X_now_idx] = False
        # mask_y[X_now_idx] = False
        data_num = X_r.shape[0]
        random_elements = np.random.choice(range(data_num), q_num, replace=False)
        mask = torch.ones(data_num, dtype=torch.bool)
        random_elements_t = torch.tensor(random_elements).to(self.device)
        mask[random_elements_t] = False
        X_new = X_r[random_elements_t, :]  # shape=(125, 18)
        y_new = y_r[random_elements_t, :]
        candidates = normalize(X_new, self.bounds)
        self.y_pred, self.y_predVar = self.get_y_pred(model, candidates, iter_num)
        return X_new, y_new, random_elements_t.cpu().numpy()

    def get_y_pred(self, model, candidates, iter_num):
        if iter_num == 0:
            y_pred, y_predVar = self.get_candidates_pred(model, normalize(self.X_trainTrans, self.bounds))
            new_y_pred, new_y_predVar = self.get_candidates_pred(model, candidates)
            y_pred = np.concatenate([y_pred, new_y_pred], axis=0)
            y_predVar = np.concatenate([y_predVar, new_y_predVar], axis=0)
        else:
            new_y_pred, new_y_predVar = self.get_candidates_pred(model, candidates)
            y_pred = np.concatenate([self.y_pred, new_y_pred], axis=0)
            y_predVar = np.concatenate([self.y_predVar, new_y_predVar], axis=0)

        return y_pred, y_predVar

    def get_candidates_pred(self, model, candidates):
        with torch.no_grad():  # TODO: check if grad is removed
            pred_mean = model.posterior(candidates).mean.detach().cpu().numpy()
            pred_var = model.posterior(candidates).variance.detach().cpu().numpy()
            pred_mean_ = self.PCA_preprocessor.pre_fndict["fn_for_y"](pred_mean, inverse_transform=True)
            pred_var_ = self.PCA_preprocessor.pre_fndict["fn_for_y"](pred_var, inverse_transform=True)
            # np.savetxt("pred_meanORE.csv", pred_mean, delimiter=",")
        return pred_mean_, pred_var_

    def get_idx_and_corObj(self, new_x, all_descs, **kwargs):
        '''generate idxs of new_x and can alos return the corresponding obj(if the obi is passed in kwargs)'''
        distmin_idx = self.compute_L2dist(new_x, all_descs)
        for key in kwargs.keys():
            kwargs[key] = kwargs[key][distmin_idx]
        return kwargs, distmin_idx

    # def compute_hv(self, hv, train_obj_qehvi):
    #     pareto_mask = is_non_dominated(train_obj_qehvi)
    #     pareto_y = train_obj_qehvi[pareto_mask]
    #     volume = hv.compute(pareto_y)
    #     return volume

    def transform_PCA_fn(self, data, all_y, validate):
        """transform the descriptors of search space using the predefined PCA"""
        assert isinstance(data, np.ndarray)
        all_desc = torch.DoubleTensor(self.PCA_preprocessor.pre_fndict['fn_input'](data.copy())).to(self.device)    
        if validate:  
            assert all_y is not None
            all_y_ = torch.DoubleTensor(self.PCA_preprocessor.pre_fndict['fn_for_y'](all_y.copy())).to(self.device)     #TODO: 神奇的bug here
        else:
            assert all_y is None
            all_y_ = None
        return all_desc, all_y_

    def initialize_model(self, train_x, train_obj, lengthscale, state_dict=None):
        assert train_x.shape[0] == train_obj.shape[0]
        train_x, train_obj = self.generate_initial_data(X=train_x, y=train_obj)
        bounds_current = self.generate_bounds(train_x, scale=(0, 1))
        ref_point_current = self.generate_ref_point(train_obj)
        ker = MaternKernel(nu=2.5, ard_num_dims=normalize(train_x, bounds_current).shape[-1],
                           lengthscale_constraint=lengthscale).to(self.device)
        ker = ScaleKernel(ker)
        model = SingleTaskGP(normalize(train_x, bounds_current), train_obj, #covar_module=ker,
                             outcome_transform=Standardize(m=train_obj.shape[-1]))
        # model_parameters = model.state_dict()
        if state_dict is not None:
            model.load_state_dict(state_dict)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        return mll, model, train_x, train_obj, bounds_current, ref_point_current

    def output_exploreSeq(self, y_seq, trial):
        '''Output the explored sequence to a CSV file with different labels to denote iters and corresponding column names'''
        iter_idx = 1
        df = pd.DataFrame(columns=['original index in excel', 'y1', 'y2', 'y1_pred', 'y2_pred',
                                   'y1_predVar', 'y2_predVar', 'iter', 'seed'])
        for i, element in enumerate(y_seq):
            # Use np.all to compare entire rows
            index = np.where((self.y_original_seq == element).all(axis=1))
            if index[0].size == 1:
                index_val = index[0][0]  # taking the first occurrence of the element
                df.loc[i, 'original index in excel'] = int(index_val)  # Assuming index of a 2D array
            else:
                logger.log(logging.WARNING, 'Element not found or repeated elements in self.y_original_seq', color="YELLOW")

            df.loc[i, 'y1'] = element[0]
            df.loc[i, 'y2'] = element[1]
            df.loc[i, 'y1_pred'] = self.y_pred[i][0]
            df.loc[i, 'y2_pred'] = self.y_pred[i][1]
            df.loc[i, 'y1_predVar'] = self.y_predVar[i][0]
            df.loc[i, 'y2_predVar'] = self.y_predVar[i][1]

            if i + 1 <= self.init_num:
                df.loc[i, 'iter'] = 0
            else:
                if (i + 1 - self.init_num) % self.q_num != 0:
                    df.loc[i, 'iter'] = iter_idx
                else:
                    df.loc[i, 'iter'] = iter_idx
                    iter_idx += 1
        # df.to_csv(f"data/explored_sequence/explored_sequence_inMOBO_batches-seed{trial}.csv", index=True, header=True)
        df.loc[:, 'seed'] = trial
        return df

    def _validate_samples(self, new_x):
        '''Check if the samples generated by the acquisition function are within the bounds.
        If not, log a warning message.
        '''
        for i in range(new_x.shape[1]):
            if not torch.all(new_x[:, i] >= self.bounds[0][i]) or not torch.all(new_x[:, i] <= self.bounds[1][i]):
                logger.log(logging.WARNING, 
                           f'Warning: Some of the generated samples are outside the bounds for dimension {i}', color='YELLOW')
    def _check_iterData(self):
        dir_ = os.path.join(os.path.dirname(self.df_space_path), 'iter_data')
        files = glob.glob(f'{dir_}/*.excel')
        logger.log(logging.INFO, f'Found {len(files)} files in {dir_}: {files}')
        return files
    

    def MOBO_batches(self, mode="qNEHVI", is_validate=True, N_TRIALS=1, MAX_N_BATCH=100):
        verbose = True
        self.init_num = self.y_train_o.shape[0]
        self.df_saveList = []
        # ================== check it is validation mode or usage mode: =================
        if is_validate:
            assert self.y_train_o.shape[0] <= self.y_original_seq.shape[0]
            self.bounds, self.ref_point = self.init_experiment(
                                            X=np.concatenate((self.X_train_o, self.X_remain_o)),
                                            y=np.concatenate((self.y_train_o, self.y_remain_o)),
                                            ref_point=self.ref_point,
                                          )
        else:
            self.X_remain_o, self.y_remain_o = self.generate_search_space(data=self.df_space)
            self.iter_files = self._check_iterData()
        for trial in range(0, N_TRIALS):
            logger.log(logging.INFO, f"\nTrial {trial:>2} of {N_TRIALS-1} \n")
            self.X_remain, self.y_remain = self.X_remain_o, self.y_remain_o
            self.X_train, self.y_train = self.X_train_o, self.y_train_o
            hvs_qehvi = []
            mll_qehvi, model_qehvi, self.X_trainTrans, self.y_trainTrans, \
            self.bounds, self.ref_point = self.initialize_model(
                train_x=self.X_train,
                train_obj=self.y_train,
                lengthscale=Interval(0.01, self.ker_lengthscale_upper)
            )
            bd = DominatedPartitioning(ref_point=self.ref_point, Y=self.y_trainTrans)
            volume = bd.compute_hypervolume().item()
            hvs_qehvi.append(volume)
            logger.log(logging.INFO, f"Init Hypervolume is {volume}", color='GREEN')
            logger.log(logging.INFO, "\n--------------start batches--------------\n", color='green'.upper())

            for iteration in range(0, MAX_N_BATCH):
                t0 = time.monotonic()
                # ================== apply method and update trainXy: =================
                all_X, all_y = self.transform_PCA_fn(data=self.X_remain, all_y=self.y_remain, validate=is_validate)
                fit_gpytorch_mll(mll_qehvi)
                if mode == "random":
                    new_x, new_obj, new_item_idx = self.optimize_and_get_random_observation(all_X,
                                                                                            all_y, 
                                                                                            q_num=self.q_num,
                                                                                            model=model_qehvi,
                                                                                            iter_num=iteration)
                else:
                    # model_qehvi.covar_module.base_kernel.lengthscale
                    new_sampler = SobolQMCNormalSampler(sample_shape=torch.Size([self.mc_samples_num]))
                    new_x, new_obj, new_item_idx = self.optimize_qnehvi_and_get_observation(
                        model=model_qehvi,
                        train_X=self.X_trainTrans, train_obj=self.y_trainTrans,
                        sampler=new_sampler,
                        num_restarts=self.num_restarts,
                        q_num=self.q_num, bounds=self.bounds,
                        raw_samples=self.mc_samples_num,
                        ref_point_=self.ref_point,
                        max_batch_size=self.bs,
                        all_descs=all_X, all_y=all_y,
                        validate=is_validate,
                        iter_num=iteration,
                    )
                self._validate_samples(new_x)
                self.X_remain, self.y_remain, self.X_train, self.y_train = self.update_Xy(new_item_idx,
                                                                                          is_validate, iteration)
                # ================== re-init models: ==================(when not validate self.y_remain == None all the time)
                if is_validate or (not is_validate and self.iter_files != []):
                    mll_qehvi, model_qehvi, self.X_trainTrans, self.y_trainTrans, \
                    self.bounds, self.ref_point = self.initialize_model(
                        train_x=self.X_train,
                        train_obj=self.y_train,
                        lengthscale=Interval(0.01, self.ker_lengthscale_upper)
                    )
                # ====================== summary: ======================
                # self.X_trainTrans = torch.cat([self.X_trainTrans, new_x])
                # self.y_trainTrans = torch.cat([self.y_trainTrans, new_obj])
                # recommend_descs_NewTrans = self.X_trainTrans[-self.q_num:]  #is new desc after new trans
                bd = DominatedPartitioning(ref_point=self.ref_point, Y=self.y_trainTrans)
                volume = bd.compute_hypervolume().item()
                hvs_qehvi.append(volume)
                # torch.cuda.empty_cache()
                t1 = time.monotonic()
                if verbose:
                    logger.log(logging.INFO,
                        f"summary: Batch {iteration:>2}: Hypervolume of {mode} after added new items = "
                        f"{hvs_qehvi[-1]:>4.2f} "
                        f"\ntime = {t1 - t0:>4.2f}.",
                    )
                    logger.log(logging.INFO, "---------------------------------")
                    if not is_validate:
                        self.save_recommend_comp(new_item_idx, self.df_space, new_x, iter=f'iter_{iteration}')
                # ==================if break loop:==================
                if (is_validate and self.X_remain.shape[0] == 0) or (not is_validate and len(self.iter_files) <= iteration):    #self.y_train.shape[0] == self.y_original_seq.shape[0]
                    assert self.y_pred.shape[0] == self.y_train.shape[0] 
                    df_seed = self.output_exploreSeq(self.y_train, trial)
                    self.df_saveList.append(df_seed)
                    break
        if is_validate:
            merged_df = pd.concat(self.df_saveList)
            merged_df.to_csv(f"data/explored_sequence/MOBO_batches-{mode}.csv", index=True, header=True)

    def SOBO_one_batch(self):
        train_x_ucb, train_obj_ucb, bounds, _ = self.init_experiment(X=self.X_train, y=self.y_train,
                                                                     ref_point=self.ref_point)

        for trial in range(1, 2):
            logger.log(logging.INFO, f"\nTrial {trial:>2} \n", color='PURPLE')
            mll_ucb, model_ucb = self.initialize_model(train_x_ucb, train_obj_ucb, bounds,
                                                       lengthscale=Interval(0.01, self.ker_lengthscale_upper))
            ucb_qkg_acqf = qUpperConfidenceBound(model_ucb, beta=self.beta)

            for __ in range(1, 2):
                fit_gpytorch_mll(mll_ucb)
                all_descs = torch.DoubleTensor(self.transform_PCA_fn(data=self.df_space)).to(self.device)

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
                logger.log(logging.INFO, "New Samples------------------------------")
                recommend_descs = train_x_ucb[-self.q_num:]
                torch.cuda.empty_cache()
                distmin_idx = self.compute_L2dist(recommend_descs, all_descs)
                self.save_recommend_comp(distmin_idx, self.df_space, recommend_descs,
                                         df_space_path=self.df_space_path)

    def save_recommend_comp(self, idx, df_space, recommend_descs, all_descs=None, iter=None):
        str1 = get_str_after_substring(self.df_space_path, 'Ru')
        df_space.iloc[idx, :].to_csv(f"data/iter_record/recommend_comp{str1}-{iter}.csv", index=True, header=True)
        logger.log(logging.INFO, f'new samples:\n{df_space.iloc[idx, 0:10]}')
        df = pd.DataFrame(recommend_descs.cpu().numpy())
        df.to_csv(f"data/iter_record/recommend_descs{str1}-{iter}.csv", index=True, header=False)
        if all_descs is not None:
            df_desc = pd.DataFrame(all_descs.cpu().numpy())
            df_desc.to_csv(f"data/iter_record/all_PCAdescs{str1}-{iter}.csv", index=True, header=False)

    def compute_L2dist(self, target_obj, space):
        dm = torch.cdist(target_obj, space)
        dist_min, distmin_idx = dm.min(dim=1)
        if dist_min.min() > 1e-4:
            logger.log(logging.WARNING, "Warning: the distance between the recommended and the actual is too large, please check it!", color='red'.upper())
        return distmin_idx.cpu().numpy()
