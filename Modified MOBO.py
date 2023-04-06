import torch
import numpy as np
from botorch.test_functions.multi_objective import BraninCurrin

# Random data for SingleRun
X_0=  [[0.8515, 0.3766, 0.8212, 0.9084, 0.3276, 0.7517],
        [0.6297, 0.0696, 0.2308, 0.4856, 0.8348, 0.6648],
        [0.3078, 0.5475, 0.8774, 0.3388, 0.1868, 0.3388],
        [0.8828, 0.6211, 0.6741, 0.5662, 0.9072, 0.8055],
        [0.6596, 0.7227, 0.5129, 0.8573, 0.4443, 0.9016],
        [0.5880, 0.8788, 0.6542, 0.4132, 0.5447, 0.1844],
        [0.1641, 0.0780, 0.9422, 0.1515, 0.0069, 0.8248],
        [0.3790, 0.8004, 0.2205, 0.1577, 0.9179, 0.2594],
        [0.1792, 0.5189, 0.1543, 0.1247, 0.9504, 0.8382],
        [0.9152, 0.8470, 0.1418, 0.2757, 0.0469, 0.8386]]
Y_0= [[-7.6157,   -7.6663],
        [ -10.9987,   -6.9429],
        [-7.9620,   -3.4366],
        [-5.7011,  -10.1440],
        [-5.4353,   -8.9274],
        [-4.8603,   -6.8815],
        [-13.4208,   -5.0611],
        [-5.8837,   -3.9369],
        [-8.4311,   -4.8428],
        [-4.5770,   -6.1225]]

bounds=[[0., 0., 0., 0., 0., 0.],
        [1., 1., 1., 1., 1., 1.]]       #.shape=[2,6]

ref_point = [-20.,-10]

ref_point_ = torch.FloatTensor(ref_point)
bounds = torch.FloatTensor(bounds)
dim = len(X_0[1])
num_objectives = len(Y_0[1])

from botorch.models.gp_regression import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch.utils.transforms import unnormalize
from botorch.utils.sampling import draw_sobol_samples

def generate_initial_data(n=dim):
    # generate training data
    train_x = torch.FloatTensor(X_0)
    train_obj = torch.FloatTensor(Y_0)
    return train_x, train_obj

def initialize_model(train_x, train_obj):
    # define models for objective and constraint
    model = SingleTaskGP(train_x, train_obj, outcome_transform=Standardize(m=train_obj.shape[-1]))
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    return mll, model

from botorch.optim.optimize import optimize_acqf, optimize_acqf_list
from botorch.acquisition.objective import GenericMCObjective
from botorch.utils.multi_objective.scalarization import get_chebyshev_scalarization
from botorch.utils.multi_objective.box_decompositions import NondominatedPartitioning
from botorch.acquisition.multi_objective.monte_carlo import qExpectedHypervolumeImprovement
from botorch.utils.sampling import sample_simplex

BATCH_SIZE = 2

standard_bounds = torch.zeros(2, dim)
standard_bounds[1] = 1


def optimize_qehvi_and_get_observation(model, train_obj, sampler):
    """Optimizes the qEHVI acquisition function, and returns a new candidate and observation."""
    partitioning = NondominatedPartitioning(ref_point=ref_point_, Y=train_obj)
    acq_func = qExpectedHypervolumeImprovement(
        model=model,
        ref_point=ref_point_.tolist(), 
        partitioning=partitioning,
        sampler=sampler,
    )
    candidates, _ = optimize_acqf(
        acq_function=acq_func,
        bounds=standard_bounds,
        q=BATCH_SIZE,
        num_restarts=20,
        raw_samples=1024,
        options={"batch_limit": 5, "maxiter": 200, "nonnegative": True},
        sequential=True,
    )
    new_x =  unnormalize(candidates.detach(), bounds=bounds)
    new_obj = torch.FloatTensor([[  -6.7064,   -5.8886],        #？为啥不用根据new_x去计算new_obj
        [ -51.7423,   -6.8102],
        [ -38.3063,   -6.8469],
        [ -13.4827,   -9.0434],
        [ -10.3850,  -10.6817],
        [ -27.7399,   -6.6023],
        [ -64.7528,   -2.1669],
        [-168.0079,   -4.3890],
        [ -17.1416,  -10.4511],
        [  -7.0856,   -5.5974]])        #.shape = [10,2]

    return new_x, new_obj


from botorch import fit_gpytorch_model
from botorch.acquisition.monte_carlo import qExpectedImprovement, qNoisyExpectedImprovement
from botorch.sampling import SobolQMCNormalSampler
from botorch.exceptions import BadInitialCandidatesWarning
from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.utils.multi_objective.hypervolume import Hypervolume
import warnings
warnings.filterwarnings('ignore', category=BadInitialCandidatesWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

N_TRIALS = 1
N_BATCH = 1
MC_SAMPLES = 1024

verbose = True

hvs_qehvi_all = []

hv = Hypervolume(ref_point = ref_point_)

# average over multiple trials
for trial in range(1, N_TRIALS + 1):
    torch.manual_seed(trial)
    
    print(f"\nTrial {trial:>2} of {N_TRIALS} ", end="\n")
    hvs_qehvi = []
    train_x_qehvi, train_obj_qehvi = generate_initial_data(n=6)
    mll_qehvi, model_qehvi = initialize_model(train_x_qehvi, train_obj_qehvi)
    

    pareto_mask = is_non_dominated(train_obj_qehvi)
    pareto_y = train_obj_qehvi[pareto_mask]

    volume = hv.compute(pareto_y)
    hvs_qehvi.append(hvs_qehvi)
   
    print("Hypervolume is ", volume)
    # run N_BATCH rounds of BayesOpt after the initial random batch
    for iteration in range(1, N_BATCH + 1):    
    
        fit_gpytorch_model(mll_qehvi)
        qehvi_sampler = SobolQMCNormalSampler(MC_SAMPLES)

        new_x_qehvi, new_obj_qehvi = optimize_qehvi_and_get_observation(
            model_qehvi, train_obj_qehvi, qehvi_sampler
        )      
        
        # update training points
        train_x_qehvi = torch.cat([train_x_qehvi, new_x_qehvi])
        train_obj_qehvi = torch.cat([train_obj_qehvi, new_obj_qehvi])
        print("New Samples--------------------------------------------")
        print(train_x_qehvi[-BATCH_SIZE:])



