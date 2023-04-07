import pandas as pd
from ax import optimize
from ax.core.objective import MultiObjective
from ax.core.optimization_config import MultiObjectiveOptimizationConfig
from ax.core.parameter import FixedParameter, RangeParameter
from ax.metrics.multi_objective import Hypervolume
from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy, get_botorch

# Load external dataset
data_df = pd.read_csv("external_data.csv")

# Create Ax experiment with fixed data
parameter_names = ["param1", "param2", "param3"]
objective_names = ["objective1", "objective2"]
fixed_values = [0.2, 0.4]  # values of fixed parameters in external dataset
factor_levels = [0, 1]  # levels of parameters to be optimized
search_space = [
    FixedParameter(name, value) if name in fixed_values for name in parameter_names
]
parameters, objectives = [], []

for name in parameter_names:
    if name in fixed_values:
        parameters.append(FixedParameter(name, fixed_values[parameter_names.index(name)]))
    else:
        parameters.append(RangeParameter(name, lower=0, upper=1))

for name in objective_names:
    objectives.append(
        optimize.Objective(
            name=name,
            metric=Hypervolume(
                # Add reference point for hypervolume calculation
                reference_point=[1.1] * len(objective_names)
            ),
            minimize=False if name == "objective1" else True,
        )
    )
    for i, row in data_df.iterrows():
        exp.add_trial(
            optimize.Trial(
                arms=[
                    optimize.arm.Arm(
                        parameters={
                            parameter_names[i]: (x if parameter_names[i] not in fixed_values else fixed_values[parameter_names.index(name)]),
                        },
                    )
                ],
            ),
        )

opt_config = MultiObjectiveOptimizationConfig(
    objective=MultiObjective(objectives),
    acquisition_function_type="qehvi",
    acquisition_function_kwargs={"num_outer_samples": 32},
    num_initialization_trials=3,
    max_trials=10,
    surrogate_model=get_botorch(
        Experiment(search_space=search_space, name="MOBO", optimization_config=opt_config)
    ),
)
gen_steps = [
    GenerationStep(
        model=get_botorch(
            Experiment(
                search_space=search_space, name="MOBO", optimization_config=opt_config
            )
        ),
        num_trials=1,
    )
    for _ in range(10)
]
gs = GenerationStrategy(gen_steps=gen_steps)
br = optimize.run_experiment(experiment=exp, optimization_config=opt_config, generation_strategy=gs)
# Get the last trial data and append it to the external dataset
trial_data = exp.fetch_trial_data(br._experiment.trials[-1].index)
data_df = data_df.append(pd.DataFrame(trial_data.arm_predictions), ignore_index=True)
