from quinine import (
    tstring,
    tinteger,
    tfloat,
    tboolean,
    stdict,
    tdict,
    default,
    required,
    allowed,
    nullable,
    tlist,
    stlist
)
from funcy import merge


model_schema = {
    "family": merge(tstring, allowed(["gpt2", "lstm"])),
    "n_positions": merge(tinteger, required),  # maximum context length
    "n_dims": merge(tinteger, required),  # latent dimension
    "n_embd": merge(tinteger, required),
    "n_layer": merge(tinteger, required),
    "n_head": merge(tinteger, required),
    "num_classes": merge(tinteger, default(10)),
    
}

curriculum_base_schema = {
    "start": merge(tinteger, required),  # initial parameter
    "end": merge(tinteger, required),  # limit of final value
    "inc": merge(tinteger, required),  # how much to increment each time
    "interval": merge(tinteger, required),  # increment every how many steps
}

curriculum_schema = {
    "dims": stdict(curriculum_base_schema),
    "points": stdict(curriculum_base_schema),
}

n_embedding_schema = {
    "x": merge(tinteger, default(100)),
    "irrelevant": merge(tinteger, default(10)),
    "task": merge(tinteger, default(5)),
}

#stlist = ["scalar_projection"]

data_sampler_kwargs_schema = {
    "n_embedding": stdict(n_embedding_schema),
    "x_y_order": merge(tstring, allowed(["x-y", "y-x", "random"]), default("random")),
    "add_task_tokens": merge(tboolean, default(False)),
    "n_start_tokens": merge(tinteger, default(3)),
    "finite_start_tokens": merge(tboolean, default(False)),
    "task_list": merge(tlist, default([])),
    "min_out_degree": merge(tinteger, default(1)),
    "max_out_degree": merge(tinteger, default(3)),
    "x_add_ratio": merge(tfloat, default(0.4)),
    "y_add_ratio": merge(tfloat, default(0.4)),
}

TASK_LIST = [
    "linear_regression",
    "quadratic_regression",
    "sparse_linear_regression",
    "linear_classification",
    "relu_2nn_regression",
    "relu_multilayernn_regression",
    "relu_2nn_regression_non-ICL",
    "decision_tree",
    "scalar_regression",
    "autoregression",
    "linear_quadratic_regression",
    "sin_regression",
    "log_regression",
    "quadratic_sum",
    "linear_regression_keep_dim",
    "quadratic_regression_keep_dim",
    "multi_task",
    "cubic_regression",
    "sqrt_regression",
    "sigmoid_2nn_regression" ,
    "linear_regression_random_mapping",
    "random_int_mapping",
    "quadratic",
    "linear_regression_diversity",
    "power_regression"
]

training_schema = {
    "task": merge(tstring, allowed(TASK_LIST), default("linear_regression")),
    "multi_tasks":merge(tlist, default([])),
    "task_kwargs": merge(tdict, required),
    "num_tasks": merge(tinteger, nullable, default(None)),
    "num_training_examples": merge(tinteger, nullable, default(None)),
    "data": merge(tstring, allowed(["gaussian", "autoregression", "random_int", "gaussian_quadratic", "multi_data","pos_gaussian"])),
    "multi_datas":merge(tlist, default([])),
    "batch_size": merge(tinteger, default(64)),
    "learning_rate": merge(tfloat, default(3e-4)),
    "train_steps": merge(tinteger, default(1000)),
    "save_every_steps": merge(tinteger, default(1000)),  # how often to checkpoint
    "keep_every_steps": merge(tinteger, default(-1)),  # permanent checkpoints
    "resume_id": merge(tstring, nullable, default(None)),  # run uuid64
    "curriculum": stdict(curriculum_schema),
    "device": merge(tstring, allowed(["cuda:0","cuda:1","cuda:2","cuda:3","cuda:4","cuda:5","cuda:6","cuda:7","cuda:8","cuda:9"])),
    "ood_task": merge(tstring, default("linear_regression")),
    "data_sampler_kwargs": stdict(data_sampler_kwargs_schema)
}

wandb_schema = {
    "project": merge(tstring, default("in-context-training")),
    "entity": merge(tstring, default("in-context")),
    "notes": merge(tstring, default("")),
    "name": merge(tstring, nullable, default(None)),
    "log_every_steps": merge(tinteger, default(10)),
}

schema = {
    "out_dir": merge(tstring, required),
    "model": stdict(model_schema),
    "training": stdict(training_schema),
    "wandb": stdict(wandb_schema),
    "test_run": merge(tboolean, default(False)),
}
