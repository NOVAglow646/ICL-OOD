import json
import os
import sys
import os


from munch import Munch
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import yaml

import models
from samplers import get_data_sampler, sample_transformation
from tasks import get_task_sampler
global_conf=None
import os

def get_model_from_run(run_path, step=-1, only_conf=False, random_init_model=False, device="cuda:0"):
    config_path = os.path.join(run_path, "config.yaml")
    #print(config_path, 66)
    with open(config_path) as fp:  # we don't Quinfig it to avoid inherits
        conf = Munch.fromDict(yaml.safe_load(fp))
        global_conf = conf
    if only_conf:
        return None, conf
    #print(conf)
    model = models.build_model(conf.model)
    if random_init_model:
        return model, conf

    if step == -1:
        state_path = os.path.join(run_path, "state.pt")
        state = torch.load(state_path, map_location=device)
        #print(state["model_state_dict"])
        model.load_state_dict(state["model_state_dict"],strict=False)
    else:
        model_path = os.path.join(run_path, f"model_{step}.pt")
        
        state_dict = torch.load(model_path, map_location=device)
        #print(state_dict)
        model.load_state_dict(state_dict,strict=False)

    return model, conf




# Functions for evaluation


def eval_batch(model, task_sampler, xs, xs_p=None, device="cuda:0", prediction_collector=None, data_name=None):
    task = task_sampler()
    #print("task66",task)
    if torch.cuda.is_available() and model.name.split("_")[0] in ["gpt2", "lstm"]:
        device = device
    else:
        device = "cpu"
    #print("xs", xs.shape) # torch.Size([64, 101, 20])
    if xs_p is None:
        #print(77)
        if data_name == "random_int":
            #print(333)
            ys = task.evaluate(xs["xs_int_index"])
            #print("task.random_mappings",task.random_mappings)
            #print("xs", xs["xs_int_index"]) # xs torch.Size([64, 101, 7])
            #print("xs", xs["xs_embedded"].shape) # xs torch.Size([64, 101, 7])
            #print("ys", ys.shape) #ys torch.Size([64, 101, 7])
            pred = model(xs["xs_embedded"].to(device), ys.to(device)).detach()
        else:
            ys = task.evaluate(xs)
            pred = model(xs.to(device), ys.to(device)).detach()
        
        
        if prediction_collector is not None:
            prediction_collector.collect(xs, ys, pred)
        #print("pred", pred)
        #print("pred-ys", (pred.cpu()-ys).mean(-1))
        metrics = task.get_metric()(pred.cpu(), ys)

        if metrics.dim()==3: # the metrics size is [bsz, sequence_length, data_dim]
            metrics = torch.sum(metrics, dim=-1)
        #print(metrics) # torch.Size([64, 101])
        #print( ys.mean(0))
        #print(task.get_metric()) # <function squared_error at 0x7fb0162885e0>
        #print( task.get_metric()(torch.zeros_like(ys), ys).mean(0).shape)
    else:
        #print(88)
        b_size, n_points, _ = xs.shape
        metrics = torch.zeros(b_size, n_points)
        for i in range(n_points):
            #print(i)
            xs_comb = torch.cat((xs[:, :i, :], xs_p[:, i:, :]), dim=1)
            ys = task.evaluate(xs_comb)

            pred = model(xs_comb.to(device), ys.to(device), inds=[i]).detach()
            
            metrics[:, i] = task.get_metric()(pred.cpu(), ys)[:, i]

    return metrics


# Functions for generating different kinds of train/test data


def gen_standard(data_sampler, n_points, b_size):
    #print(n_points)
    seeds = torch.arange(b_size).tolist()
    xs = data_sampler.sample_xs(n_points, b_size, seeds=seeds)
    return xs, None

def gen_random_int(data_sampler, n_points, b_size):
    #print(n_points)
    xs_int_index, xs_embedded = data_sampler.sample_xs(n_points, b_size)
    return {"xs_int_index":xs_int_index, "xs_embedded":xs_embedded}, None

def gen_opposite_quadrants(data_sampler, n_points, b_size):
    xs = data_sampler.sample_xs(n_points, b_size)
    pattern = torch.randn([b_size, 1, xs.shape[2]]).sign()

    xs_train_pre = xs.abs() * pattern
    xs_test_post = -xs_train_pre

    return xs_train_pre, xs_test_post


def gen_random_quadrants(data_sampler, n_points, b_size):
    
    xs = data_sampler.sample_xs(n_points, b_size)
    pattern = torch.randn([b_size, 1, xs.shape[2]]).sign()

    xs_train_pre = xs.abs() * pattern
    xs_test_post = xs

    return xs_train_pre, xs_test_post


def gen_orthogonal_train_test(data_sampler, n_points, b_size):
    xs = data_sampler.sample_xs(n_points, b_size)
    n_dim = xs.shape[2]
    n_points = min(n_points, n_dim)
    # raise ValueError("number of points should be at most the dimension.")
    xs_train_pre = xs
    xs_test_post = torch.zeros(xs.shape)
    for i in range(n_points):
        xs_test_post_i = xs[:, i : i + 1, :]
        xs_train_pre_i = xs[:, :i, :]
        _, _, Vt = torch.linalg.svd(xs_train_pre_i, full_matrices=False)
        xs_train_pre_i_projection = Vt.transpose(1, 2) @ Vt
        xs_test_post_i_orthogonalized = (
            xs_test_post_i - xs_test_post_i @ xs_train_pre_i_projection
        )
        xs_test_post_i_normalized = (
            xs_test_post_i_orthogonalized
            * xs_test_post_i.norm(dim=2).unsqueeze(2)
            / xs_test_post_i_orthogonalized.norm(dim=2).unsqueeze(2)
        )

        xs_test_post[:, i : i + 1, :] = xs_test_post_i_normalized

    return xs_train_pre, xs_test_post


def gen_overlapping_train_test(data_sampler, n_points, b_size):
    xs = data_sampler.sample_xs(n_points, b_size)
    xs_train_pre = xs
    xs_test_post = xs.clone()
    b_size = xs.shape[0]
    for i in range(1, n_points):
        xs_train_pre_i = xs[:, :i, :]
        perm = torch.stack([torch.randperm(i) for _ in range(b_size)]).unsqueeze(dim=1)
        ind_mat = (perm == 0) + 0.0
        xs_test_post[:, i : i + 1, :] = ind_mat @ xs_train_pre_i

    return xs_train_pre, xs_test_post


def aggregate_metrics(metrics, bootstrap_trials=1000):
    """
    Takes as input a tensor of shape (num_eval, n_points) and returns a dict with
    per-point mean, stddev, and bootstrap limits
    """
    results = {}
    #print(metrics.shape) # [1280, 101]
    results["mean"] = metrics.mean(dim=0)
    results["std"] = metrics.std(dim=0, unbiased=True)
    n = len(metrics)
    bootstrap_indices = torch.randint(n, size=(bootstrap_trials, n))
    bootstrap_means = metrics[bootstrap_indices].mean(dim=1).sort(dim=0)[0]
    results["bootstrap_low"] = bootstrap_means[int(0.05 * bootstrap_trials), :]
    results["bootstrap_high"] = bootstrap_means[int(0.95 * bootstrap_trials), :]

    return {k: v.tolist() for k, v in results.items()}


def eval_model(
    pretrain_path,
    model,
    device,
    task_name,
    data_name,
    n_dims,
    n_points,
    prompting_strategy,
    num_eval_examples=1280,
    batch_size=64,
    data_sampler_kwargs={},
    task_sampler_kwargs={},
    collect_prediction=False,
    fixed_embedding=None, # use a fixed vocab for gaussian X
    rank=2,
    ood_task=None
):
    """
    Evaluate a model on a task with a variety of strategies.
       Args:
       - task: which base task we are evaluating on. E.g., "linear_regression"
       - prompting_strategy: how to construct the prompt, e.g., "random_quadrants"
       - num_eval_examples: total number of examples to evaluate on
       - **sampler_kwargs: remaining arguments to pass directly to the sampler
    """
    #print(fixed_embedding)
    assert num_eval_examples % batch_size == 0
    if data_name == 'gaussian':
        data_sampler_kwargs['fixed_embedding'] = fixed_embedding
        
    task_sampler_kwargs['seeds'] = torch.arange(batch_size).tolist()
    data_sampler = get_data_sampler(data_name, n_dims, **data_sampler_kwargs)
    task_sampler = get_task_sampler(
        task_name, n_dims, batch_size, **task_sampler_kwargs
    )

    all_metrics = []

    generating_func = globals()[f"gen_{prompting_strategy}"]
    prediction_collector = None
    if collect_prediction:
        prediction_collector = PredictionCollector()
    #print(num_eval_examples)
    for i in range(num_eval_examples // batch_size):
        #print(i)
        
        xs, xs_p = generating_func(data_sampler, n_points, batch_size)

        metrics = eval_batch(model, task_sampler, xs, xs_p, device, prediction_collector=prediction_collector, data_name=data_name)
        all_metrics.append(metrics)
    #print(model.name) # gpt2_embd=256_layer=12_head=8
    
    if collect_prediction and 'gpt2' in model.name:
        pretrain_name = pretrain_path.split('/')[-2] # what type of prertrain
        prediction_collector.save_prediction(pretrain_name=pretrain_name, task_name=task_name, other_info=str(num_eval_examples // batch_size))
    metrics = torch.cat(all_metrics, dim=0)

    return aggregate_metrics(metrics)


def build_evals(conf):
    n_dims = conf.model.n_dims
    n_points = conf.training.curriculum.points.end
    batch_size = conf.training.batch_size

    task_name = conf.training.task
    data_name = conf.training.data

    base_kwargs = {
        "task_name": task_name,
        "n_dims": n_dims,
        "n_points": n_points,
        "batch_size": batch_size,
        "data_name": data_name,
        "prompting_strategy": "standard",
    }

    evaluation_kwargs = {}

    evaluation_kwargs["standard"] = {"prompting_strategy": "standard"}
    if task_name != "linear_regression":
        if task_name in ["relu_2nn_regression"]:
            evaluation_kwargs["linear_regression"] = {"task_name": "linear_regression"}
        for name, kwargs in evaluation_kwargs.items():
            # allow kwargs to override base_kwargs values
            evaluation_kwargs[name] = base_kwargs.copy()
            evaluation_kwargs[name].update(kwargs)
        return evaluation_kwargs

    for strategy in [
        "random_quadrants",
        "orthogonal_train_test",
        "overlapping_train_test",
    ]:
        evaluation_kwargs[strategy] = {"prompting_strategy": strategy}

    for method in ["half_subspace", "skewed"]:
        if "subspace" in method:
            eigenvals = torch.zeros(n_dims)
            eigenvals[: n_dims // 2] = 1
        else:
            eigenvals = 1 / (torch.arange(n_dims) + 1)

        scale = sample_transformation(eigenvals, normalize=True)
        evaluation_kwargs[f"{method}"] = {
            "data_sampler_kwargs": {"scale": scale},
        }

    for dim in ["x", "y"]:
        for scale in [0.333, 0.5, 2, 3]:
            if dim == "x":
                eigenvals = scale * torch.ones(n_dims)
                t = sample_transformation(eigenvals)
                scaling_args = {"data_sampler_kwargs": {"scale": t}}
            else:
                eigenvals = scale * torch.ones(n_dims)
                scaling_args = {"task_sampler_kwargs": {"scale": scale}}

            evaluation_kwargs[f"scale-{dim}={scale}"] = scaling_args

    evaluation_kwargs[f"noisyLR"] = {
        "task_sampler_kwargs": {"renormalize_ys": True, "noise_std": 1},
        "task_name": "noisy_linear_regression",
    }

    for name, kwargs in evaluation_kwargs.items():
        # allow kwargs to override base_kwargs values
        evaluation_kwargs[name] = base_kwargs.copy()
        evaluation_kwargs[name].update(kwargs)

    return evaluation_kwargs


def build_evals_ood_task(conf, ood_task, n_points, rank, data_type): 
    # pretrain on conf.training.task, evaluate on ood_task
    n_dims = conf.model.n_dims
    #print(conf)
    #n_points = conf.training.curriculum.points.end
    batch_size = conf.training.batch_size
    if ood_task == 'random_int_mapping':
        data_name = 'random_int'
    else:
        data_name = data_type # we only test on gaussian data (autoregression data are only for pretraining), default: gaussian
    task_name = conf.training.task # pretrain task
    base_kwargs = {
        #"task_name": task_name,
        "n_dims": n_dims,
        #"n_points": n_points,
        "batch_size": batch_size,
        "data_name": data_name,
        #"prompting_strategy": "standard",
    }

    evaluation_kwargs = {}
    #evaluation_kwargs["standard"] = {"prompting_strategy": "standard"} # we don't need to evaluate the standard case, instead only the ood_task will be evaluated
    
    if ood_task == "linear_regression": # standard linear regression
        evaluation_kwargs["linear_regression"] = {"task_name": "linear_regression"}
        evaluation_kwargs["linear_regression"]["prompting_strategy"]="standard"
    # below are all linear regression OOD variants
    elif ood_task == "random_quadrants":
        evaluation_kwargs["random_quadrants"] = {"prompting_strategy": "random_quadrants"}
        evaluation_kwargs["random_quadrants"]["task_name"]="linear_regression"
    elif ood_task == "orthogonal_train_test":
        evaluation_kwargs["orthogonal_train_test"] = {"prompting_strategy": "orthogonal_train_test"}
        evaluation_kwargs["orthogonal_train_test"]["task_name"]="linear_regression"
    elif ood_task == "overlapping_train_test":
        evaluation_kwargs["overlapping_train_test"] = {"prompting_strategy": "overlapping_train_test"}
        evaluation_kwargs["overlapping_train_test"]["task_name"]="linear_regression"
    elif ood_task in ["half_subspace", "skewed"]:
        if "subspace" in ood_task:
            eigenvals = torch.zeros(n_dims)
            eigenvals[: n_dims // 2] = 1
        else:
            eigenvals = 1 / (torch.arange(n_dims) + 1)

        scale = sample_transformation(eigenvals, normalize=True)
        evaluation_kwargs[f"{ood_task}"] = {
            "data_sampler_kwargs": {"scale": scale},
        }
        evaluation_kwargs[ood_task]["task_name"]="linear_regression"
        evaluation_kwargs[ood_task]["prompting_strategy"]="standard"
    elif ood_task == "scaling":
        for dim in ["x", "y"]:
            for scale in [0.333, 0.5, 2, 3]:
                if dim == "x":
                    eigenvals = scale * torch.ones(n_dims)
                    t = sample_transformation(eigenvals)
                    scaling_args = {"data_sampler_kwargs": {"scale": t}}
                else:
                    eigenvals = scale * torch.ones(n_dims)
                    scaling_args = {"task_sampler_kwargs": {"scale": scale}}

                evaluation_kwargs[f"scale-{dim}={scale}"] = scaling_args
                evaluation_kwargs[f"scale-{dim}={scale}"]["task_name"]="linear_regression"
                evaluation_kwargs[f"scale-{dim}={scale}"]["prompting_strategy"]="standard"
    elif ood_task == "noisyLR":
        evaluation_kwargs[f"noisyLR"] = {
            "task_sampler_kwargs": {"renormalize_ys": True, "noise_std": 1},
            "task_name": "noisy_linear_regression",
            "prompting_strategy": "standard"
        }
    elif ood_task in ["random_int_mapping"]:
        evaluation_kwargs[ood_task] = {
            "task_name": ood_task,
            "prompting_strategy": "random_int",
            "task_sampler_kwargs": {"num_classes": 10000, "shift_low": 500, "shift_high":600}
            }
    elif ood_task in ["linear_regression_random_mapping"]:
        evaluation_kwargs[ood_task] = {
            "task_name": ood_task,
            "prompting_strategy": "standard",
            "task_sampler_kwargs": {"shift_low": 3000, "shift_high":3100}
            }
    elif ood_task in ["quadratic_regression_random_mapping"]:
        evaluation_kwargs[ood_task] = {
            "task_name": ood_task,
            "prompting_strategy": "standard",
            "task_sampler_kwargs": {"shift_low": 3000, "shift_high":3100}
            }
    elif ood_task in ["linear_regression_keep_dim", "quadratic_regression_keep_dim"]:
        evaluation_kwargs[ood_task] = {
            "task_name": ood_task,
            "prompting_strategy": "standard",
            "rank": rank,
            }
    elif ood_task in ["power_regression"]:
        evaluation_kwargs[ood_task] = {
            "task_name": ood_task,
            "prompting_strategy": "standard",
            "task_sampler_kwargs": {"powers": [4.0]}
            }
    else:
        evaluation_kwargs[ood_task] = {
            "task_name": ood_task,
            "prompting_strategy": "standard",
            }

    for name, kwargs in evaluation_kwargs.items():
        # allow kwargs to override base_kwargs values
        evaluation_kwargs[name] = base_kwargs.copy()
        evaluation_kwargs[name].update(kwargs)
        evaluation_kwargs[name]['n_points'] = n_points

    return evaluation_kwargs


def compute_evals(pretrain_path, all_models, evaluation_kwargs, save_path=None, recompute=False, device="cuda:0", collect_prediction=False, fixed_embedding=None, rank=None, ood_task=None):
    try:
        with open(save_path) as fp:
            #print("save_path:", save_path)
            all_metrics = json.load(fp)
            #print("all metrics:", all_metrics)
    except Exception:
        all_metrics = {}

    for eval_name, kwargs in tqdm(evaluation_kwargs.items()):
        metrics = {}
        # {'task_name': 'linear_rergession', 'n_dims': 20, 'n_points': 101, 'batch_size': 64, 'data_name': 'gaussian', 'prompting_strategy': 'standard'}
        kwargs['collect_prediction'] = collect_prediction
        kwargs['fixed_embedding'] = fixed_embedding
        #print(eval_name, kwargs)
        if eval_name in all_metrics and not recompute:
            metrics = all_metrics[eval_name]
        for model in all_models:
            #print(model.name)
            if model.name in metrics and not recompute:
                #print(666)
                continue
            #if 'gpt2' not in model.name:
            #    continue
            metrics[model.name] = eval_model(pretrain_path, model, device, ood_task=ood_task, **kwargs)
            #break ##################################################################################
        eval_task_name = eval_name
        if "task_sampler_kwargs" in kwargs.keys():
            for k,v in kwargs["task_sampler_kwargs"].items():
                if k == "seeds":
                    continue
                eval_task_name+= "-"+k+"-"+str(v)
        #print(eval_task_name)
        all_metrics[eval_task_name] = metrics

    if save_path is not None:
        with open(save_path, "w") as fp:
            json.dump(all_metrics, fp, indent=2)

    return all_metrics


def get_run_metrics(
    run_path, step=-1, cache=True, skip_model_load=False, skip_baselines=False, device=None
):
    
    if skip_model_load:
        _, conf = get_model_from_run(run_path, only_conf=True, device=device)
        device=conf.training.device
        all_models = []
    else:
        model, conf = get_model_from_run(run_path, step, device=device)
        device=conf.training.device
        model = model.to(conf.training.device)
        model = model.eval()
        all_models = [model]
        if not skip_baselines:
            all_models += models.get_relevant_baselines(conf.training.task, device)
    evaluation_kwargs = build_evals(conf)

   

    if not cache:
        save_path = None
    elif step == -1:
        save_path = os.path.join(run_path, "metrics.json")
    else:
        save_path = os.path.join(run_path, f"metrics_{step}.json")

    recompute = False
    if save_path is not None and os.path.exists(save_path):
        checkpoint_created = os.path.getmtime(run_path)
        cache_created = os.path.getmtime(save_path)
        if checkpoint_created > cache_created:
            recompute = True

    all_metrics = compute_evals(run_path, all_models, evaluation_kwargs, save_path, recompute, device)
    return all_metrics


def get_ood_task_run_metrics(
    run_path, ood_task="linear_regression", device="cuda:0", n_context_test=101, step=-1, cache=True, skip_model_load=False, skip_baselines=False
, random_init_model=False, collect_prediction=False, fixed_embedding=None, rank=None, data_type=None): # run a sigle OOD evaluation task. for example, we pretrain on a ReluNN task and eval on a linear regression task
    if skip_model_load:
        _, conf = get_model_from_run(run_path, only_conf=True, random_init_model=random_init_model, device=device)
        all_models = []
    else:
        model, conf = get_model_from_run(run_path, step, random_init_model=random_init_model, device=device)
        model = model.to(device)
        model = model.eval()
        all_models = [model]
        if not skip_baselines:
            all_models += models.get_relevant_baselines(conf.training.task, device)
    evaluation_kwargs = build_evals_ood_task(conf, ood_task, n_context_test, rank=rank, data_type=data_type)
    #evaluation_kwargs['n_points'] = n_context_test
    #print(evaluation_kwargs)
    # evaluation_kwargs = {'standard': {'task_name': 'relu_multilayernn_regression', 'n_dims': 20, 'n_points': 101, 'batch_size': 64, 'data_name': 'gaussian', 'prompting_strategy': 'standard'}}
    #evaluation_kwargs['standard']['task_name'] = ood_task
    if not cache:
        save_path = None
    elif step == -1:
        save_path = os.path.join(run_path, "metrics.json")
    else:
        save_path = os.path.join(run_path, f"metrics_{step}.json")

    #recompute = False
    recompute = True
    if save_path is not None and os.path.exists(save_path):
        checkpoint_created = os.path.getmtime(run_path)
        cache_created = os.path.getmtime(save_path)
        if checkpoint_created > cache_created:
            recompute = True
    #print(fixed_embedding)
    all_metrics = compute_evals(run_path, all_models, evaluation_kwargs, save_path, recompute, device, collect_prediction=collect_prediction, fixed_embedding=fixed_embedding, rank=rank, ood_task=ood_task)
    return all_metrics


def conf_to_model_name(conf):
    if conf.model.family == "gpt2":
        return {
            (3, 2): "Transformer-xs",
            (6, 4): "Transformer-small",
            (12, 8): "Transformer",
            (24, 12): "Transformer-large",
            (24, 16): "Transformer-huge",
        }[(conf.model.n_layer, conf.model.n_head)]
    else:
        return conf.wandb.name


def baseline_names(name):
    if "OLS" in name:
        return "Least Squares"
    if name == "averaging":
        return "Averaging"
    if "NN" in name:
        k = name.split("_")[1].split("=")[1]
        return f"{k}-Nearest Neighbors"
    if "lasso" in name:
        alpha = name.split("_")[1].split("=")[1]
        return f"Lasso (alpha={alpha})"
    if "NeuralNetwork" in name: # if "gd" in name:
        return "2-layer NN, GD"
    if ".LinearRegressionModel" in name: 
        return "1-layer Linear Regression, GD"
    if ".QuadraticRegressionModel" in name: 
        #print(666)
        return "1-layer Quadratic Regression, GD"
    if ".PowerRegressionModel" in name: 
        #print(666)
        return "1-layer Power Regression, GD"
    if ".LinearQuadraticRegressionModel" in name: 
        #print(666)
        return "1-layer Linear Quadratic Regression, GD"
    if "decision_tree" in name:
        return "Greedy Tree Learning"
    if "xgboost" in name:
        return "XGBoost"
    return name


def read_run_dir(run_dir, device):
    all_runs = {}
    for task in os.listdir(run_dir):
        task_dir = os.path.join(run_dir, task)
        for run_id in os.listdir(task_dir):
            run_path = os.path.join(task_dir, run_id)
            _, conf = get_model_from_run(run_path, only_conf=True, device=device)
            params = {}
            params["run_id"] = run_id
            params["task"] = task
            params["model"] = conf_to_model_name(conf)
            params["kwargs"] = "_".join(
                f"{k}={v}" for k, v in conf.training.task_kwargs.items()
            )
            num_tasks = (
                conf.training.num_tasks if "num_tasks" in conf.training else None
            )
            params["num_tasks"] = num_tasks if num_tasks is not None else -1
            num_examples = (
                conf.training.num_training_examples
                if "num_training_examples" in conf.training
                else None
            )
            params["num_examples"] = num_examples if num_examples is not None else -1
            params["n_dims"] = conf.model.n_dims
            params["n_layer"] = conf.model.n_layer
            params["n_head"] = conf.model.n_head
            params["run_name"] = conf.wandb.name

            for k, v in params.items():
                if k not in all_runs:
                    all_runs[k] = []
                all_runs[k].append(v)

    df = pd.DataFrame(all_runs).sort_values("run_name")
    #print(df)
    #print(df.run_name.unique())
    assert len(df) == len(df.run_name.unique())
    return df

class PredictionCollector():
    def __init__(self) -> None:
        self.xs = []
        self.ys = []
        self.preds = []
    def collect(self, x, y, pred):
        #print("x.shape", x.shape, "pred.shape", pred.shape)
        self.xs.append(x)
        self.ys.append(y)
        self.preds.append(pred)
    def save_prediction(self, pretrain_name, task_name, other_info):
        torch.save((self.xs, self.ys, self.preds), '../results/saved_predictions/pt-'+pretrain_name+'_eval-'+task_name+'_evalsamplenum'+other_info+'.pt')
        pass
'''if __name__ == "__main__":
    run_dir = sys.argv[1]
    for task in os.listdir(run_dir):
        task_dir = os.path.join(run_dir, task)
        print(f"Evaluating task {task}")
        for run_id in tqdm(os.listdir(task_dir)):
            run_path = os.path.join(run_dir, task, run_id)
            metrics = get_run_metrics(run_path)'''
