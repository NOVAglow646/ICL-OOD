import os

import matplotlib.pyplot as plt
import seaborn as sns

from eval import get_run_metrics, baseline_names, get_model_from_run
from models import build_model

sns.set_theme("notebook", "darkgrid")
palette = sns.color_palette("colorblind")


relevant_model_names = {
    "linear_regression": [
        "Transformer",
        "1-layer Linear Regression, GD",
        #"Least Squares",
        #"3-Nearest Neighbors",
        #"Averaging",
    ],
    "quadratic_regression": [
        "Transformer",
        #"Least Squares",
        #"3-Nearest Neighbors",
        #"2-layer NN, GD",
        "1-layer Quadratic Regression, GD",
    ],
    "sparse_linear_regression": [
        "Transformer",
        "Least Squares",
        "3-Nearest Neighbors",
        "Averaging",
        "Lasso (alpha=0.01)",
    ],
    "decision_tree": [
        "Transformer",
        "3-Nearest Neighbors",
        "2-layer NN, GD",
        "Greedy Tree Learning",
        "XGBoost",
    ],
    "relu_2nn_regression": [
        "Transformer",
        #"Least Squares",
        #"3-Nearest Neighbors",
        "2-layer NN, GD",
    ],
    "relu_2nn_regression_non-ICL": [
        "Transformer",
        "Least Squares",
        "3-Nearest Neighbors",
        "2-layer NN, GD",
    ],
    "random_init":[
        "Transformer",
        "3-Nearest Neighbors"
    ],
    "linear_classification":[
        "Transformer",
        "3-Nearest Neighbors"
    ],
    "autoregression":[
        "Transformer",
        #"3-Nearest Neighbors"
    ],
    "linear_quadratic_regression": [
        "Transformer",
        "1-layer Linear Quadratic Regression, GD"
        #"Least Squares",
        #"3-Nearest Neighbors",
        #"2-layer NN, GD",
    ],
    "sin_regression":[
        "Transformer",
        #"Least Squares",
        "3-Nearest Neighbors",
    ],
    "multi_task":[
        "Transformer",
        #"3-Nearest Neighbors",
    ],
    "linear_regression_keep_dim":[
        "Transformer",
        "3-Nearest Neighbors"
    ],
    "random_int_mapping":[
        "Transformer",
        #"3-Nearest Neighbors"
    ],
    "linear_regression_random_mapping":[
        "Transformer",
        #"3-Nearest Neighbors"
    ],
    "quadratic_regression_random_mapping":[
        "Transformer",
        #"3-Nearest Neighbors"
    ],
    "linear_regression_diversity":[
        "Transformer",
    ],
    "power_regression":[
        "Transformer",
        #"1-layer Power Regression, GD",
        ]
}


def basic_plot(name, metrics, models=None, trivial=1.0):
    fig, ax = plt.subplots(1, 1)
    #print(name)
    if models is not None:
        metrics = {k: metrics[k] for k in models}

    color = 0
    if name in ['scalar_regression']:
        trivial = ((1.5-0.5)**2/12 + (0.5+1.5)**2/4)
    elif name in ['linear_classification']:
        trivial = 0.5
    elif name in ['linear_regression_random_mapping']:
        trivial = 1/70
    
    #print(metrics)
    #print(trivial)
    if name not in ['power_regression', 'relu_2nn_regression']:
        ax.axhline(trivial, ls="--", color="gray", label="Expec zero-pred")
        
    for name_, vs in metrics.items():
        #print(name_)
        #print(vs["mean"])
        label=name_ if name_!='Transformer' else 'ICL'
        ax.plot(vs["mean"], "-", label=label, color=palette[color % 10], lw=2)
        low = vs["bootstrap_low"]
        high = vs["bootstrap_high"]
        ax.fill_between(range(len(low)), low, high, alpha=0.3)
        color += 1
    ax.set_xlabel("in-context examples")
    if name in ['linear_classification', 'linear_regression_random_mapping']:
        ax.set_ylabel("accuracy")
    else:
        ax.set_ylabel("squared error")
    
    ax.set_xlim(-1, len(low) + 0.1)
    ax.set_ylim(-0.1, 1.25)

    legend = ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    fig.set_size_inches(4, 3)
    for line in legend.get_lines():
        line.set_linewidth(3)

    return fig, ax


def basic_plot_a_task(name, metrics, models=None, max_y=3.0, fig=None, ax=None, pretrain_task_name=None, color=None, keep_legend=True):
    #print(name)
    if models is not None:
        metrics = {k: metrics[k] for k in models}
    #color = 0
    if name in ['scalar_regression']:
        trivial = ((1.5-0.5)**2/12 + (0.5+1.5)**2/4)
    elif name in ['linear_classification']:
        trivial = 0.5
    elif name in ['linear_regression_random_mapping']:
        trivial = 1/70

    #print(trivial)
    #ax.axhline(trivial, ls="--", color="gray", label="Expec zero-pred")
    #print(color)
    for name_, vs in metrics.items():
        #print(name_)
        #print(vs["mean"])
        label=name_ if name_!='Transformer' else pretrain_task_name
        #print(label)
        if 'GD' in label:
            if label == "1-layer Linear Regression, GD":
                label = "1-layer linear regression, GD"
            if label == "1-layer Quadratic Regression, GD":
                label = "1-layer quadratic regression, GD"
            ax.plot(vs["mean"], "--", label=label, color=palette[color % 10], lw=2)
        else:
            ax.plot(vs["mean"], "-", label=label, color=palette[color % 10], lw=2)
        #if pretrain_task_name == 'ICL relu NN regression':
        #    ax.plot([100 for _ in vs["mean"]], "--", label="2-layer NN, GD", color=palette[color % 10], lw=2)
        low = vs["bootstrap_low"]
        high = vs["bootstrap_high"]
        ax.fill_between(range(len(low)), low, high, alpha=0.3)
        #color += 1
    ax.set_xlabel("in-context examples")
    if name in ['linear_classification']:
        ax.set_ylabel("accuracy")
    else:
        ax.set_ylabel("squared error")
    
    ax.set_xlim(-1, len(low) + 0.1)
    ax.set_ylim(-0.1, max_y)

    legend = ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    fig.set_size_inches(4, 3)
    for line in legend.get_lines():
        line.set_linewidth(3)
        
    if not keep_legend:
        legend.remove()
    return fig, ax




def collect_results(run_dir, df, valid_row=None, rename_eval=None, rename_model=None, task_para=None, device=None, step=-1):
    all_metrics = {}
    for _, r in df.iterrows():
        #print(r)
        if valid_row is not None and not valid_row(r):
            continue
        #print(r)
        if 'AR' in r.task or 'linear_quadratic' in r.task:
            run_path = os.path.join(run_dir, task_para, r.run_id)
        else:
            run_path = os.path.join(run_dir, r.task, r.run_id)
        _, conf = get_model_from_run(run_path, only_conf=True)
        #print(666, run_path)
        #print(r.run_name, r.run_id)
        metrics = get_run_metrics(run_path, skip_model_load=True, device=device, step=step)
        #print(666, metrics)
        for eval_name, results in sorted(metrics.items()):
            
            processed_results = {}
            for model_name, m in results.items():
                if "gpt2" in model_name in model_name:
                    model_name = r.model
                    if rename_model is not None:
                        model_name = rename_model(model_name, r)
                else:
                    #print("model_name", model_name)
                    model_name = baseline_names(model_name)
                m_processed = {}
                n_dims = conf.model.n_dims

                xlim = 2 * n_dims + 1
                if r.task in ["relu_2nn_regression", "relu_2nn_regression_non-ICL", "decision_tree"]:
                    xlim = 200
                elif 'AR' in r.task:
                    xlim = 30
                elif 'linear_quadratic' in r.task:
                    xlim =100

                normalization = n_dims
                if r.task == "sparse_linear_regression":
                    normalization = int(r.kwargs.split("=")[-1])
                if r.task in ["decision_tree"]:
                    normalization = 1
                if eval_name in ['linear_classification', 'linear_regression_random_mapping']:
                    normalization = 1
                
                for k, v in m.items():
                    #print(eval_name, k, len(v))
                    #v = v[:xlim]
                    v = [vv / normalization for vv in v]
                    m_processed[k] = v
                
                processed_results[model_name] = m_processed
            if rename_eval is not None:
                eval_name = rename_eval(eval_name, r)
            if eval_name not in all_metrics:
                all_metrics[eval_name] = {}
                
            
            all_metrics[eval_name].update(processed_results)
    return all_metrics
