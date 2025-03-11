import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2Config
from tqdm import tqdm
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression, Lasso
import warnings
from sklearn import tree
import xgboost as xgb

from base_models import NeuralNetwork, ParallelNetworks, LinearRegressionModel, QuadraticRegressionModel, LinearQuadraticRegressionModel, PowerRegressionModel

from transformers import AutoModelForCausalLM

def build_model(conf):
    if conf.family == "gpt2":
        model = TransformerModel(
            n_dims=conf.n_dims,
            n_positions=conf.n_positions,
            n_embd=conf.n_embd,
            n_layer=conf.n_layer,
            n_head=conf.n_head,
#            num_classes = conf.num_classes
        )
    elif conf.family == "llama":
        model = AutoModelForCausalLM.from_pretrained(conf.model_path)
    else:
        raise NotImplementedError

    return model


def get_relevant_baselines(task_name, device):
    task_to_baselines = {
        "linear_regression": [
            #(LeastSquaresModel, {}),
            #(NNModel, {"n_neighbors": 3}),
            #(AveragingModel, {}),
            (GDModel,{"model_class": LinearRegressionModel,"model_class_args": {"in_size": 20,"out_size": 1,},"opt_alg": "adam","batch_size": 100,"lr": 5e-3,"num_steps": 200,},),
            
        ],
        "quadratic_regression": [
            #(LeastSquaresModel, {}),
            #(NNModel, {"n_neighbors": 3}),
            #(GDModel,{"model_class": NeuralNetwork,"model_class_args": {"in_size": 20,"hidden_size": 100,"out_size": 1,},"opt_alg": "adam","batch_size": 100,"lr": 5e-3,"num_steps": 100,},),
            (GDModel,{"model_class": QuadraticRegressionModel,"model_class_args": {"in_size": 20,"out_size": 1,},"opt_alg": "adam","batch_size": 100,"lr": 5e-3,"num_steps": 200,},),
        ],
        "linear_classification": [
            (NNModel, {"n_neighbors": 3}),
            (AveragingModel, {}),
        ],
        "sparse_linear_regression": [
            (LeastSquaresModel, {}),
            (NNModel, {"n_neighbors": 3}),
            (AveragingModel, {}),
        ]
        + [(LassoModel, {"alpha": alpha}) for alpha in [1, 0.1, 0.01, 0.001, 0.0001]],
        "relu_2nn_regression": [
            #(LeastSquaresModel, {}),
            #(NNModel, {"n_neighbors": 3}),
            #(AveragingModel, {}),
            (GDModel,{"model_class": NeuralNetwork,"model_class_args": {"in_size": 20,"hidden_size": 100,"out_size": 1,},"opt_alg": "adam","batch_size": 100,"lr": 5e-3,"num_steps": 100,},),
            #(GDModel,{"model_class": LinearRegressionModel,"model_class_args": {"in_size": 20,"out_size": 1,},"opt_alg": "adam","batch_size": 100,"lr": 5e-3,"num_steps": 100,},),
            #(GDModel,{"model_class": QuadraticRegressionModel,"model_class_args": {"in_size": 20,"out_size": 1,},"opt_alg": "adam","batch_size": 100,"lr": 5e-3,"num_steps": 100,},),
        ],
        "relu_multilayernn_regression": [
            (LeastSquaresModel, {}),
            (NNModel, {"n_neighbors": 3}),
            (AveragingModel, {}),
            (
                GDModel,
                {
                    "model_class": NeuralNetwork,
                    "model_class_args": {
                        "in_size": 20,
                        "hidden_size": 100,
                        "out_size": 1,
                    },
                    "opt_alg": "adam",
                    "batch_size": 100,
                    "lr": 5e-3,
                    "num_steps": 100,
                },
            ),
        ],
        "relu_2nn_regression_non-ICL": [
            '''(LeastSquaresModel, {}),
            (NNModel, {"n_neighbors": 3}),
            (AveragingModel, {}),
            (
                GDModel,
                {
                    "model_class": NeuralNetwork,
                    "model_class_args": {
                        "in_size": 20,
                        "hidden_size": 100,
                        "out_size": 1,
                    },
                    "opt_alg": "adam",
                    "batch_size": 100,
                    "lr": 5e-3,
                    "num_steps": 100,
                },
            ),'''
        ],
        "decision_tree": [
            (LeastSquaresModel, {}),
            (NNModel, {"n_neighbors": 3}),
            (DecisionTreeModel, {"max_depth": 4}),
            (DecisionTreeModel, {"max_depth": None}),
            (XGBoostModel, {}),
            (AveragingModel, {}),
        ],
        "random_init": [
            #(LeastSquaresModel, {}),
            (NNModel, {"n_neighbors": 3}),
            #(AveragingModel, {})
        ],
        "autoregression": [
            #(LeastSquaresModel, {}),
            (NNModel, {"n_neighbors": 3}),
        ],
        "linear_quadratic_regression":[
            #(LeastSquaresModel, {}),
            #(NNModel, {"n_neighbors": 3}),
            #(AveragingModel, {}),
            (GDModel,{"model_class": LinearQuadraticRegressionModel,"model_class_args": {"in_size": 20,"out_size": 1,},"opt_alg": "adam","batch_size": 100,"lr": 5e-3,"num_steps": 200,},),
        ],
        "sin_regression":[
            #(LeastSquaresModel, {}),
            #(NNModel, {"n_neighbors": 3}),
        ],
        "quadratic_sum": [
            #(LeastSquaresModel, {}),
            (NNModel, {"n_neighbors": 3}),
            #(AveragingModel, {}),
        ],
        "linear_regression_keep_dim": [
            #(LeastSquaresModel, {}),
            #(NNModel, {"n_neighbors": 3}),
            #(AveragingModel, {}),
        ],
        "quadratic_regression_keep_dim": [
            #(LeastSquaresModel, {}),
            (NNModel, {"n_neighbors": 3}),
            #(AveragingModel, {}),
        ],
        "multi_task":[
            #(NNModel, {"n_neighbors": 3}),
        ],
        "cubic_regression":[
            (NNModel, {"n_neighbors": 3}),
        ],
        "sqrt_regression":[
            (NNModel, {"n_neighbors": 3}),
        ],
        "sigmoid_2nn_regression":[
            (NNModel, {"n_neighbors": 3}),
        ],
        "linear_regression_random_mapping":[
            #(NNModel, {"n_neighbors": 3}),
        ],
        "quadratic_regression_random_mapping":[
            #(NNModel, {"n_neighbors": 3}),
        ],
        "random_int_mapping":[
            #(NNModel, {"n_neighbors": 3}),
        ],
        "quadratic":[
            #(NNModel, {"n_neighbors": 3}),
        ],
        "linear_regression_diversity":[
        ],
        "power_regression":[
            #(GDModel,{"model_class": PowerRegressionModel,"model_class_args": {"in_size": 20,"out_size": 1, "power":2.4},"opt_alg": "adam","batch_size": 100,"lr": 5e-3,"num_steps": 1000,},),
        ]
    }
    print(task_to_baselines[task_name])
    models = [model_cls(**kwargs) for model_cls, kwargs in task_to_baselines[task_name]]
    return models


class TransformerModel(nn.Module):
    def __init__(self, n_dims, n_positions, n_embd=128, n_layer=12, n_head=4, num_classes = 70):
        super(TransformerModel, self).__init__()
        configuration = GPT2Config(
            n_positions=2 * n_positions,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
            use_cache=False,
        )
        self.name = f"gpt2_embd={n_embd}_layer={n_layer}_head={n_head}"

        self.n_positions = n_positions
        self.n_dims = n_dims
        self.num_classes = num_classes
        self._read_in = nn.Linear(n_dims, n_embd)
        self._backbone = GPT2Model(configuration)
        self._read_out = nn.Linear(n_embd, 1)
        
        self._read_out_keep_dim = nn.Linear(n_embd, n_dims)
        #self._read_out_after_keep_dim  = nn.Linear(n_dims, 1) # for linear_regression_diversity pretrain
        
        #self._read_out_multi_classification = nn.Linear(n_embd, self.num_classes)

    @staticmethod
    def _combine(xs_b, ys_b):
        """Interleaves the x's and the y's into a single sequence."""
        bsize, points, dim = xs_b.shape
        if xs_b.shape != ys_b.shape:
            #print(444)
            ys_b = torch.cat(
                (
                    ys_b.view(bsize, points, 1),
                    torch.zeros(bsize, points, dim - 1, device=ys_b.device),
                ),
                axis=2,
            )
        zs = torch.stack((xs_b, ys_b), dim=2)
        zs = zs.view(bsize, 2 * points, dim)
        return zs

    def forward(self, xs, ys, inds=None, output_attentions = False):
        #print(55)
        if ys.shape[1]==xs.shape[1]: # ICL setting
            if inds is None:
                inds = torch.arange(ys.shape[1])
            else:
                inds = torch.tensor(inds)
                if max(inds) >= ys.shape[1] or min(inds) < 0:
                    raise ValueError("inds contain indices where xs and ys are not defined")
        
            #print('xs', xs.shape) # torch.Size([64, 101, 20])
            #print('ys', ys.shape) # regression/nn: torch.Size([64, 101]), scalar: torch.Size([64, 101, 20])
            
            zs = self._combine(xs, ys)
            #print('zs', zs.shape) # torch.Size([64, 202, 20])
            
            embeds = self._read_in(zs)
            #print('embeds', embeds.shape) # torch.Size([256, 64, 1024])
            #print(len(self._backbone(inputs_embeds=embeds, output_attentions = True).attentions)) # 12
            #print(self._backbone(inputs_embeds=embeds, output_attentions = True).attentions[0].shape) #torch.Size([64, 8, 20, 20])
            GPT2_output = self._backbone(inputs_embeds=embeds, output_attentions=output_attentions)
            output = GPT2_output.last_hidden_state
            
            #print('output', output.shape) # torch.Size([256, 64, 1024])
            if ys.dim()==2: # map the last dim from hidden_dim to 1
                #print(66)
                if ys.dtype == torch.int64: #multi_classification, for LR random mapping & random_int_mapping
                    prediction = self._read_out_multi_classification(output)
                    if output_attentions:
                        return prediction[:, ::2, :][:, inds], GPT2_output.attentions
                    else:
                        return prediction[:, ::2, :][:, inds]
                else:
                    prediction = self._read_out(output) # prediction.shape torch.Size([256, 64, 1]) # LR, QR, ...
                    #prediction = self._read_out_after_keep_dim( self._read_out_keep_dim(output) )# training for evaluate on the compositional task: LRkeepdim-LR, and linear_regression_diversity
                    if output_attentions:
                        return prediction[:, ::2, 0][:, inds], GPT2_output.attentions
                    else:
                        return prediction[:, ::2, 0][:, inds] 
            elif ys.dim()==3:
                #print(77)
                prediction = self._read_out_keep_dim(output)
                #print(prediction[:, ::2].shape, inds)
                if output_attentions:
                    return prediction[:, ::2, :][:, inds], GPT2_output.attentions
                else:
                    return prediction[:, ::2, :][:, inds]
        elif ys.shape[1]==xs.shape[1]-1: # no y, autoregression setting
            embeds = self._read_in(xs)
            output = self._backbone(inputs_embeds=embeds).last_hidden_state
            if torch.sum(xs[1,1:])==0: # the Ys in the autoregression task is a scalar, only the first dim of Y is non-zero
                prediction = self._read_out(output)
                if output_attentions:
                    return prediction[:,:-1,0], GPT2_output.attentions
                else:
                    return prediction[:,:-1,0]
            else:
                prediction = self._read_out_keep_dim(output) 
                if output_attentions:
                    return prediction[:,:-1,:], GPT2_output.attentions
                else:
                    return prediction[:,:-1,:]
        else:
            raise NotImplementedError
        
class LlamaTransformerModel(nn.Module):
    def __init__(self, n_dims, n_positions, n_embd=128, n_layer=12, n_head=4):
        super(LlamaTransformerModel, self).__init__()
        configuration = GPT2Config(
            n_positions=2 * n_positions,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
            use_cache=False,
        )
        self.name = f"gpt2_embd={n_embd}_layer={n_layer}_head={n_head}"

        self.n_positions = n_positions
        self.n_dims = n_dims
        self._read_in = nn.Linear(n_dims, n_embd)
        self._backbone = GPT2Model(configuration)
        self._read_out = nn.Linear(n_embd, 1)
        
        self._read_out_keep_dim = nn.Linear(n_embd, n_dims)

    @staticmethod
    def _combine(xs_b, ys_b):
        """Interleaves the x's and the y's into a single sequence."""
        bsize, points, dim = xs_b.shape
        if xs_b.shape != ys_b.shape:
            ys_b = torch.cat(
                (
                    ys_b.view(bsize, points, 1),
                    torch.zeros(bsize, points, dim - 1, device=ys_b.device),
                ),
                axis=2,
            )
        zs = torch.stack((xs_b, ys_b), dim=2)
        zs = zs.view(bsize, 2 * points, dim)
        return zs

    def forward(self, xs, ys, inds=None):
        if ys.shape[1]==xs.shape[1]: # ICL setting
            if inds is None:
                inds = torch.arange(ys.shape[1])
            else:
                inds = torch.tensor(inds)
                if max(inds) >= ys.shape[1] or min(inds) < 0:
                    raise ValueError("inds contain indices where xs and ys are not defined")
        
            #print('xs', xs.shape) # torch.Size([64, 101, 20])
            #print('ys', ys.shape) # regression/nn: torch.Size([64, 101]), scalar: torch.Size([64, 101, 20])
            
            zs = self._combine(xs, ys)
            #print('zs', zs.shape) # torch.Size([64, 202, 20])
            
            embeds = self._read_in(zs)
            #print('embeds', embeds.shape) # torch.Size([256, 64, 1024])
            output = self._backbone(inputs_embeds=embeds).last_hidden_state
            #print('output', output.shape) # torch.Size([256, 64, 1024])
            if ys.dim()==2: # map the last dim from hidden_dim to 1
                prediction = self._read_out(output) # prediction.shape torch.Size([256, 64, 1])
                return prediction[:, ::2, 0][:, inds]
            elif ys.dim()==3:
                prediction = self._read_out_keep_dim(output)
                #print(prediction[:, ::2].shape, inds)
                return prediction[:, ::2, :][:, inds]
        elif ys.shape[1]==xs.shape[1]-1: # no y, autoregression setting
            embeds = self._read_in(xs)
            output = self._backbone(inputs_embeds=embeds).last_hidden_state
            if torch.sum(xs[1,1:])==0: # the Ys in the autoregression task is a scalar, only the first dim of Y is non-zero
                prediction = self._read_out(output)
                return prediction[:,:-1,0]
            else:
                prediction = self._read_out_keep_dim(output) 
                return prediction[:,:-1,:]
        else:
            raise NotImplementedError        


class NNModel:
    def __init__(self, n_neighbors, weights="uniform"):
        # should we be picking k optimally
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.name = f"NN_n={n_neighbors}_{weights}"

    def __call__(self, xs, ys, inds=None):
        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []
        #print(xs.shape, ys.shape) # LR: torch.Size([64, 101, 20]) torch.Size([64, 101])
        for i in inds:
            if i == 0:
                preds.append(torch.zeros_like(ys[:, 0]))  # predict zero for first point
                continue
            train_xs, train_ys = xs[:, :i], ys[:, :i]
            test_x = xs[:, i : i + 1]
            #print(train_xs.shape)
            dist = (train_xs - test_x).square().sum(dim=2).sqrt()
            #print(66, dist.shape)
            if self.weights == "uniform":
                weights = torch.ones_like(dist)
            else:
                weights = 1.0 / dist
                inf_mask = torch.isinf(weights).float()  # deal with exact match
                inf_row = torch.any(inf_mask, axis=1)
                weights[inf_row] = inf_mask[inf_row]

            pred = []
            k = min(i, self.n_neighbors)
            ranks = dist.argsort()[:, :k]
            for y, w, n in zip(train_ys, weights, ranks):
                y, w = y[n], w[n]
                if ys.dim()==3:
                    w.unsqueeze_(-1)
                
                pred.append((w * y).sum(0) / w.sum())
            preds.append(torch.stack(pred))

        return torch.stack(preds, dim=1)


# xs and ys should be on cpu for this method. Otherwise the output maybe off in case when train_xs is not full rank due to the implementation of torch.linalg.lstsq.
class LeastSquaresModel:
    def __init__(self, driver=None):
        self.driver = driver
        self.name = f"OLS_driver={driver}"

    def __call__(self, xs, ys, inds=None):
        xs, ys = xs.cpu(), ys.cpu()
        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []

        for i in inds:
            if i == 0:
                preds.append(torch.zeros_like(ys[:, 0]))  # predict zero for first point
                continue
            train_xs, train_ys = xs[:, :i], ys[:, :i]
            test_x = xs[:, i : i + 1]
            #print(train_xs.shape, train_ys.unsqueeze(2).shape)
            ws, _, _, _ = torch.linalg.lstsq(
                train_xs, train_ys.unsqueeze(2), driver=self.driver
            )

            pred = test_x @ ws
            preds.append(pred[:, 0, 0])

        return torch.stack(preds, dim=1)


class AveragingModel:
    def __init__(self):
        self.name = "averaging"

    def __call__(self, xs, ys, inds=None):
        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []

        for i in inds:
            if i == 0:
                preds.append(torch.zeros_like(ys[:, 0]))  # predict zero for first point
                continue
            train_xs, train_ys = xs[:, :i], ys[:, :i]
            test_x = xs[:, i : i + 1]

            train_zs = train_xs * train_ys.unsqueeze(dim=-1)
            w_p = train_zs.mean(dim=1).unsqueeze(dim=-1)
            pred = test_x @ w_p
            preds.append(pred[:, 0, 0])

        return torch.stack(preds, dim=1)


# Lasso regression (for sparse linear regression).
# Seems to take more time as we decrease alpha.
class LassoModel:
    def __init__(self, alpha, max_iter=100000):
        # the l1 regularizer gets multiplied by alpha.
        self.alpha = alpha
        self.max_iter = max_iter
        self.name = f"lasso_alpha={alpha}_max_iter={max_iter}"

    # inds is a list containing indices where we want the prediction.
    # prediction made at all indices by default.
    def __call__(self, xs, ys, inds=None):
        xs, ys = xs.cpu(), ys.cpu()

        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []  # predict one for first point

        # i: loop over num_points
        # j: loop over bsize
        for i in inds:
            pred = torch.zeros_like(ys[:, 0])

            if i > 0:
                pred = torch.zeros_like(ys[:, 0])
                for j in range(ys.shape[0]):
                    train_xs, train_ys = xs[j, :i], ys[j, :i]

                    # If all points till now have the same label, predict that label.

                    clf = Lasso(
                        alpha=self.alpha, fit_intercept=False, max_iter=self.max_iter
                    )

                    # Check for convergence.
                    with warnings.catch_warnings():
                        warnings.filterwarnings("error")
                        try:
                            clf.fit(train_xs, train_ys)
                        except Warning:
                            print(f"lasso convergence warning at i={i}, j={j}.")
                            raise

                    w_pred = torch.from_numpy(clf.coef_).unsqueeze(1)

                    test_x = xs[j, i : i + 1]
                    y_pred = (test_x @ w_pred.float()).squeeze(1)
                    pred[j] = y_pred[0]

            preds.append(pred)

        return torch.stack(preds, dim=1)


# Gradient Descent and variants.
# Example usage: gd_model = GDModel(NeuralNetwork, {'in_size': 50, 'hidden_size':400, 'out_size' :1}, opt_alg = 'adam', batch_size = 100, lr = 5e-3, num_steps = 200)
class GDModel:
    def __init__(
        self,
        model_class,
        model_class_args,
        opt_alg="sgd",
        batch_size=1,
        num_steps=1000,
        lr=1e-3,
        loss_name="squared",
        device="cuda:0"
    ):
        # model_class: torch.nn model class
        # model_class_args: a dict containing arguments for model_class
        # opt_alg can be 'sgd' or 'adam'
        # verbose: whether to print the progress or not
        # batch_size: batch size for sgd
        self.model_class = model_class
        self.model_class_args = model_class_args
        self.opt_alg = opt_alg
        self.lr = lr
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.loss_name = loss_name
        self.device = device
        self.name = f"gd_model_class={model_class}_model_class_args={model_class_args}_opt_alg={opt_alg}_lr={lr}_batch_size={batch_size}_num_steps={num_steps}_loss_name={loss_name}"

    def __call__(self, xs, ys, inds=None, verbose=False, print_step=100):
        # inds is a list containing indices where we want the prediction.
        # prediction made at all indices by default.
        # xs: bsize X npoints X ndim.
        # ys: bsize X npoints.
        xs, ys = xs.to(self.device), ys.to(self.device)

        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []  # predict one for first point

        # i: loop over num_points
        for i in tqdm(inds):
            pred = torch.zeros_like(ys[:, 0])
            model = ParallelNetworks(
                ys.shape[0], self.model_class, **self.model_class_args
            )
            model.to(self.device)
            if i > 0:
                pred = torch.zeros_like(ys[:, 0])

                train_xs, train_ys = xs[:, :i], ys[:, :i]
                test_xs, test_ys = xs[:, i : i + 1], ys[:, i : i + 1]

                if self.opt_alg == "sgd":
                    optimizer = torch.optim.SGD(model.parameters(), lr=self.lr)
                elif self.opt_alg == "adam":
                    optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
                else:
                    raise NotImplementedError(f"{self.opt_alg} not implemented.")

                if self.loss_name == "squared":
                    loss_criterion = nn.MSELoss()
                else:
                    raise NotImplementedError(f"{self.loss_name} not implemented.")

                # Training loop
                for j in range(self.num_steps):

                    # Prepare batch
                    mask = torch.zeros(i).bool()
                    perm = torch.randperm(i)
                    mask[perm[: self.batch_size]] = True
                    train_xs_cur, train_ys_cur = train_xs[:, mask, :], train_ys[:, mask]

                    if verbose and j % print_step == 0:
                        model.eval()
                        with torch.no_grad():
                            outputs = model(train_xs_cur)
                            loss = loss_criterion(
                                outputs[:, :, 0], train_ys_cur
                            ).detach()
                            outputs_test = model(test_xs)
                            test_loss = loss_criterion(
                                outputs_test[:, :, 0], test_ys
                            ).detach()
                            print(
                                f"ind:{i},step:{j}, train_loss:{loss.item()}, test_loss:{test_loss.item()}"
                            )

                    optimizer.zero_grad()

                    model.train()
                    outputs = model(train_xs_cur)
                    loss = loss_criterion(outputs[:, :, 0], train_ys_cur)
                    loss.backward()
                    optimizer.step()

                model.eval()
                pred = model(test_xs).detach()

                assert pred.shape[1] == 1 and pred.shape[2] == 1
                pred = pred[:, 0, 0]

            preds.append(pred)

        return torch.stack(preds, dim=1)


class DecisionTreeModel:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.name = f"decision_tree_max_depth={max_depth}"

    # inds is a list containing indices where we want the prediction.
    # prediction made at all indices by default.
    def __call__(self, xs, ys, inds=None):
        xs, ys = xs.cpu(), ys.cpu()

        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []

        # i: loop over num_points
        # j: loop over bsize
        for i in inds:
            pred = torch.zeros_like(ys[:, 0])

            if i > 0:
                pred = torch.zeros_like(ys[:, 0])
                for j in range(ys.shape[0]):
                    train_xs, train_ys = xs[j, :i], ys[j, :i]

                    clf = tree.DecisionTreeRegressor(max_depth=self.max_depth)
                    clf = clf.fit(train_xs, train_ys)
                    test_x = xs[j, i : i + 1]
                    y_pred = clf.predict(test_x)
                    pred[j] = y_pred[0]

            preds.append(pred)

        return torch.stack(preds, dim=1)


class XGBoostModel:
    def __init__(self):
        self.name = "xgboost"

    # inds is a list containing indices where we want the prediction.
    # prediction made at all indices by default.
    def __call__(self, xs, ys, inds=None):
        xs, ys = xs.cpu(), ys.cpu()

        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []

        # i: loop over num_points
        # j: loop over bsize
        for i in tqdm(inds):
            pred = torch.zeros_like(ys[:, 0])
            if i > 0:
                pred = torch.zeros_like(ys[:, 0])
                for j in range(ys.shape[0]):
                    train_xs, train_ys = xs[j, :i], ys[j, :i]

                    clf = xgb.XGBRegressor()

                    clf = clf.fit(train_xs, train_ys)
                    test_x = xs[j, i : i + 1]
                    y_pred = clf.predict(test_x)
                    pred[j] = y_pred[0].item()

            preds.append(pred)

        return torch.stack(preds, dim=1)
