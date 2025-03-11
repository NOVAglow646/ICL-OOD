import math
import numpy as np
import torch
import torch.nn.functional
import random

def squared_error(ys_pred, ys):
    #print((ys_pred).shape)
    return (ys - ys_pred).square()

def squared_error_sum(ys_pred, ys):
    #print((ys_pred).shape)
    return (ys - ys_pred).square().sum()

def mean_squared_error(ys_pred, ys):
    return (ys - ys_pred).square().mean()

def binary_accuracy(ys_pred, ys):
    return (ys == ys_pred.sign()).float()

def multi_accuracy(pred, y):
    #print(pred.shape, y.shape)
    return (torch.argmax(pred, dim=-1) == y).float()


sigmoid = torch.nn.Sigmoid()
bce_loss = torch.nn.BCELoss()
mce_loss = torch.nn.CrossEntropyLoss()

def binary_cross_entropy(ys_pred, ys):
    output = sigmoid(ys_pred)
    target = (ys + 1) / 2
    return bce_loss(output, target)

def multi_cross_entropy(pred, y):
    #print(pred.shape)
    pred = pred.permute(0, 2, 1) # [bsz, len, num_classes] -> [bsz, num_classes, len]
    return mce_loss(pred, y)

class Task:
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None):
        self.n_dims = n_dims
        self.b_size = batch_size
        self.pool_dict = pool_dict
        self.seeds = seeds
        assert pool_dict is None or seeds is None

    def evaluate(self, xs):
        raise NotImplementedError

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks):
        raise NotImplementedError

    @staticmethod
    def get_metric():
        raise NotImplementedError

    @staticmethod
    def get_training_metric():
        raise NotImplementedError


def get_task_sampler(
    task_name, n_dims, batch_size, pool_dict=None, num_tasks=None, **kwargs
):
    task_names_to_classes = {
        "linear_regression": LinearRegression,
        "sparse_linear_regression": SparseLinearRegression,
        "linear_classification": LinearClassification,
        "noisy_linear_regression": NoisyLinearRegression,
        "quadratic_regression": QuadraticRegression,
        "relu_2nn_regression": Relu2nnRegression,
        "relu_2nn_regression_non-ICL": Relu2nnRegression,
        "relu_multilayernn_regression": ReluMultiLayernnRegression,
        "decision_tree": DecisionTree,
        "scalar_regression": ScalarRegression,
        "autoregression": AutoRegression,
        "linear_regression_batch_share": LinearRegressionBatchShare,
        "quadratic_regression_batch_share": QuadraticRegressionBatchShare,
        "linear_quadratic_regression": LinearQuadraticRegression,
        "sin_regression": SinRegression,
        "quadratic_sum": QuadraticSum,
        "linear_regression_keep_dim": LinearRegressionKeepDim,
        "quadratic_regression_keep_dim": QuadraticRegressionKeepDim,
        "cubic_regression": CubicRegression,
        "sqrt_regression":SqrtRegression,
        "sigmoid_2nn_regression": Sigmoid2nnRegression,
        "linear_regression_random_mapping":LinearRegressionRandomMapping,
        "quadratic_regression_random_mapping":QuadraticRegressionRandomMapping,
        "random_int_mapping": RandomIntMapping,
        "tokenize_random_mapping": TokenizeRandomMapping,
        "quadratic": Quadratic,
        "linear_regression_diversity": LinearRegressionDiversity,
        "power_regression": PowerRegression,
        #"linear_regression_overparameterize": LinearRegressionOverparameterize
    }
    if type(task_name) == list:
        returned_tasks = []
        for task_name_ in task_name:
            #print(task_name_)
            if task_name_ in task_names_to_classes:
                
                task_cls = task_names_to_classes[task_name_]
                if num_tasks is not None:
                    if pool_dict is not None:
                        raise ValueError("Either pool_dict or num_tasks should be None.")
                    pool_dict = task_cls.generate_pool_dict(n_dims, num_tasks, **kwargs)
                #print("task_cls=", task_cls)
                returned_tasks.append(lambda task_cls=task_cls, **args: task_cls(n_dims, batch_size, pool_dict, **args, **kwargs))
                #print(returned_tasks[0](**{}).w_b)
            else:
                print("Unknown task")
                raise NotImplementedError
            
        return returned_tasks
    else:
        if task_name in task_names_to_classes:
            task_cls = task_names_to_classes[task_name]
            if num_tasks is not None:
                if pool_dict is not None:
                    raise ValueError("Either pool_dict or num_tasks should be None.")
                pool_dict = task_cls.generate_pool_dict(n_dims, num_tasks, **kwargs)
            return lambda **args: task_cls(n_dims, batch_size, pool_dict, **args, **kwargs)
        else:
            print("Unknown task")
            raise NotImplementedError


class LinearRegression(Task):
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None, scale=1):
        """scale: a constant by which to scale the randomly sampled weights."""
        super(LinearRegression, self).__init__(n_dims, batch_size, pool_dict, seeds)
        self.scale = scale
        #print(seeds)
        if pool_dict is None and seeds is None:
            self.w_b = torch.randn(self.b_size, self.n_dims, 1)
        elif seeds is not None:
            self.w_b = torch.zeros(self.b_size, self.n_dims, 1)
            generator = torch.Generator()
            assert len(seeds) == self.b_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                self.w_b[i] = torch.randn(self.n_dims, 1, generator=generator)
        else:
            assert "w" in pool_dict
            indices = torch.randperm(len(pool_dict["w"]))[:batch_size]
            self.w_b = pool_dict["w"][indices]

    def evaluate(self, xs_b):
        w_b = self.w_b.to(xs_b.device)
        ys_b = self.scale * (xs_b @ w_b)[:, :, 0]
        return ys_b

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks, **kwargs):  # ignore extra args
        return {"w": torch.randn(num_tasks, n_dims, 1)}

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error
    
class ScalarRegression(Task):
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None):
        """scale: a constant by which to scale the randomly sampled weights."""
        super(ScalarRegression, self).__init__(n_dims, batch_size, pool_dict, seeds)
        if pool_dict is None and seeds is None:
            #print((self.b_size, 1))
            self.scale = torch.empty(self.b_size, 1, 1).uniform_(0.5,1.5) # ori: (1,2)
        elif seeds is not None:
            self.scale = torch.empty(self.b_size, 1, 1)
            generator = torch.Generator()
            assert len(seeds) == self.b_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                self.scale[i] = torch.empty(1).uniform_(0.5,1.5, generator=generator)
        else:
            raise NotImplementedError
    def evaluate(self, xs_b):
        scale = self.scale.to(xs_b.device)
        #print(xs_b.shape) # torch.Size([64, 101, 20])
        ys_b = (scale * xs_b)[:, :, :]
        return ys_b

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks, **kwargs):  # ignore extra args
        return {"w": torch.randn(num_tasks, n_dims, 1)}

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error


class LinearAutoregressionDiversity(Task):
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None, scale=1, n_tasks=100000):
        """scale: a constant by which to scale the randomly sampled weights."""
        super(LinearAutoregressionDiversity, self).__init__(n_dims, batch_size, pool_dict, seeds)
        self.scale = scale

        if pool_dict is None and seeds is None:
            self.w_b = torch.randn(self.b_size, self.n_dims, 1)
        elif seeds is not None:
            self.w_b = torch.zeros(self.b_size, self.n_dims, 1)
            generator = torch.Generator()
            assert len(seeds) == self.b_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                self.w_b[i] = torch.randn(self.n_dims, 1, generator=generator)
        else:
            assert "w" in pool_dict
            indices = torch.randperm(len(pool_dict["w"]))[:batch_size]
            self.w_b = pool_dict["w"][indices]

    def evaluate(self, xs_b):
        w_b = self.w_b.to(xs_b.device)
        ys_b = self.scale * (xs_b @ w_b)[:, :, 0]
        return ys_b

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks, **kwargs):  # ignore extra args
        return {"w": torch.randn(num_tasks, n_dims, 1)}

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error


class SparseLinearRegression(LinearRegression):
    def __init__(
        self,
        n_dims,
        batch_size,
        pool_dict=None,
        seeds=None,
        scale=1,
        sparsity=3,
        valid_coords=None,
    ):
        """scale: a constant by which to scale the randomly sampled weights."""
        super(SparseLinearRegression, self).__init__(
            n_dims, batch_size, pool_dict, seeds, scale
        )
        self.sparsity = sparsity
        if valid_coords is None:
            valid_coords = n_dims
        assert valid_coords <= n_dims

        for i, w in enumerate(self.w_b):
            mask = torch.ones(n_dims).bool()
            if seeds is None:
                perm = torch.randperm(valid_coords)
            else:
                generator = torch.Generator()
                generator.manual_seed(seeds[i])
                perm = torch.randperm(valid_coords, generator=generator)
            mask[perm[:sparsity]] = False
            w[mask] = 0

    def evaluate(self, xs_b):
        w_b = self.w_b.to(xs_b.device)
        ys_b = self.scale * (xs_b @ w_b)[:, :, 0]
        return ys_b

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error


class LinearClassification(LinearRegression):
    def evaluate(self, xs_b):
        ys_b = super().evaluate(xs_b)
        return ys_b.sign()

    @staticmethod
    def get_metric():
        return binary_accuracy

    @staticmethod
    def get_training_metric():
        return binary_cross_entropy


class NoisyLinearRegression(LinearRegression):
    def __init__(
        self,
        n_dims,
        batch_size,
        pool_dict=None,
        seeds=None,
        scale=1,
        noise_std=0,
        renormalize_ys=False,
    ):
        """noise_std: standard deviation of noise added to the prediction."""
        super(NoisyLinearRegression, self).__init__(
            n_dims, batch_size, pool_dict, seeds, scale
        )
        self.noise_std = noise_std
        self.renormalize_ys = renormalize_ys

    def evaluate(self, xs_b):
        ys_b = super().evaluate(xs_b)
        ys_b_noisy = ys_b + torch.randn_like(ys_b) * self.noise_std
        if self.renormalize_ys:
            ys_b_noisy = ys_b_noisy * math.sqrt(self.n_dims) / ys_b_noisy.std()

        return ys_b_noisy


class QuadraticRegression(LinearRegression):
    def evaluate(self, xs_b):
        w_b = self.w_b.to(xs_b.device)
        ys_b_quad = ((xs_b**2) @ w_b)[:, :, 0]
        #         ys_b_quad = ys_b_quad * math.sqrt(self.n_dims) / ys_b_quad.std()
        # Renormalize to Linear Regression Scale
        ys_b_quad = ys_b_quad / math.sqrt(3)
        ys_b_quad = self.scale * ys_b_quad
        return ys_b_quad
    
class CubicRegression(LinearRegression):
    def evaluate(self, xs_b):
        w_b = self.w_b.to(xs_b.device)
        ys_b_quad = ((xs_b**3) @ w_b)[:, :, 0]
        #         ys_b_quad = ys_b_quad * math.sqrt(self.n_dims) / ys_b_quad.std()
        # Renormalize to Linear Regression Scale
        ys_b_quad = ys_b_quad / math.sqrt(3)
        ys_b_quad = self.scale * ys_b_quad
        return ys_b_quad

class SqrtRegression(LinearRegression):
    def evaluate(self, xs_b):
        w_b = self.w_b.to(xs_b.device)
        ys_b_quad = ((xs_b**(1/2)) @ w_b)[:, :, 0]
        #         ys_b_quad = ys_b_quad * math.sqrt(self.n_dims) / ys_b_quad.std()
        # Renormalize to Linear Regression Scale
        ys_b_quad = ys_b_quad / math.sqrt(3)
        ys_b_quad = self.scale * ys_b_quad
        return ys_b_quad

class Relu2nnRegression(Task):
    def __init__(
        self,
        n_dims,
        batch_size,
        pool_dict=None,
        seeds=None,
        scale=1,
        hidden_layer_size=100,
    ):
        """scale: a constant by which to scale the randomly sampled weights."""
        super(Relu2nnRegression, self).__init__(n_dims, batch_size, pool_dict, seeds)
        self.scale = scale
        self.hidden_layer_size = hidden_layer_size

        if pool_dict is None and seeds is None:
            self.W1 = torch.randn(self.b_size, self.n_dims, hidden_layer_size)
            self.W2 = torch.randn(self.b_size, hidden_layer_size, 1)
        elif seeds is not None:
            self.W1 = torch.zeros(self.b_size, self.n_dims, hidden_layer_size)
            self.W2 = torch.zeros(self.b_size, hidden_layer_size, 1)
            generator = torch.Generator()
            assert len(seeds) == self.b_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                self.W1[i] = torch.randn(
                    self.n_dims, hidden_layer_size, generator=generator
                )
                self.W2[i] = torch.randn(hidden_layer_size, 1, generator=generator)
        else:
            assert "W1" in pool_dict and "W2" in pool_dict
            assert len(pool_dict["W1"]) == len(pool_dict["W2"])
            indices = torch.randperm(len(pool_dict["W1"]))[:batch_size]
            self.W1 = pool_dict["W1"][indices]
            self.W2 = pool_dict["W2"][indices]

    def evaluate(self, xs_b):
        W1 = self.W1.to(xs_b.device)
        W2 = self.W2.to(xs_b.device)
        # Renormalize to Linear Regression Scale
        ys_b_nn = (torch.nn.functional.relu(xs_b @ W1) @ W2)[:, :, 0]
        ys_b_nn = ys_b_nn * math.sqrt(2 / self.hidden_layer_size)
        ys_b_nn = self.scale * ys_b_nn
        #         ys_b_nn = ys_b_nn * math.sqrt(self.n_dims) / ys_b_nn.std()
        return ys_b_nn

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks, hidden_layer_size=4, **kwargs):
        return {
            "W1": torch.randn(num_tasks, n_dims, hidden_layer_size),
            "W2": torch.randn(num_tasks, hidden_layer_size, 1),
        }

    @staticmethod
    def get_metric(): 
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error
    
class Sigmoid2nnRegression(Relu2nnRegression):
    def evaluate(self, xs_b):
        W1 = self.W1.to(xs_b.device)
        W2 = self.W2.to(xs_b.device)
        # Renormalize to Linear Regression Scale
        ys_b_nn = (torch.nn.functional.sigmoid(xs_b @ W1) @ W2)[:, :, 0]
        ys_b_nn = ys_b_nn * math.sqrt(2 / self.hidden_layer_size)
        ys_b_nn = self.scale * ys_b_nn
        #         ys_b_nn = ys_b_nn * math.sqrt(self.n_dims) / ys_b_nn.std()
        return ys_b_nn

class ReluMultiLayernnRegression(Task):
    def __init__(
        self,
        n_dims,
        batch_size,
        pool_dict=None,
        seeds=None,
        scale=1,
        hidden_layer_size=100,
        depth=4
    ):
        """scale: a constant by which to scale the randomly sampled weights."""
        super(ReluMultiLayernnRegression, self).__init__(n_dims, batch_size, pool_dict, seeds)
        self.scale = scale
        self.hidden_layer_size = hidden_layer_size
        self.depth = depth

        if pool_dict is None and seeds is None:
            self.W1 = torch.randn(self.b_size, self.n_dims, hidden_layer_size)
            self.hidden_W = torch.randn(depth-2, self.b_size, hidden_layer_size, hidden_layer_size)
            self.W2 = torch.randn(self.b_size, hidden_layer_size, 1)
        elif seeds is not None:
            self.W1 = torch.zeros(self.b_size, self.n_dims, hidden_layer_size)
            self.hidden_W = torch.zeros(depth-2, self.b_size, hidden_layer_size, hidden_layer_size)
            self.W2 = torch.zeros(self.b_size, hidden_layer_size, 1)
            generator = torch.Generator()
            assert len(seeds) == self.b_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                self.W1[i] = torch.randn(
                    self.n_dims, hidden_layer_size, generator=generator
                )
                self.hidden_W[i] = torch.randn(depth-2, self.b_size, hidden_layer_size, hidden_layer_size)
                self.W2[i] = torch.randn(hidden_layer_size, 1, generator=generator)
        else:
            raise NotImplementedError
            assert "W1" in pool_dict and "W2" in pool_dict
            assert len(pool_dict["W1"]) == len(pool_dict["W2"])
            indices = torch.randperm(len(pool_dict["W1"]))[:batch_size]
            self.W1 = pool_dict["W1"][indices]
            self.W2 = pool_dict["W2"][indices]

    def evaluate(self, xs_b):
        W1 = self.W1.to(xs_b.device)
        hidden_W = self.hidden_W.to(xs_b.device)
        W2 = self.W2.to(xs_b.device)
        # Renormalize to Linear Regression Scale
        ys_b_nn = torch.nn.functional.relu(xs_b @ W1) 
        for l in range(self.depth - 2):
            ys_b_nn = torch.nn.functional.relu(ys_b_nn @ hidden_W[l])
        ys_b_nn = (ys_b_nn @ W2)[:, :, 0]
        ys_b_nn = ys_b_nn * math.sqrt(2 / self.hidden_layer_size)
        ys_b_nn = self.scale * ys_b_nn
        #         ys_b_nn = ys_b_nn * math.sqrt(self.n_dims) / ys_b_nn.std()
        return ys_b_nn

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks, hidden_layer_size=4, **kwargs):
        return {
            "W1": torch.randn(num_tasks, n_dims, hidden_layer_size),
            "W2": torch.randn(num_tasks, hidden_layer_size, 1),
        }

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error
    

class DecisionTree(Task):
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None, depth=4):

        super(DecisionTree, self).__init__(n_dims, batch_size, pool_dict, seeds)
        self.depth = depth

        if pool_dict is None:

            # We represent the tree using an array (tensor). Root node is at index 0, its 2 children at index 1 and 2...
            # dt_tensor stores the coordinate used at each node of the decision tree.
            # Only indices corresponding to non-leaf nodes are relevant
            self.dt_tensor = torch.randint(
                low=0, high=n_dims, size=(batch_size, 2 ** (depth + 1) - 1)
            )

            # Target value at the leaf nodes.
            # Only indices corresponding to leaf nodes are relevant.
            self.target_tensor = torch.randn(self.dt_tensor.shape)
        elif seeds is not None:
            self.dt_tensor = torch.zeros(batch_size, 2 ** (depth + 1) - 1)
            self.target_tensor = torch.zeros_like(dt_tensor)
            generator = torch.Generator()
            assert len(seeds) == self.b_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                self.dt_tensor[i] = torch.randint(
                    low=0,
                    high=n_dims - 1,
                    size=2 ** (depth + 1) - 1,
                    generator=generator,
                )
                self.target_tensor[i] = torch.randn(
                    self.dt_tensor[i].shape, generator=generator
                )
        else:
            raise NotImplementedError

    def evaluate(self, xs_b):
        dt_tensor = self.dt_tensor.to(xs_b.device)
        target_tensor = self.target_tensor.to(xs_b.device)
        ys_b = torch.zeros(xs_b.shape[0], xs_b.shape[1], device=xs_b.device)
        for i in range(xs_b.shape[0]):
            xs_bool = xs_b[i] > 0
            # If a single decision tree present, use it for all the xs in the batch.
            if self.b_size == 1:
                dt = dt_tensor[0]
                target = target_tensor[0]
            else:
                dt = dt_tensor[i]
                target = target_tensor[i]

            cur_nodes = torch.zeros(xs_b.shape[1], device=xs_b.device).long()
            for j in range(self.depth):
                cur_coords = dt[cur_nodes]
                cur_decisions = xs_bool[torch.arange(xs_bool.shape[0]), cur_coords]
                cur_nodes = 2 * cur_nodes + 1 + cur_decisions

            ys_b[i] = target[cur_nodes]

        return ys_b

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks, hidden_layer_size=4, **kwargs):
        raise NotImplementedError

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error

class AutoRegression(Task):
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None, scale=1):
        """scale: a constant by which to scale the randomly sampled weights."""
        super(AutoRegression, self).__init__(n_dims, batch_size, pool_dict, seeds)


    def evaluate(self, xs_b):
        if torch.sum(xs_b[1,1:])==0: # the Ys in the autoregression task is a scalar, only the first dim of Y is non-zero
            return xs_b[:,1:,0]
        else:
            return xs_b[:,1:,:]

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks, **kwargs):  # ignore extra args
        return {"w": torch.randn(num_tasks, n_dims, 1)}

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error
    
class LinearRegressionBatchShare(Task):
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None, scale=1):
        """scale: a constant by which to scale the randomly sampled weights."""
        super(LinearRegressionBatchShare, self).__init__(n_dims, batch_size, pool_dict, seeds)
        self.scale = scale

        if pool_dict is None and seeds is None:
            self.w_b = torch.randn(1, self.n_dims, 1)
        elif seeds is not None:
            self.w_b = torch.zeros(1, self.n_dims, 1)
            generator = torch.Generator()
            assert len(seeds) == self.b_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                self.w_b[i] = torch.randn(self.n_dims, 1, generator=generator)
        else:
            assert "w" in pool_dict
            indices = torch.randperm(len(pool_dict["w"]))[:batch_size]
            self.w_b = pool_dict["w"][indices]

    def evaluate(self, xs_b):
        w_b = self.w_b.to(xs_b.device)
        ys_b = self.scale * (xs_b @ w_b)[:, :, 0]
        return ys_b

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks, **kwargs):  # ignore extra args
        return {"w": torch.randn(num_tasks, n_dims, 1)}

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error
    
class QuadraticRegressionBatchShare(LinearRegressionBatchShare):
    def evaluate(self, xs_b):
        w_b = self.w_b.to(xs_b.device)
        ys_b_quad = ((xs_b**2) @ w_b)[:, :, 0]
        #         ys_b_quad = ys_b_quad * math.sqrt(self.n_dims) / ys_b_quad.std()
        # Renormalize to Linear Regression Scale
        ys_b_quad = ys_b_quad / math.sqrt(3)
        ys_b_quad = self.scale * ys_b_quad
        return ys_b_quad
    
class LinearQuadraticRegression(Task):
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None, scale=1):
        """scale: a constant by which to scale the randomly sampled weights."""
        super(LinearQuadraticRegression, self).__init__(n_dims, batch_size, pool_dict, seeds)
        self.scale = scale

        if pool_dict is None and seeds is None:
            self.w_b1 = torch.randn(self.b_size, self.n_dims, 1)
            self.w_b2 = torch.randn(self.b_size, self.n_dims, 1)
        elif seeds is not None:
            self.w_b1 = torch.zeros(self.b_size, self.n_dims, 1)
            self.w_b2 = torch.zeros(self.b_size, self.n_dims, 1)
            generator = torch.Generator()
            assert len(seeds) == self.b_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                self.w_b1[i] = torch.randn(self.n_dims, 1, generator=generator)
                self.w_b2[i] = torch.randn(self.n_dims, 1, generator=generator)
        else:
            raise NotImplementedError

    def evaluate(self, xs_b):
        w_b1 = self.w_b1.to(xs_b.device)
        w_b2 = self.w_b2.to(xs_b.device)
        ys_b1 = self.scale * (xs_b @ w_b1)[:, :, 0]
        ys_b2 = self.scale * ((xs_b**2) @ w_b2)[:, :, 0]
        return ys_b1 + ys_b2

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks, **kwargs):  # ignore extra args
        return {"w": torch.randn(num_tasks, n_dims, 1)}

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error
    
    
class SinRegression(LinearRegression):
    def evaluate(self, xs_b):
        w_b = self.w_b.to(xs_b.device)
        ys_b_quad = ((torch.sin(xs_b*torch.pi)) @ w_b)[:, :, 0]
        #         ys_b_quad = ys_b_quad * math.sqrt(self.n_dims) / ys_b_quad.std()
        # Renormalize to Linear Regression Scale
        ys_b_quad = ys_b_quad / ((torch.e**(-2)+1)/2)
        ys_b_quad = self.scale * ys_b_quad
        return ys_b_quad
    
    
    
class QuadraticSum(Task):
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None, scale=1):
        """scale: a constant by which to scale the randomly sampled weights."""
        super(QuadraticSum, self).__init__(n_dims, batch_size, pool_dict, seeds)
        self.scale = scale
        self.n_dims
    def evaluate(self, xs_b):
        return torch.sum(xs_b**2, dim=-1)

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks, **kwargs):  # ignore extra args
        return {"w": torch.randn(num_tasks, n_dims, 1)}

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error
    
    
class LinearRegressionKeepDim(LinearRegression):
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None, scale=1, rank=3):
        """scale: a constant by which to scale the randomly sampled weights."""
        super(LinearRegressionKeepDim, self).__init__(n_dims, batch_size, pool_dict, seeds)
        self.scale = scale
        rank = min(rank, n_dims)
        self.rank = rank
        if pool_dict is None and seeds is None:
            self.A = torch.randn(self.b_size, self.n_dims, rank)
            self.B = torch.randn(self.b_size, rank, self.n_dims)
        elif seeds is not None:
            self.A = torch.zeros(self.b_size, self.n_dims, rank)
            self.B = torch.zeros(self.b_size, rank, self.n_dims)
            generator = torch.Generator()
            assert len(seeds) == self.b_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                self.A[i] = torch.randn(self.n_dims, rank, generator=generator)
                self.B[i] = torch.randn(rank, self.n_dims, generator=generator)
        else:
            raise NotImplementedError

    def evaluate(self, xs_b):
        w_b = torch.bmm( self.A.to(xs_b.device) , self.B.to(xs_b.device) )
        ys_b = self.scale * (xs_b @ w_b)[:, :, :]
        return ys_b

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks, **kwargs):  # ignore extra args
        return {"w": torch.randn(num_tasks, n_dims, 1)}

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error
    
    
class QuadraticRegressionKeepDim(LinearRegressionKeepDim):
    def evaluate(self, xs_b):
        w_b = torch.bmm( self.A.to(xs_b.device) , self.B.to(xs_b.device) )
        ys_b = self.scale * ((xs_b**2) @ w_b)[:, :, 0]
        return ys_b

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks, **kwargs):  # ignore extra args
        return {"w": torch.randn(num_tasks, n_dims, 1)}

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error
    
class LinearRegressionRandomMapping(LinearRegression):
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None, scale=1, shift_low=100, shift_high=200):
        """scale: a constant by which to scale the randomly sampled weights."""
        super(LinearRegressionRandomMapping, self).__init__(n_dims, batch_size, pool_dict, seeds)
        self.max_value = 10000  # 
        base_mapping = torch.arange(self.max_value).unsqueeze(0).repeat(batch_size, 1)
        # Shuffle each row independently
        random_mappings = base_mapping.clone()
        #for i in range(batch_size):
        #    random_mappings[i] = random_mappings[i][torch.randperm(self.max_value)]
            
        shift = torch.randint(shift_low, shift_high, (batch_size,)).unsqueeze(1).repeat(1, self.max_value)
        # when dim = 7, scale = 1.0, x@w-x@w.min() almost lies in [1, 25] 
        # so shift > 25 should be OOD
         
        self.random_mappings = (random_mappings + shift) % self.max_value
        self.embedding = torch.load('/data1/qxwang/codes/in-context-learning/results/embeddings/embedding_sz10000_dim20.pt')

    def evaluate(self, xs_b):
        w_b = self.w_b.to(xs_b.device)
        ys_b = self.scale * (xs_b @ w_b)
        scale = None
        
        scale=0.4
       
        y_int = torch.floor(ys_b*scale) # mapping to a random int token, the interval size is 1.0
        
        y_norm_int = (y_int - torch.min(y_int)).long().repeat(1,1,self.max_value)
        #print("y_norm_int",y_norm_int)
        bsz, len, _ = ys_b.shape
        #print(self.random_mappings)
        expanded_mappings = self.random_mappings.unsqueeze(1).expand(-1, len, -1)
        random_mapped_id = torch.gather(expanded_mappings, 2, y_norm_int)[:,:,0].unsqueeze(-1)
        expanded_embedding = self.embedding.unsqueeze(1).expand(-1, len, -1)
        return torch.gather(expanded_embedding, dim=0, index=(random_mapped_id).repeat(1, 1, self.n_dims))
        #print("final", torch.gather(expanded_mappings, 2, y_norm_int).squeeze(-1)[:,:,0])
        #return torch.gather(expanded_mappings, 2, y_norm_int).squeeze(-1)[:,:,0]
    
    @staticmethod
    def get_metric():
        return squared_error
        #return multi_accuracy

    @staticmethod
    def get_training_metric():
        return mean_squared_error
        #return multi_cross_entropy
        
class QuadraticRegressionRandomMapping(LinearRegression):
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None, scale=1, shift_low=100, shift_high=200):
        """scale: a constant by which to scale the randomly sampled weights."""
        super(QuadraticRegressionRandomMapping, self).__init__(n_dims, batch_size, pool_dict, seeds)
        self.max_value = 10000  # 
        base_mapping = torch.arange(self.max_value).unsqueeze(0).repeat(batch_size, 1)
        # Shuffle each row independently
        random_mappings = base_mapping.clone()
        #for i in range(batch_size):
        #    random_mappings[i] = random_mappings[i][torch.randperm(self.max_value)]
            
        shift = torch.randint(shift_low, shift_high, (batch_size,)).unsqueeze(1).repeat(1, self.max_value)
        # when dim = 7, scale = 1.0, x@w-x@w.min() almost lies in [1, 25] 
        # so shift > 25 should be OOD
         
        self.random_mappings = (random_mappings + shift) % self.max_value
        self.embedding = torch.load('/data1/qxwang/codes/in-context-learning/results/embeddings/embedding_sz10000_dim20.pt')

    def evaluate(self, xs_b):
        w_b = self.w_b.to(xs_b.device)
        ys_b = self.scale * (xs_b**2 @ w_b)
        scale = None
        
        scale=0.4
       
        y_int = torch.floor(ys_b*scale) # mapping to a random int token, the interval size is 1.0
        
        y_norm_int = (y_int - torch.min(y_int)).long().repeat(1,1,self.max_value)
        #print("y_norm_int",y_norm_int)
        bsz, len, _ = ys_b.shape
        #print(self.random_mappings)
        expanded_mappings = self.random_mappings.unsqueeze(1).expand(-1, len, -1)
        random_mapped_id = torch.gather(expanded_mappings, 2, y_norm_int)[:,:,0].unsqueeze(-1)
        expanded_embedding = self.embedding.unsqueeze(1).expand(-1, len, -1)
        return torch.gather(expanded_embedding, dim=0, index=(random_mapped_id).repeat(1, 1, self.n_dims))
        #print("final", torch.gather(expanded_mappings, 2, y_norm_int).squeeze(-1)[:,:,0])
        #return torch.gather(expanded_mappings, 2, y_norm_int).squeeze(-1)[:,:,0]
    
    @staticmethod
    def get_metric():
        return squared_error
        #return multi_accuracy

    @staticmethod
    def get_training_metric():
        return mean_squared_error
        #return multi_cross_entropy        

    
class RandomIntMapping(Task):
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None, scale=1, num_classes=10, shift_low=500, shift_high=600):
        """scale: a constant by which to scale the randomly sampled weights."""
        super(RandomIntMapping, self).__init__(n_dims, batch_size, pool_dict, seeds)
        self.n_dims = n_dims
        self.max_value = num_classes

        base_mapping = torch.arange(self.max_value).unsqueeze(0).repeat(batch_size, 1)

        random_mappings = base_mapping.clone()

        shift = torch.randint(shift_low, shift_high, (batch_size,)).unsqueeze(1).repeat(1, self.max_value)
        self.random_mappings = (random_mappings + shift) % self.max_value
        #print(self.random_mappings)
        self.embedding = torch.load('/data1/qxwang/codes/in-context-learning/results/embeddings/embedding_sz10000_dim20.pt')

    def evaluate(self, xs_b):
        # xs_b [bsz, n_points, 1]
        #print("xs_b.shape",xs_b.shape, xs_b)
        xs_b_expand = xs_b.long().repeat(1,1,self.max_value)
        bsz, len, _ = xs_b.shape
        expanded_random_mappings = self.random_mappings.unsqueeze(1).expand(-1, len, -1)
        random_mapped_xs_b = torch.gather(expanded_random_mappings, 2, xs_b_expand)[:,:,0].unsqueeze(-1)
        #print("expanded_mappings.shape=",expanded_mappings, "xs_b_expand.shape=",xs_b_expand)
        #print("label=",torch.gather(expanded_mappings, 2, xs_b_expand).squeeze(-1)[:,:,0])
        #print("random_mapped_xs_b=",random_mapped_xs_b)
        expanded_embedding = self.embedding.unsqueeze(1).expand(-1, len, -1)
        xs_embedded = torch.gather(expanded_embedding, dim=0, index=(random_mapped_xs_b).repeat(1, 1, self.n_dims))
        #print("xs_embedded.shape", xs_embedded.shape) [64,11,20]
        #print("xs_b=",xs_b)
        #print("gt=",xs_embedded)
        return xs_embedded
    
    @staticmethod
    def get_metric():
        return squared_error
        #return multi_accuracy

    @staticmethod
    def get_training_metric():
        return mean_squared_error
        #return multi_cross_entropy
    
class TokenizeRandomMapping(Task):
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None, scale=1, num_classes=10):
        """scale: a constant by which to scale the randomly sampled weights."""
        super(TokenizeRandomMapping, self).__init__(n_dims, batch_size, pool_dict, seeds)
        self.max_value = num_classes # 
        base_mapping = torch.arange(self.max_value).unsqueeze(0).repeat(batch_size, 1)
        # Shuffle each row independently
        random_mappings = base_mapping.clone()
        for i in range(batch_size):
            random_mappings[i] = random_mappings[i][torch.randperm(self.max_value)]
        self.random_mappings = random_mappings

    def evaluate(self, xs_b):
        ys_b = xs_b
        scale = None
        if self.max_value == 10:
            scale = 0.1
        elif self.max_value == 20:
            scale = 0.3
        elif self.max_value == 70:
            scale = 1.0
        y_int = torch.floor(ys_b*scale) # mapping to a random int token, the interval size is 1.0
        y_norm_int = (y_int - torch.min(y_int)).long().repeat(1,1,self.max_value)
        bsz, len, _ = ys_b.shape
        expanded_mappings = self.random_mappings.unsqueeze(1).expand(-1, len, -1)
        return torch.gather(expanded_mappings, 2, y_norm_int).squeeze(-1)[:,:,0]
    
    @staticmethod
    def get_metric():
        return multi_accuracy

    @staticmethod
    def get_training_metric():
        return multi_cross_entropy
    
    
class Quadratic(Task):
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None, scale=1, rank=3):
        """scale: a constant by which to scale the randomly sampled weights."""
        super(Quadratic, self).__init__(n_dims, batch_size, pool_dict, seeds)
        
    def evaluate(self, xs_b):
        return xs_b**2

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks, **kwargs):  # ignore extra args
        return {"w": torch.randn(num_tasks, n_dims, 1)}

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error
    
class LinearRegressionDiversity(Task):
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None, scale=1, num_pretraining_tasks=100):
        """scale: a constant by which to scale the randomly sampled weights."""
        super(LinearRegressionDiversity, self).__init__(n_dims, batch_size, pool_dict, seeds)
        self.scale = scale
        self.num_tasks = num_pretraining_tasks
        self.n_dims = n_dims
        self.bsz = batch_size
        if pool_dict is None and seeds is None:
            self.w_b = torch.randn(self.num_tasks, self.n_dims, 1)
        elif seeds is not None:
            self.w_b = torch.zeros(self.num_tasks, self.n_dims, 1)
            generator = torch.Generator()
            assert len(seeds) == self.num_tasks
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                self.w_b[i] = torch.randn(self.n_dims, 1, generator=generator)
        else:
            assert "w" in pool_dict
            indices = torch.randperm(len(pool_dict["w"]))[:batch_size]
            self.w_b = pool_dict["w"][indices]

    def evaluate(self, xs_b):
        random_id = torch.randint(0, self.num_tasks, (self.bsz, 1, 1))
        sampled_ws = torch.gather(self.w_b, dim=0, index=random_id.repeat(1, self.n_dims, 1))
        sampled_ws = sampled_ws.to(xs_b.device)
        ys_b = self.scale * (xs_b @ sampled_ws)[:, :, 0]
        return ys_b

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks, **kwargs):  # ignore extra args
        return {"w": torch.randn(num_tasks, n_dims, 1)}

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error
    
    
class PowerRegression(LinearRegression):
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None, scale=1, powers=None):
        super(PowerRegression, self).__init__(n_dims, batch_size, pool_dict, seeds)
        self.powers=powers

        
    def evaluate(self, xs_b):
        w_b = self.w_b.to(xs_b.device)
        power=np.random.choice(self.powers)
        ys_b = ((xs_b**power) @ w_b)[:, :, 0]
        #         ys_b_quad = ys_b_quad * math.sqrt(self.n_dims) / ys_b_quad.std()
        # Renormalize to Linear Regression Scale
        ys_b = ys_b / math.sqrt(3)
        return self.scale * ys_b
