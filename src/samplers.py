import math
import random
import torch


class DataSampler:
    def __init__(self, n_dims):
        self.n_dims = n_dims

    def sample_xs(self):
        raise NotImplementedError


def get_data_sampler(data_name, n_dims, fixed_embedding=None, **kwargs):
    names_to_classes = {
        "gaussian": GaussianSampler,
        "gaussian_quadratic":GaussianQuadraticSampler,
        "pos_gaussian": PosGaussianSampler,
        "random_int": RandomIntSampler,
        "autoregression": AutoregressiveSampler
    }
    if type(data_name) == list:
        returned_datas = []
        for data_name_ in data_name:
            if data_name_ in names_to_classes:
                sampler_cls = names_to_classes[data_name_]
                if data_name_ == 'gaussian':
                    kwargs['fixed_embedding'] = fixed_embedding
                returned_datas.append( sampler_cls(n_dims, **kwargs) )
            else:
                print("Unknown sampler")
                raise NotImplementedError
        return returned_datas
    else:
        if data_name in names_to_classes:
            sampler_cls = names_to_classes[data_name]
            if data_name == 'gaussian':
                kwargs['fixed_embedding'] = fixed_embedding
            return sampler_cls(n_dims, **kwargs)
        else:
            print("Unknown sampler")
            raise NotImplementedError


def sample_transformation(eigenvalues, normalize=False):
    n_dims = len(eigenvalues)
    U, _, _ = torch.linalg.svd(torch.randn(n_dims, n_dims))
    t = U @ torch.diag(eigenvalues) @ torch.transpose(U, 0, 1)
    if normalize:
        norm_subspace = torch.sum(eigenvalues**2)
        t *= math.sqrt(n_dims / norm_subspace)
    return t


class GaussianSampler(DataSampler):
    def __init__(self, n_dims, bias=None, scale=None, fixed_embedding=None):
        super().__init__(n_dims)
        self.bias = bias
        self.scale = scale
        self.fixed_embedding = fixed_embedding
        self.n_dims = n_dims
        #print(fixed_embedding)

    def sample_xs(self, n_points, b_size, n_dims_truncated=None, seeds=None):
        if self.fixed_embedding is not None:
            num_embeddings = self.fixed_embedding.shape[0] 
            sampled_id = torch.randint(0, num_embeddings, (b_size, n_points)).unsqueeze(-1).repeat(1, 1, self.n_dims) # [bsz, n_points, dim]
            expaned_fixed_embedding = self.fixed_embedding.unsqueeze(1).repeat(1, n_points, 1) # [bsz, n_points, dim]
            xs_b = torch.gather(expaned_fixed_embedding, dim=0, index=sampled_id)
        else:
            if seeds is None:
                xs_b = torch.randn(b_size, n_points, self.n_dims)
            else:
                xs_b = torch.zeros(b_size, n_points, self.n_dims)
                generator = torch.Generator()
                assert len(seeds) == b_size
                for i, seed in enumerate(seeds):
                    generator.manual_seed(seed)
                    xs_b[i] = torch.randn(n_points, self.n_dims, generator=generator)
        if self.scale is not None:
            xs_b = xs_b @ self.scale
        if self.bias is not None:
            xs_b += self.bias
        if n_dims_truncated is not None:
            xs_b[:, :, n_dims_truncated:] = 0
        return xs_b
    
class GaussianQuadraticSampler(DataSampler):
    def __init__(self, n_dims, bias=None, scale=None, fixed_embedding=None):
        super().__init__(n_dims)
        self.bias = bias
        self.scale = scale
        self.fixed_embedding = fixed_embedding
        self.n_dims = n_dims
        #print(fixed_embedding)

    def sample_xs(self, n_points, b_size, n_dims_truncated=None, seeds=None):
        if self.fixed_embedding is not None:
            num_embeddings = self.fixed_embedding.shape[0] 
            sampled_id = torch.randint(0, num_embeddings, (b_size, n_points)).unsqueeze(-1).repeat(1, 1, self.n_dims) # [bsz, n_points, dim]
            expaned_fixed_embedding = self.fixed_embedding.unsqueeze(1).repeat(1, n_points, 1) # [bsz, n_points, dim]
            xs_b = torch.gather(expaned_fixed_embedding, dim=0, index=sampled_id)
        else:
            if seeds is None:
                xs_b = torch.randn(b_size, n_points, self.n_dims)
            else:
                xs_b = torch.zeros(b_size, n_points, self.n_dims)
                generator = torch.Generator()
                assert len(seeds) == b_size
                for i, seed in enumerate(seeds):
                    generator.manual_seed(seed)
                    xs_b[i] = torch.randn(n_points, self.n_dims, generator=generator)
        if self.scale is not None:
            xs_b = xs_b @ self.scale
        if self.bias is not None:
            xs_b += self.bias
        if n_dims_truncated is not None:
            xs_b[:, :, n_dims_truncated:] = 0
        return xs_b**2
    
class PosGaussianSampler(DataSampler):
    def __init__(self, n_dims, bias=None, scale=None, fixed_embedding=None):
        super().__init__(n_dims)
        self.bias = bias
        self.scale = scale
        self.fixed_embedding = fixed_embedding
        self.n_dims = n_dims
        #print(fixed_embedding)

    def sample_xs(self, n_points, b_size, n_dims_truncated=None, seeds=None):
        if self.fixed_embedding is not None:
            num_embeddings = self.fixed_embedding.shape[0] 
            sampled_id = torch.randint(0, num_embeddings, (b_size, n_points)).unsqueeze(-1).repeat(1, 1, self.n_dims) # [bsz, n_points, dim]
            expaned_fixed_embedding = self.fixed_embedding.unsqueeze(1).repeat(1, n_points, 1) # [bsz, n_points, dim]
            xs_b = torch.gather(expaned_fixed_embedding, dim=0, index=sampled_id)
        else:
            if seeds is None:
                xs_b = torch.randn(b_size, n_points, self.n_dims).abs()
            else:
                xs_b = torch.zeros(b_size, n_points, self.n_dims)
                generator = torch.Generator()
                assert len(seeds) == b_size
                for i, seed in enumerate(seeds):
                    generator.manual_seed(seed)
                    xs_b[i] = torch.randn(n_points, self.n_dims, generator=generator).abs()
        if self.scale is not None:
            xs_b = xs_b @ self.scale
        if self.bias is not None:
            xs_b += self.bias
        if n_dims_truncated is not None:
            xs_b[:, :, n_dims_truncated:] = 0
        return xs_b

class RandomIntSampler(DataSampler):
    def __init__(self, n_dims,  max_value=5):
        super().__init__(n_dims)
        self.max_value = max_value
        self.n_dims = n_dims
        #print(fixed_embedding)
        self.embedding = torch.load('/data1/qxwang/codes/in-context-learning/results/embeddings/embedding_sz10000_dim20.pt')
    def sample_xs(self, n_points, b_size, n_dims_truncated=None, seeds=None):
        if seeds is None:
            xs_b = torch.randint(0, self.max_value, (b_size, n_points, 1))
        else:
            xs_b = torch.zeros(b_size, n_points, 1)
            generator = torch.Generator()
            assert len(seeds) == b_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                xs_b[i] = torch.randint(0, self.max_value, (n_points, 1), generator=generator)
        #xs_b=torch.cat((xs_b, torch.zeros(b_size, n_points, self.n_dims-1)), dim=-1)
        expanded_embedding = self.embedding.unsqueeze(1).expand(-1, n_points, -1)
        xs_embedded = torch.gather(expanded_embedding, dim=0, index=xs_b.repeat(1, 1, self.n_dims))
        if n_dims_truncated is not None:
            xs_embedded[:, :, n_dims_truncated:] = 0
        #print("xs_b",xs_b, "xs_embedded",xs_embedded)
        return xs_b, xs_embedded
    


    
class AutoregressiveSampler(DataSampler):
    def __init__(self, n_dims, n_embedding={}, x_y_order='random', add_task_tokens=False, finite_start_tokens=False, \
        n_start_tokens=3, task_list=[], min_out_degree=1, max_out_degree=4, total_len=21, x_add_ratio=0.4, y_add_ratio=0.4):
        super().__init__(n_dims)
        generator = torch.Generator()
        generator.manual_seed(43) # global seed for sampling the adj matrix, tasks
        
        self.num_x_embedding = n_embedding.x
        self.vocab_size = n_embedding.irrelevant # including X, Y (Task) token
        self.total_len = total_len
        self.x_embeddings = torch.randn((self.num_x_embedding, n_dims), generator=generator)
        self.irr_embeddings = torch.randn((self.vocab_size, n_dims), generator=generator)
        self.task_functions = [] #
        self.task_list=task_list # what tasks will be included in the training data
        self.seed = None
        self.x_add_ratio = x_add_ratio
        self.y_add_ratio = y_add_ratio
        num_tasks = n_embedding.task # vocabulary size of task tokens, (which is also the number of task functions of each task)
        self.task_classes ={
            "scalar_projection":(ScalarProjectionBatchShare, {"low": 0.5, "high":1.5, "generator":generator}), # []
            "linear_regression":(LinearRegressionBatchShare,{"mean": 0.0, "std":1.0, "dim":n_dims, "generator":generator}),
            "linear_projection":(LinearProjectionBatchShare,{"mean": 0.0, "std":1.0, "dim":n_dims, "generator":generator})
        }
        
        if add_task_tokens:
            self.task_embeddings = torch.randn((num_tasks, n_dims), generator=generator)
            
        for task_name in task_list:
            for t in range(num_tasks): #  sample num_task times for task task_name
                task_class, task_args = self.task_classes[task_name]
                self.task_functions.append(task_class(**task_args) )
        torch.save(self.x_embeddings, f'/data1/qxwang/codes/in-context-learning/src/conf/autoregression_word_embeddings/x_embeddings_{self.num_x_embedding}_dim{n_dims}.pt')
        torch.save(self.x_embeddings, f'/data1/qxwang/codes/in-context-learning/src/conf/autoregression_word_embeddings/func{num_tasks}.pt')
        #print("self.x_embeddings",self.x_embeddings[:10])
        #print("task_functions",self.task_functions[0].w,self.task_functions[1].w,self.task_functions[2].w)
        self.adj_matrix = self.generate_adj_matrix(self.vocab_size, min_out_degree, max_out_degree)


    def sample_xs(self, n_points, b_size, n_dims_truncated=None, seeds=None):
        self.seed = seeds # batch seed for torch random processes
        generator = torch.Generator()
        generator.manual_seed(seeds)
        task_func = random.sample(self.task_functions, 1)[0] # sample a task function for a batch
        
        batch_paths = self.sample_batch_paths(total_len=n_points, bsz=b_size, seed=seeds)
        batch_data = self.irr_embeddings[batch_paths]
        #print('1',batch_data)
        batch_x_embeddings = self.x_embeddings[ torch.randint(0, self.num_x_embedding, (b_size,), generator=generator) ] #[bsz, embed_dim]
        #print('x', batch_x_embeddings)
        batch_y_embeddings = task_func.predict(batch_x_embeddings)#[bsz, embed_dim]
        #print('y', batch_y_embeddings)
        batch_x_embeddings = batch_x_embeddings.unsqueeze(1).expand(-1, n_points, -1) #[bsz, path_len, embed_dim]
        batch_data = torch.where((batch_paths == 0).unsqueeze(-1), batch_x_embeddings, batch_data)
        batch_y_embeddings = batch_y_embeddings.unsqueeze(1).expand(-1, n_points, -1) #[bsz, path_len, embed_dim]
        batch_data = torch.where((batch_paths == 1).unsqueeze(-1), batch_y_embeddings, batch_data)
        #print('2',batch_data)
        return batch_data

    def generate_adj_matrix(self, num_tokens, min_out_degree, max_out_degree):
        adj_matrix = torch.zeros(num_tokens, num_tokens)
        random_nodes_connected_to_x = random.sample(range(2, num_tokens), math.ceil((num_tokens-2)*self.x_add_ratio)) # nodes that have non-zero prob to go to X
        if 0 in random_nodes_connected_to_x:
            random_nodes_connected_to_x.remove(0)
        random_nodes_connected_to_y = random.sample(range(2, num_tokens), math.ceil((num_tokens-2)*self.y_add_ratio)) # nodes that have non-zero prob to go to Y
        if 1 in random_nodes_connected_to_y:
            random_nodes_connected_to_y.remove(1)
        #print(random_nodes_connected_to_x, random_nodes_connected_to_y)
        for i in range(num_tokens):
            out_degree = random.randint(min_out_degree, max_out_degree)
            next_node_candidates = list(range(num_tokens))
            next_node_candidates.remove(i) # can't go to itself
            next_nodes = random.sample(next_node_candidates, out_degree)
            adj_matrix[i, next_nodes] = torch.empty_like(torch.tensor(next_nodes), dtype=torch.float).uniform_(0.1, 1.0) 
            if i in random_nodes_connected_to_x:
                adj_matrix[i,0] = adj_matrix[i,0].uniform_(0.7, 1) # experiments before 2024/6/30: (0.3, 1)
            if i in random_nodes_connected_to_y:
                adj_matrix[i,1] = adj_matrix[i,1].uniform_(0.9, 1) # experiments before 2024/6/30: (0.3, 1)
            adj_matrix[i] = adj_matrix[i]/adj_matrix[i].sum(-1)
        #print(adj_matrix)
        return adj_matrix 

    def sample_batch_paths(self, total_len, bsz, seed):
        #print(total_len)
        generator = torch.Generator()
        generator.manual_seed(seed)
        prev_nodes =  torch.randint(0, self.vocab_size, (bsz,), generator=generator) 
        paths = torch.zeros(bsz, total_len, dtype=torch.long)
        #print(torch.nn.functional.one_hot(prev_nodes, num_classes=self.vocab_size))
        for l in range(total_len):
            next_node_distribution = torch.matmul(torch.nn.functional.one_hot(prev_nodes, num_classes=self.vocab_size).float(), self.adj_matrix)
            paths[:, l] = prev_nodes
            prev_nodes = torch.multinomial(next_node_distribution, 1, generator=generator).squeeze(-1)
        #print("sequence of a batch:",paths)
        #print("ratio of X:", torch.sum(paths==0)/paths.numel(), "ratio of Y:", torch.sum(paths==1)/paths.numel())
        return paths
    
class ScalarProjectionBatchShare():
    def __init__(self, low=0.5, high=1.5, generator=None):
        self.w = torch.empty(size=()).uniform_(low, high, generator=generator)
    def predict(self, x):
        return x * self.w # [num_x_tokens, n_dim]

class LinearRegressionBatchShare():
    def __init__(self, mean=0, std=1, dim=3, generator=None):
        self.w = torch.normal(mean, std, (dim, 1), generator=generator)
        self.dim=dim
    def predict(self, x):
        y = x @ self.w # [bsz, n_dim, 1]
        y = torch.cat((y, torch.zeros_like(y).repeat(1,self.dim-1)), dim=-1)
        return y

class LinearProjectionBatchShare():
    def __init__(self, mean=0, std=1, dim=3, generator=None):
        self.w = torch.normal(mean, std, (dim, dim), generator=generator)
    def predict(self, x):
        return x @ self.w


