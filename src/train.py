import os
from random import randint
import uuid
import random
from quinine import QuinineArgumentParser
from tqdm import tqdm
import torch
import yaml

from eval import get_run_metrics
from tasks import get_task_sampler
from samplers import get_data_sampler
from curriculum import Curriculum
from schema import schema
from models import build_model

import wandb

torch.backends.cudnn.benchmark = True


def train_step(model, xs, ys, optimizer, loss_func):
    optimizer.zero_grad()
    #print("xs", xs)
    #print("ys", ys)
    output = model(xs, ys)
    #print("pred-ys", (output.cpu()-ys.cpu()).mean(-1))
    loss = loss_func(output, ys)
    loss.backward()
    optimizer.step()
    return loss.detach().item(), output.detach()


def sample_seeds(total_seeds, count):
    seeds = set()
    while len(seeds) < count:
        seeds.add(randint(0, total_seeds - 1))
    return seeds


def train(model, args):
    
    random.seed(42)
    
    device=args.training.device
    if args.model.family in ['gpt2']:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.training.learning_rate)
    elif args.model.family in ['llama']: # for llama, only tune the input and output linear embedding
        optimizer = torch.optim.Adam([
        {"params": model._read_in.parameters()},
        {"params": model._read_out.parameters()}
    ], lr=args.training.learning_rate)
    curriculum = Curriculum(args.training.curriculum)

    starting_step = 0
    state_path = os.path.join(args.out_dir, "state.pt")
    if os.path.exists(state_path):
        state = torch.load(state_path)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        starting_step = state["train_step"]
        for i in range(state["train_step"] + 1):
            curriculum.update()

    n_dims = model.n_dims
    bsize = args.training.batch_size
    if args.training.data == 'multi_data':
        data_sampler = get_data_sampler(args.training.multi_datas, n_dims=n_dims)
    else:
        if args.training.data in ['autoregression']:
            data_sampler_kwargs_ = args.training.data_sampler_kwargs
            data_sampler_kwargs = {}
            data_sampler_kwargs['x_y_order'] = data_sampler_kwargs_.x_y_order
            data_sampler_kwargs['add_task_tokens'] = data_sampler_kwargs_.add_task_tokens
            data_sampler_kwargs['n_embedding'] = data_sampler_kwargs_.n_embedding
            data_sampler_kwargs['finite_start_tokens'] = data_sampler_kwargs_.finite_start_tokens
            data_sampler_kwargs['n_start_tokens'] = data_sampler_kwargs_.n_start_tokens
            data_sampler_kwargs['task_list'] = data_sampler_kwargs_.task_list
            data_sampler_kwargs['min_out_degree'] = data_sampler_kwargs_.min_out_degree
            data_sampler_kwargs['max_out_degree'] = data_sampler_kwargs_.max_out_degree
            data_sampler_kwargs['total_len'] = curriculum.n_points
            data_sampler_kwargs['x_add_ratio'] = data_sampler_kwargs_.x_add_ratio
            data_sampler_kwargs['y_add_ratio'] = data_sampler_kwargs_.y_add_ratio
            
            data_sampler = get_data_sampler(args.training.data, n_dims=n_dims, **data_sampler_kwargs)
        else:
            data_sampler = get_data_sampler(args.training.data, n_dims=n_dims)
    if len(args.training.multi_tasks)>0:
        task_sampler = get_task_sampler(
        args.training.multi_tasks,
        n_dims,
        bsize,
        num_tasks=args.training.num_tasks,
        **args.training.task_kwargs,
    )
    else:
        task_sampler = get_task_sampler(
            args.training.task,
            n_dims,
            bsize,
            num_tasks=args.training.num_tasks,
            **args.training.task_kwargs,
        )
    pbar = tqdm(range(starting_step, args.training.train_steps))

    num_training_examples = args.training.num_training_examples

    for i in pbar:
        data_sampler_args = {}
        task_sampler_args = {}
        if "sparse" in args.training.task:
            task_sampler_args["valid_coords"] = curriculum.n_dims_truncated
        if num_training_examples is not None:
            assert num_training_examples >= bsize
            if args.training.data in ['autoregression']:
                batch_seed = random.randint(0, num_training_examples // bsize + 1)
                data_sampler_args["seeds"] = batch_seed # int
            else:
                seeds = sample_seeds(num_training_examples, bsize)
                data_sampler_args["seeds"] = seeds
            task_sampler_args["seeds"] = [s + 1 for s in seeds]
        else:
            if args.training.data in ['autoregression']:
                batch_seed = i
                data_sampler_args["seeds"] = batch_seed # int

        if type(data_sampler) != list: # single type of training distribution
            if args.training.data in ['autoregression']:
                xs = data_sampler.sample_xs(
                        curriculum.n_points,
                        bsize,
                        curriculum.n_dims_truncated,
                        **data_sampler_args,
                    )
                ys = torch.empty(1)
            elif args.training.data in ['random_int']:
                xs_int_index, xs_embeded = data_sampler.sample_xs(
                        curriculum.n_points,
                        bsize,
                        curriculum.n_dims_truncated,
                        **data_sampler_args,
                    )
            else:
                if 'non-ICL' in args.training.task: #
                    xs = data_sampler.sample_xs(
                        n_points=1,
                        b_size= bsize,
                        n_dims_truncated= curriculum.n_dims_truncated,
                        **data_sampler_args,
                    )
                else:
                    xs = data_sampler.sample_xs(
                        curriculum.n_points,
                        bsize,
                        curriculum.n_dims_truncated,
                        **data_sampler_args,
                    )
            
        if type(task_sampler) == list: # multiple training tasks
                
            wandb_dict = {
                    "n_points": curriculum.n_points,
                    "n_dims": curriculum.n_dims_truncated,
                }
            pbar_str = ""
            for t in range(len(task_sampler)):
                if type(data_sampler) == list: # each task has different training data distribution, so we need to resample here
                    data_sampler_ = data_sampler[t]
                    data_name = args.training.multi_datas[t]
                    if data_name in ['random_int']:
                        xs_int_index, xs_embeded = data_sampler_.sample_xs(
                                curriculum.n_points,
                                bsize,
                                curriculum.n_dims_truncated,
                                **data_sampler_args,
                            )
                    else:
                        if 'non-ICL' in args.training.task: #
                            xs = data_sampler_.sample_xs(
                                n_points=1,
                                b_size= bsize,
                                n_dims_truncated= curriculum.n_dims_truncated,
                                **data_sampler_args,
                            )
                        else:
                            xs = data_sampler_.sample_xs(
                                curriculum.n_points,
                                bsize,
                                curriculum.n_dims_truncated,
                                **data_sampler_args,
                            )  
                else: # single training data
                    data_name = args.training.data 
                task_sampler_ = task_sampler[t]
                #print("task_sampler_", task_sampler_, "data_name", data_name)
                #print(task_sampler_args)
                
                task = task_sampler_(**task_sampler_args)
                if data_name in ['random_int']:
                    ys = task.evaluate(xs_int_index)
                else:
                    ys = task.evaluate(xs)
                #print(task.w_b)
                #print("xs=",xs)
                #print("ys=",ys)
                loss_func = task.get_training_metric()
                if data_name in ['random_int']:
                    loss, output = train_step(model, xs_embeded.to(device), ys.to(device), optimizer, loss_func)
                else:
                    loss, output = train_step(model, xs.to(device), ys.to(device), optimizer, loss_func)
                cur_task_name = args.training.multi_tasks[t]
                wandb_dict[cur_task_name+"_loss"]=loss
                pbar_str += cur_task_name+f"_loss:{loss} "
            if i % args.wandb.log_every_steps == 0 and not args.test_run:
                wandb.log(
                    wandb_dict,
                    step=i,
                )
            pbar.set_description(pbar_str)
        else:
            task = task_sampler(**task_sampler_args)
            if args.training.data in ['random_int']:
                ys = task.evaluate(xs_int_index)
                #print("x_int_index", xs_int_index)
            else:
                ys = task.evaluate(xs)
            loss_func = task.get_training_metric()

            if args.training.data in ['random_int']:
                loss, output = train_step(model, xs_embeded.to(device), ys.to(device), optimizer, loss_func)
            else:
                loss, output = train_step(model, xs.to(device), ys.to(device), optimizer, loss_func)
            #print(task.random_mappings)
            point_wise_tags = list(range(curriculum.n_points))
            point_wise_loss_func = task.get_metric()
            point_wise_loss = point_wise_loss_func(output, ys.to(device)).mean(dim=0)

            '''baseline_loss = (
                sum(
                    max(curriculum.n_dims_truncated - ii, 0)
                    for ii in range(curriculum.n_points)
                )
                / curriculum.n_points
            )'''
            
            if i % args.wandb.log_every_steps == 0 and not args.test_run:
                    wandb.log(
                        {
                            "overall_loss": loss,
                            #"excess_loss": loss / baseline_loss,
                            "pointwise/loss": dict(
                                zip(point_wise_tags, point_wise_loss.cpu().numpy())
                            ),
                            "n_points": curriculum.n_points,
                            "n_dims": curriculum.n_dims_truncated,
                        },
                        step=i,
                    )
            pbar.set_description(f"loss {loss}")

        curriculum.update()

        
        if i % args.training.save_every_steps == 0 and not args.test_run:
            training_state = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_step": i,
            }
            torch.save(training_state, state_path)

        if (
            args.training.keep_every_steps > 0
            and i % args.training.keep_every_steps == 0
            and not args.test_run
            and i > 0
        ):
            torch.save(model.state_dict(), os.path.join(args.out_dir, f"model_{i}.pt"))


def main(args):
    device=args.training.device
    if args.test_run:
        curriculum_args = args.training.curriculum
        curriculum_args.points.start = curriculum_args.points.end
        curriculum_args.dims.start = curriculum_args.dims.end
        args.training.train_steps = 100
    else:
        wandb.init(
            dir=args.out_dir,
            project=args.wandb.project,
            entity=args.wandb.entity,
            config=args.__dict__,
            notes=args.wandb.notes,
            name=args.wandb.name,
            resume=True,
        )

    model = build_model(args.model)
    model.to(device)
    model.train()

    train(model, args)

    if not args.test_run:
        _ = get_run_metrics(args.out_dir)  # precompute metrics for eval


if __name__ == "__main__":
    parser = QuinineArgumentParser(schema=schema)
    args = parser.parse_quinfig()
    assert args.model.family in ["gpt2", "lstm"]
    print(f"Running with: {args}")

    if not args.test_run:
        run_id = args.training.resume_id
        if run_id is None:
            run_id = str(uuid.uuid4())

        out_dir = os.path.join(args.out_dir, run_id)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        args.out_dir = out_dir

        with open(os.path.join(out_dir, "config.yaml"), "w") as yaml_file:
            yaml.dump(args.__dict__, yaml_file, default_flow_style=False)
        #print(args.training.data_sampler_kwargs)
    main(args)
