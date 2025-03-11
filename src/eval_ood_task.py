from eval import get_ood_task_run_metrics
import argparse
import torch
parser=argparse.ArgumentParser()
parser.add_argument('--pretrain_path', type=str, default='/data3/qxwang/checkpoints/icl-garg-checkpoints/relu_multilayernn_regression_4l/49eadf29-aa7b-48a0-870a-26bc95bca49c')
parser.add_argument('--ood_task', type=str, default='linear_regression')
parser.add_argument('--device', type=str, default='cuda:1')
parser.add_argument('--n_context_test', type=int, default=101)
parser.add_argument('--collect_prediction', action='store_true', help='collect the Xs and predictions when evaluation')
parser.add_argument('--fixed_embedding_path', type=str, default='', help='only applied to gaussian data evaluation, sample Xs a fixed embedding')
parser.add_argument('--rank', type=int, default=2, help='Rank for regression tasks that keep out dim')
parser.add_argument('--step', type=int, default=-1, help='Model of how many training steps to load')
parser.add_argument('--data_type', type=str, help='Data type for test', default='gaussian')
args = parser.parse_args()
fixed_embedding = torch.load(args.fixed_embedding_path) if args.fixed_embedding_path != '' else None

if 'random_init' in args.pretrain_path:
    get_ood_task_run_metrics(args.pretrain_path, args.ood_task, args.device, args.n_context_test, random_init_model=True, 
                             collect_prediction=args.collect_prediction, fixed_embedding=fixed_embedding, rank=args.rank,
                             step=args.step, data_type=args.data_type)
else:
    get_ood_task_run_metrics(args.pretrain_path, args.ood_task, args.device, args.n_context_test, collect_prediction=args.collect_prediction,
                             fixed_embedding=fixed_embedding, rank=args.rank, step=args.step, data_type=args.data_type)