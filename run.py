"""Implementation of prototypical networks for Omniglot."""
import os
import sys

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import argparse
from torch.utils import tensorboard
import omniglot

NUM_TEST_TASKS = 600

from methods.protonet import ProtoNet
from methods.maml import MAML
from methods.proto_maml import ProtoMAML


def main(args):
    log_dir = args.log_dir
    if log_dir is None:
        if args.method == 'protonet':
            log_dir = f'./logs/{args.method}/omniglot.way:{args.num_way}.support:{args.num_support}.query:{args.num_query}.lr:{args.learning_rate}.batch_size:{args.batch_size}'  # pylint: disable=line-too-long
        else:  # maml or protomaml
            log_dir = f'./logs/{args.method}/omniglot.way:{args.num_way}.support:{args.num_support}.query:{args.num_query}.inner_steps:{args.num_inner_steps}.inner_lr:{args.inner_lr}.learn_inner_lrs:{args.learn_inner_lrs}.outer_lr:{args.outer_lr}.batch_size:{args.batch_size}'  # pylint: disable=line-too-long
    print(f'log_dir: {log_dir}')
    writer = tensorboard.SummaryWriter(log_dir=log_dir)

    if args.method == 'protonet':
        net = ProtoNet(args.learning_rate, log_dir)
    elif args.method == 'maml':
        net = MAML(
            args.num_way,
            args.num_inner_steps,
            args.inner_lr,
            args.learn_inner_lrs,
            args.outer_lr,
            log_dir
        )
    elif args.method == 'protomaml':
        net = ProtoMAML(
            args.num_way,
            args.num_inner_steps,
            args.inner_lr,
            args.learn_inner_lrs,
            args.outer_lr,
            args.output_lr,
            log_dir
        )
    else:
        raise ValueError

    if args.checkpoint_step > -1:
        net.load(args.checkpoint_step)
    else:
        print('Checkpoint loading skipped.')

    if not args.test:
        num_training_tasks = args.batch_size * (args.num_train_iterations -
                                                args.checkpoint_step - 1)
        print(
            f'Training on tasks with composition '
            f'num_way={args.num_way}, '
            f'num_support={args.num_support}, '
            f'num_query={args.num_query}'
        )
        dataloader_train = omniglot.get_omniglot_dataloader(
            'train',
            args.batch_size,
            args.num_way,
            args.num_support,
            args.num_query,
            num_training_tasks
        )
        dataloader_val = omniglot.get_omniglot_dataloader(
            'val',
            args.batch_size,
            args.num_way,
            args.num_support,
            args.num_query,
            args.batch_size * 4
        )
        net.train(
            dataloader_train,
            dataloader_val,
            writer
        )
    else:
        print(
            f'Testing on tasks with composition '
            f'num_way={args.num_way}, '
            f'num_support={args.num_support}, '
            f'num_query={args.num_query}'
        )
        dataloader_test = omniglot.get_omniglot_dataloader(
            'test',
            1,
            args.num_way,
            args.num_support,
            args.num_query,
            NUM_TEST_TASKS
        )
        net.test(dataloader_test)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Train a ProtoNet!')
    parser.add_argument('--method', type=str, default=None,
                        help='method to use [protonet, maml, protomaml]')
    parser.add_argument('--log_dir', type=str, default=None,
                        help='directory to save to or load from')
    parser.add_argument('--num_way', type=int, default=5,
                        help='number of classes in a task')
    parser.add_argument('--num_support', type=int, default=1,
                        help='number of support examples per class in a task')
    parser.add_argument('--num_query', type=int, default=15,
                        help='number of query examples per class in a task')
    parser.add_argument('--num_inner_steps', type=int, default=1,
                        help='number of inner-loop updates (for MAML-related only)')
    parser.add_argument('--inner_lr', type=float, default=0.4,
                        help='inner-loop learning rate initialization (for MAML-related only)')
    parser.add_argument('--learn_inner_lrs', default=False, action='store_true',
                        help='whether to optimize inner-loop learning rates (for MAML-related only)')
    parser.add_argument('--outer_lr', type=float, default=0.001,
                        help='outer-loop learning rate (for MAML-related only)')
    parser.add_argument('--output_lr', type=float, default=0.4,
                        help='inner-loop learning rate initialization for output layer (for Proto-MAML only)')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='learning rate for the network (for ProtoNet-related only)')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='number of tasks per outer-loop update')
    parser.add_argument('--num_train_iterations', type=int, default=15001,
                        help='number of outer-loop updates to train for')
    parser.add_argument('--test', default=False, action='store_true',
                        help='train or test')
    parser.add_argument('--checkpoint_step', type=int, default=-1,
                        help=('checkpoint iteration to load for resuming '
                              'training, or for evaluation (-1 is ignored)'))

    main_args = parser.parse_args()
    main(main_args)
