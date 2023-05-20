"""Implementation of Proto-MAML for Omniglot."""
import os
import sys

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import torch
import torch.nn.functional as F
from torch import autograd

import util
from methods import Backbone

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SAVE_INTERVAL = 100
LOG_INTERVAL = 10
VAL_INTERVAL = LOG_INTERVAL * 10
NUM_TEST_TASKS = 600

from methods.maml import MAML
from methods.protonet import ProtoNet


class ProtoMAML(MAML):
    """Trains and assesses a Proto-MAML."""

    def __init__(
            self,
            num_outputs,
            num_inner_steps,
            inner_lr,
            learn_inner_lrs,
            outer_lr,
            output_lr,
            log_dir
    ):
        MAML.__init__(
            self,
            num_outputs,
            num_inner_steps,
            inner_lr,
            learn_inner_lrs,
            outer_lr,
            log_dir
        )

        # Extend learning rate for output layer
        self._inner_lrs.update({
            'w_out': output_lr, 
            'b_out': output_lr
        })
    
    
    def _forward(self, images, parameters):
        """Computes predicted classification logits.

        Args:
            images (Tensor): batch of Omniglot images
                shape (num_images, channels, height, width)
            parameters (dict[str, Tensor]): parameters to use for
                the computation

        Returns:
            a Tensor consisting of a batch of logits
                shape (num_images, classes)
        """

        x = images
        x = Backbone.forward(x, parameters)
        x = F.linear(x, parameters['w_out'], parameters['b_out'])
        
        return x


    def _init_output_layer(self, prototypes):
        """Initialize output layer weights with prototype-based initialization.
        """
        init_weight = 2 * prototypes
        init_bias = -torch.norm(prototypes, dim=1)**2

        output_weight = init_weight.detach().requires_grad_()
        output_bias = init_bias.detach().requires_grad_()

        output_layer = {'w_out': output_weight, 'b_out': output_bias}

        return output_layer


    def _inner_loop(self, images, labels, train):
        # Determine prototype initialization
        features = Backbone.forward(images, self._meta_parameters)
        prototypes, _ = ProtoNet._compute_prototypes(features, labels)

        # Create output layer weights with prototype-based initialization
        output_layer = self._init_output_layer(prototypes)

        # Make a clone of the meta parameters
        local_parameters = {
            k: torch.clone(v)
            for k, v in self._meta_parameters.items()
        }
        # Append a clone of the output layer parameters
        local_parameters.update({
            k: torch.clone(v)
            for k, v in output_layer.items()
        })

        accuracies = []
        for _ in range(self._num_inner_steps):
            loss, acc = self._classify_feats(images, labels, local_parameters)
            accuracies.append(acc)

            # Calculate gradients and perform inner loop update
            grads = autograd.grad(loss, local_parameters.values())
            local_parameters = {
                k: v - self._inner_lrs[k] * g
                for k, v, g in zip(local_parameters.keys(), local_parameters.values(), grads)
            }

        # Compute accuracy on the support set after adaptation
        features = self._forward(images, local_parameters)
        preds = F.softmax(features, dim=1)
        acc = util.score(preds, labels)
        accuracies.append(acc)
        assert len(accuracies) == self._num_inner_steps + 1

        return local_parameters, accuracies
    