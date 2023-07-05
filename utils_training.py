import numpy as np
import tqdm
import torch
import torch.nn as nn
import wandb
from copy import copy
from collections import defaultdict
from copy import deepcopy

from torch.utils.data import DataLoader
import torch.nn.functional as F
from continuum import TaskSet
from continuum.tasks import TaskType, get_balanced_sampler

from Models.model import freeze_features, unfreeze_model
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
from global_settings import *  # sets the device globally


def get_loss(out, labels, masking, num_classes, loss_func):
    if masking == "group":
        label_unique = labels.unique()
        full_mask = torch.zeros_like(out)
        full_mask[:, label_unique] = 1
        out.masked_fill_(full_mask == 0, -1e9)

    if len(out) > 0:
        loss = loss_func(out, labels.long())
    else:
        loss = 0
    assert loss == loss, print("There should be some Nan")
    return loss


def cat_tasksets(taskset, memory, replay_classes):
    if replay_classes == []:
        return taskset

    x, y, _ = taskset.get_samples(np.arange(len(taskset)))
    new_taskset = TaskSet(copy(x), copy(y), None, None, data_type=TaskType.TENSOR)

    for class_ in replay_classes:
        indexes = np.where(memory.memory_set._y == class_)[0]
        new_taskset.add_samples(memory.memory_set._x[indexes], memory.memory_set._y[indexes], None)

    return new_taskset


def run_taskset(config, taskset, model, opt=None, balance=False, replay_buffer=None, replay_classes=[]):
    vector_pred = np.zeros(0)
    vector_label = np.zeros(0)
    if opt is None:
        model.eval()
    else:
        model.train()

    sampler = None
    if balance:
        sampler = get_balanced_sampler(taskset)
    tot_loss = 0.0

    if len(replay_classes) > 0:
        taskset = cat_tasksets(taskset, replay_buffer, replay_classes)

    loader = DataLoader(taskset, batch_size=config.batch_size, sampler=sampler, shuffle=opt is not None)

    num_classes = config.num_classes

    for x_, y_, t_ in loader:

        x_ = x_.to(device)
        output = model(x_)

        if output.dim() == 1:
            output = output.unsqueeze(0)
        predictions = np.array(output.max(dim=1)[1].cpu())
        vector_pred = np.concatenate([vector_pred, predictions])
        vector_label = np.concatenate([vector_label, y_.numpy()])

        loss = get_loss(output, y_.to(device), config.masking, num_classes, F.cross_entropy)

        if opt is not None and loss != 0:
            opt.zero_grad()
            loss.backward()
            # to avoid NaN
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            opt.step()

        if loss != 0:
            tot_loss += loss.detach().cpu().item()

    acc_per_class = np.zeros(config.num_classes)
    if opt is None:
        # we log accuracy per class only for test
        for _, label in enumerate(taskset.get_classes()):
            label_remaped = label
            indexes_class = np.where(vector_label == label_remaped)[0]
            classes_correctly_predicted = (vector_pred[indexes_class] == label_remaped).sum()
            acc_per_class[label] = (
                    1.0 * classes_correctly_predicted / (1.0 * len(indexes_class))
            )
    correct = (vector_pred == vector_label).sum()
    accuracy = (1.0 * correct) / len(vector_pred)

    if opt is None:
        print(f"Accuracy (val): {accuracy * 100:.2f} %")
    else:
        print(f"Accuracy (train): {accuracy * 100:.2f} %")
    return accuracy, acc_per_class, tot_loss


def reset_all_weights(model: nn.Module) -> None:
    """
    refs:
        - https://discuss.pytorch.org/t/how-to-re-set-alll-parameters-in-a-network/20819/6
        - https://stackoverflow.com/questions/63627997/reset-parameters-of-a-neural-network-in-pytorch
        - https://pytorch.org/docs/stable/generated/torch.nn.Module.html
    """

    @torch.no_grad()
    def weight_reset(m: nn.Module):
        # - check if the current module has reset_parameters & if it's callabed called it on m
        reset_parameters = getattr(m, "reset_parameters", None)
        if callable(reset_parameters):
            m.reset_parameters()

    # Applies fn recursively to every submodule see: https://pytorch.org/docs/stable/generated/torch.nn.Module.html
    model.apply(fn=weight_reset)