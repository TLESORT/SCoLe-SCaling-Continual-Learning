import torch
from torchvision import transforms
import numpy as np
import wandb
from copy import deepcopy
import torch.nn as nn
import pandas as pd
import torch.nn.functional as F
import torch.optim as optim
from utils_training import reset_all_weights
from continuum.tasks import TaskType, get_balanced_sampler
from continuum.datasets import CIFAR10, CIFAR100, KMNIST, MNIST, FashionMNIST, CUB200, Car196, FGVCAircraft, TinyImageNet200

from torch.utils.data import DataLoader 
from Models.model import Classifier, Model, get_CIFAR_Model, EncoderClassifier
from Models.encoders import encoders, PreparedModel, EncoderTuple

from continuum.tasks.utils import split_train_val
from continuum import TaskSet

from torchvision import transforms
from copy import deepcopy
from global_settings import * # sets the device globally



def get_dataset(config, dataset_name, data_dir, architecture="default"):
    transformations = None
    transformations_te = None
    if dataset_name == "MNIST":
        dataset_train = MNIST(data_dir, train=True)
        dataset_test = MNIST(data_dir, train=False)
        nb_classes = 10
        input_d = 28
    elif dataset_name == "fashion":
        dataset_train = FashionMNIST(data_dir, train=True)
        dataset_test = FashionMNIST(data_dir, train=False)
        nb_classes = 10
        input_d = 28
    elif dataset_name == "KMNIST":
        dataset_train = KMNIST(data_dir, train=True)
        dataset_test = KMNIST(data_dir, train=False)
        nb_classes = 10
        input_d = 28
    elif dataset_name == "CUB200":
        dataset_train = CUB200(data_dir, train=True)
        dataset_test = CUB200(data_dir, train=False)
        nb_classes = 200
        input_d = 100
        horizontal_flip = 0.5
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        size = [3, 100, 100]
        # use continuum transfroms
        transformations = [
            transforms.Resize([size[-1], size[-1]]),
            transforms.RandomHorizontalFlip(horizontal_flip) if horizontal_flip is not None else None,
            transforms.ToTensor(),
            transforms.Normalize(mean, std)]

        transformations_te = [
            transforms.Resize([size[-1], size[-1]]),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)]
    elif dataset_name == "Car196":
        dataset_train = Car196(data_dir, train=True)
        dataset_test = Car196(data_dir, train=False)
        nb_classes = 196
        input_d = 100
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        size = [3, 100, 100]
        # use continuum transfroms
        transformations = [
            transforms.Resize([size[-1], size[-1]]),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)]
        # transforms.Normalize(mean, std)]

        transformations_te = [
            transforms.Resize([size[-1], size[-1]]),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)]

    elif dataset_name == "Aircraft":
        dataset_train = FGVCAircraft(data_dir, train=True)
        dataset_test = FGVCAircraft(data_dir, train=False)
        nb_classes = 100
        input_d = 100
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        size = [3, 100, 100]
        # use continuum transfroms
        transformations = [
            transforms.Resize([size[-1], size[-1]]),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)]
        # transforms.Normalize(mean, std)]

        transformations_te = [
            transforms.Resize([size[-1], size[-1]]),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)]
    elif dataset_name == "Tiny":
        dataset_train = TinyImageNet200(data_dir, train=True)
        dataset_test = TinyImageNet200(data_dir, train=False)
        nb_classes = 200
        input_d = 64
    elif dataset_name == "CIFAR100":
        dataset_train = CIFAR100(data_dir, train=True)
        dataset_test = CIFAR100(data_dir, train=False)
        nb_classes = 100
        input_d = 32
    elif dataset_name == "CIFAR100Lifelong":
        dataset_train = CIFAR100(data_dir, train=True, labels_type="category", task_labels="lifelong")
        dataset_test = CIFAR100(data_dir, train=False, labels_type="category", task_labels="lifelong")
        nb_classes = 20
        input_d = 32
    else:
        dataset_train = CIFAR10(data_dir, train=True)
        dataset_test = CIFAR10(data_dir, train=False)
        nb_classes = 10
        input_d = 32

    if architecture not in ["default", "default2"]:
        size = 224
        if architecture == "inception":
            size = 299
        transformations = [transforms.Resize((size, size)),transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]
        transformations_te = transformations

    if config.num_tasks == 1:
        # we are in a iid training
        print(
            f"We are in a IID training config.num_classes = {nb_classes} and  config.classes_per_task = {nb_classes}"
        )
        config.num_classes = nb_classes
        config.classes_per_task = nb_classes

    # if we do not use all classes we select a random subset of classes
    if nb_classes != config.num_classes:
        class_set = np.random.choice(np.arange(nb_classes), size=config.num_classes, replace=False)

        dataset_train = dataset_train.slice(keep_classes=class_set)
        dataset_test = dataset_test.slice(keep_classes=class_set)

        unique_vals, new_y = np.unique(dataset_train.data[1], return_inverse=True)
        dataset_train.data = (dataset_train.data[0], new_y, dataset_train.data[2])

        unique_vals, new_y = np.unique(dataset_test.data[1], return_inverse=True)
        dataset_test.data = (dataset_test.data[0], new_y, dataset_test.data[2])

    return dataset_train, dataset_test, nb_classes, input_d, transformations, transformations_te


# Adadelta, Adagrad, AdamW, SparseAdam, Adamax, ASGD, LBFGS, NAdam, RMSprop, Rprop, SGD, Adam
def get_optim(model, name, lr, momentum):
    if name == "SGD":
        opt = optim.SGD(params=model.parameters(), lr=lr, momentum=momentum)
    elif name == "Adadelta":
        opt = optim.Adadelta(params=model.parameters(), lr=lr)
    elif name == "Adagrad":
        opt = optim.Adagrad(params=model.parameters(), lr=lr)
    elif name == "AdamW":
        opt = optim.AdamW(params=model.parameters(), lr=lr)
    elif name == "SparseAdam":
        opt = optim.SparseAdam(params=model.parameters(), lr=lr)
    elif name == "Adamax":
        opt = optim.Adamax(params=model.parameters(), lr=lr)
    elif name == "ASGD":
        opt = optim.ASGD(params=model.parameters(), lr=lr)
    elif name == "LBFGS":
        opt = optim.LBFGS(params=model.parameters(), lr=lr)
    elif name == "NAdam":
        opt = optim.NAdam(params=model.parameters(), lr=lr)
    elif name == "RAdam":
        opt = optim.RAdam(params=model.parameters(), lr=lr)
    elif name == "RMSprop":
        opt = optim.RMSprop(params=model.parameters(), lr=lr)
    elif name == "Rprop":
        opt = optim.Rprop(params=model.parameters(), lr=lr)
    else:
        opt = optim.Adam(params=model.parameters(), lr=lr)
    return opt

def get_model(config, device=device):

    if config.pretrained_model is None:
        if config.dataset in ["MNIST", 'mnist_fellowship', 'fashion', 'KMNIST']:
            model = Model(head_name=config.head).to(device)
        else:
            model = get_CIFAR_Model(config, num_classes=config.num_classes, head_name=config.head, nb_layers=config.nb_layers, ).to(device)
    else:
        # This does not work with MNIST
        encoder_tuple: EncoderTuple = encoders[config.pretrained_model]
        encoder: PreparedModel = encoder_tuple.partial_encoder(device=device, input_shape=config.input_d,
                                                               fix_batchnorms_encoder=False,
                                                               width_factor=config.wrn_width_factor,
                                                               droprate=config.wrn_dropout)
        tr, tr_te = encoder.transformation, encoder.transformation_val
        if tr is not None:
            transformations = tr
            if tr_te is not None:
                transformations_te = tr_te
            else:
                transformations_te = tr
        classifier = encoder.classifier
        if classifier is None:
            classifier = Classifier(num_classes=config.num_classes, in_d=encoder.latent_dim, head_name=config.head).to(device)
        model = EncoderClassifier(encoder=encoder.encoder, classifier=classifier, latent_dim=encoder.latent_dim).to(device)
        if config.reinit_model:
            reset_all_weights(model)

    return model

def init_state_dict(config, taskset):
    dict_state = {}
    size = len(np.where(taskset._t >= 0)[0])

    # taskset sanity check (we want all no negative indexes to be in the start by order)
    assert np.all(np.arange(size) == taskset._t[:size])

    dict_state["scores"] = np.zeros(size)
    if config.selection == "forgetting":
        dict_state["last_pred"] = np.zeros(size)

    if config.integration:
        dict_state["nb_updates"] = np.zeros(size)
    return dict_state



def update_frequency(counters, classes):
    for class_ in classes:
        counters[class_] += 1
    return counters

def get_classes_2_replay(counters, treshold = 0.0001):
    frequencies = counters / counters.sum()
    classes_2_replay = np.where(frequencies < treshold)[0]
    return classes_2_replay

def add_data_replay(scenario, classes, replay_classes, replay_amount=100):
    """add replay data but ignore potential data augmentation"""

    taskset = scenario[classes]
    if len(replay_classes) > 0:
        for class_ in replay_classes:
            if not (class_ in taskset.get_classes()):
                indexes = np.random.randint(0, len(scenario[class_]), replay_amount)
                x, y, _ = scenario[class_].get_raw_samples(indexes)
                taskset.add_samples(x, y)

    return taskset

def create_buffer(task_set, nb_samples, transformations):
    x, y, _ = task_set.get_raw_samples()
    indexes = np.where(y == task_set.get_classes()[0])[0]
    select_indexes = np.random.choice(indexes, nb_samples, replace=False)
    buffer = TaskSet(x[select_indexes].copy(), y[select_indexes].copy(), None, transformations,
                               data_type=task_set.data_type)
    return buffer

def update_buffer(buffer, task_set, nb_samples):

    x, y, _ = task_set.get_raw_samples()
    for class_ in task_set.get_classes():
        if not (class_ in buffer.get_classes()):
            indexes = np.where(y == class_)[0]
            select_indexes = np.random.choice(indexes, nb_samples, replace=False)
            buffer.add_samples(x[select_indexes].copy(), y[select_indexes].copy())

    return buffer

def get_tasksets(config, task_id, classes, scenario, full_tr_dataset, counters, transformations, selection_buffer, replay_buffer):

    nb_replay_samples = 30

    if (task_id == 0) and config.rand_transform == "perturbations":
        # import is conditionned because it needs additional dependencies
        from perturbations.utils_perturbations import get_perturbation
        from perturbations.test_perturbations import PerturbationTransform

    if config.dataset == "CIFAR100Lifelong":
        if config.num_tasks == 1:
            taskset_tr = full_tr_dataset
        else:
            env_id = task_id % 5
            taskset_tr = deepcopy(scenario[env_id])
            assert len(classes) == 2
            indexes = np.where((taskset_tr._y == classes[0]) | (taskset_tr._y == classes[1]))[0]
            taskset_tr._x = taskset_tr._x[indexes]
            taskset_tr._y = taskset_tr._y[indexes]
            taskset_tr._t = taskset_tr._t[indexes]
    else:
        taskset_tr = scenario[classes]
        taskset_tr, taskset_val = split_train_val(taskset_tr, val_split=0.1)

    if config.rand_transform == "perturbations":
        perturbation = get_perturbation()
        if perturbation is not None:
            severity = config.severity
            if severity == -1:
                # random choice among 0,1,2,3,4
                severity = np.random.choice([0, 1, 2, 3, 4])
            trsf = PerturbationTransform(perturbation, severity=severity)
            taskset_tr.trsf = transforms.Compose(taskset_tr.trsf.transforms + [trsf])

    if config.eval_on == "buffer":

        x, y, _ = taskset_val.get_raw_samples()
        if task_id == 0:
            indexes = np.where(y == classes[0])[0]
            select_indexes = np.random.choice(indexes, nb_replay_samples, replace=False)
            selection_buffer = TaskSet(x[select_indexes].copy(), y[select_indexes].copy(), None, transformations,
                                       data_type=scenario[classes[0]].data_type)
        for class_ in classes:
            if not (class_ in selection_buffer.get_classes()):
                indexes = np.where(y == class_)[0]
                select_indexes = np.random.choice(indexes, nb_replay_samples, replace=False)
                selection_buffer.add_samples(x[select_indexes].copy(), y[select_indexes].copy())


    if config.replay == "frequency_replay":
        assert config.rand_transform != "perturbations", print("This implementation does not take data augmentation into account")

        # 1 - countinuously building the buffer
        x, y, _ = taskset_tr.get_raw_samples()
        if task_id == 0:
            assert replay_buffer is None
            indexes = np.where(y == classes[0])[0]
            select_indexes = np.random.choice(indexes, nb_replay_samples, replace=False)
            replay_buffer = TaskSet(x[select_indexes].copy(), y[select_indexes].copy(), None, transformations,
                                       data_type=scenario[classes[0]].data_type)
            replay_buffer.counters = np.zeros(config.num_classes)
        for class_ in classes:
            if not (class_ in selection_buffer.get_classes()):
                indexes = np.where(y == class_)[0]
                select_indexes = np.random.choice(indexes, nb_replay_samples, replace=False)
                replay_buffer.add_samples(x[select_indexes].copy(), y[select_indexes].copy())

        # sampling the buffer
        replay_buffer.counters = update_frequency(replay_buffer.counters, classes)
        classes_2_replay = get_classes_2_replay(counters, treshold=config.treshold_replay)

        taskset_tr = add_data_replay(scenario, classes, classes_2_replay, replay_amount=100)

    return taskset_tr, taskset_val, selection_buffer, replay_buffer

def weight_reset(m):
    if hasattr(m, 'reset_parameters') and callable(getattr(m, 'reset_parameters')):
        m.reset_parameters()




