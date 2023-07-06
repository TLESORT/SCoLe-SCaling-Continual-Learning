import argparse
import os
import types
import sys
import copy
import torch
import wandb
import torchvision
import numpy as np
from continuum.scenarios import ClassIncremental, ContinualScenario
from sampling import get_probability, get_classes
from Utils import *
from utils_training import run_taskset
from Models.encoders import encoders
from continuum.tasks.utils import split_train_val
from replay import FrequencyReplay, RandomReplay
import timeit

from global_settings import * # sets the device globally
from torchvision import transforms


def run_scenario(config):
    if config.rand_transform == "perturbations":
        # import is conditionned because it needs additional dependencies
        from perturbations.utils_perturbations import get_perturbation
        from perturbations.test_perturbations import PerturbationTransform

    dataset_train, dataset_test, nb_classes, input_d, transformations, transformations_te = \
        get_dataset(config, config.dataset, config.data_dir, config.architecture)
    config.input_d = input_d


    if config.dataset == "CIFAR100Lifelong":
        scenario = ContinualScenario(dataset_train, transformations=transformations)
        scenario_te = ContinualScenario(dataset_test, transformations=transformations_te)
    else:
        scenario = ClassIncremental(dataset_train, nb_tasks=config.num_classes, transformations=transformations)
        scenario_te = ClassIncremental(dataset_test, nb_tasks=config.num_classes,
                                        transformations=transformations_te)

    model = get_model(config, device=device)

    run = wandb.init(
        dir=config.root_dir,
        project=config.project,
        settings=wandb.Settings(start_method="fork"),
        group="Scole",
        id=wandb.util.generate_id(),
        tags=[config.optim],
        config=config,
    )
    print(f"wandb run {run.name}")
    full_tr_dataset = scenario[:]
    full_te_dataset = scenario_te[:]

    ReplayBuffer = None
    if config.replay in ["frequency", "freq_acc"]:
        ReplayBuffer = FrequencyReplay(config)
    elif config.replay == "random":
        ReplayBuffer = RandomReplay(config)

    opt = get_optim(model, name=config.optim, lr=config.lr, momentum=config.momentum)

    probability = get_probability(config)
    if config.setup == "incremental":

        if config.seed == 0:
            class_vec = np.arange(config.num_classes)
        else:
            class_vec = np.random.permutation(config.num_classes)
        class_vec = class_vec[:config.classes_per_task * config.num_tasks]

        task_collection = list(class_vec.reshape([config.num_tasks, -1]))

    id_epoch = 0
    for task_id in range(config.num_tasks):

        starttime = timeit.default_timer()

        if config.reinit_opt == "Yes":
            del opt
            opt = get_optim(model, name=config.optim, lr=config.lr, momentum=config.momentum)

        nb_epochs = config.nb_epochs

        # sample with seed for reproducibility
        # the scenario is composed of 5 binary classification classes randomly ordered
        if config.setup == "online":
            classes = get_classes(config, task_id, probability)
            if config.prob_reduction != 0:
                probability[classes] /= config.prob_reduction
                probability = probability / probability.sum()
        else:
            classes = task_collection[task_id]

        classes_replay = []
        if config.replay in ["frequency", "freq_acc", "random"]:
            classes_replay = ReplayBuffer.classes_2_replay(classes)
            print(f"Replay: {classes_replay}")

        if config.dataset == "CIFAR100Lifelong":
            if config.num_tasks == 1:
                taskset_tr = full_tr_dataset
                env_id = task_id % 5
                taskset_tr = deepcopy(scenario[env_id])
                indexes = np.where(np.isin(taskset_tr._y, classes))[0]
                taskset_tr._x = taskset_tr._x[indexes]
                taskset_tr._y = taskset_tr._y[indexes]
                taskset_tr._t = taskset_tr._t[indexes]
        else:
            taskset_tr = scenario[classes]

        assert len(taskset_tr) > 0
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

        print(f"train: {classes}")

        lr_scheduler = None
        saved_model = None
        if opt is not None and config.lr_aneal:
            # reset optimizer for wach task
            opt = get_optim(model, name=config.optim, lr=config.lr, momentum=config.momentum)
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=config.nb_epochs)

        best_epoch = 0
        model_before = None

        val_acc_ES = 0
        cpt = 0
        for epoch in range(nb_epochs):
            id_epoch += 1
            print("epoch", epoch)
            eval_set = taskset_val
            if config.eval_on == "test":
                eval_set = full_te_dataset

            train_acc, train_acc_per_class, _ = run_taskset(config, taskset_tr, model, opt=opt,
                                                            replay_buffer=ReplayBuffer,
                                                            replay_classes=classes_replay)

            if lr_scheduler is not None:
                lr_scheduler.step()

            # log each epoch
            if config.num_tasks == 1:
                test_acc, test_acc_per_class, _ = run_taskset(config, full_te_dataset, model, opt=None)
                if config.class_acc:
                    wandb.log({"train_acc": train_acc, "test_acc": test_acc, "epoch": epoch,
                               "train_acc_per_class": {str(i): acc for i, acc in enumerate(train_acc_per_class)},
                               "test_acc_per_class": {str(i): acc for i, acc in enumerate(test_acc_per_class)}})
                else:
                    wandb.log({"train_acc": train_acc, "test_acc": test_acc, "epoch": epoch})

            print("test (full test set)")
            test_acc, test_acc_per_class, _ = run_taskset(config, full_te_dataset, model, opt=None)

            val_acc, val_acc_per_class, val_loss = run_taskset(config, eval_set, model, opt=None)

            dict_epoch_result = {"id_epoch": id_epoch, "local_epoch": epoch, "local_task_id": task_id,
                                 "val_acc": val_acc,
                                 "val_loss": val_loss}

            wandb.log(dict_epoch_result)

            if val_acc > val_acc_ES:
                cpt = 0
                best_epoch = epoch
                val_acc_ES = val_acc
                if config.early_stopping != 0:
                    # save parameter dictionary
                    saved_model = model.state_dict()
            else:
                cpt += 1
                if (config.early_stopping != 0) and (saved_model is not None):
                    if cpt > config.early_stopping and (epoch >= cpt):
                        model.load_state_dict(saved_model)
                        saved_model = None
                        opt = get_optim(model, name=config.optim, lr=config.lr, momentum=config.momentum)
                        break

        if config.replay in ["frequency", "random"]:
            all_classes = list(classes) + classes_replay
            ReplayBuffer.add_data(taskset_tr)
            ReplayBuffer.update_stats(all_classes)
        elif config.replay == "freq_acc":
            ReplayBuffer.add_data(taskset_tr)
            all_classes = list(classes) + classes_replay
            ReplayBuffer.update_stats(all_classes, val_acc_per_class[all_classes])

        dict_task_results = {"train_acc": train_acc, "test_acc": test_acc, "task_index": task_id,
                             "classes": list(classes) + classes_replay, "epoch": epoch, "best_epoch": best_epoch}

        if config.class_acc:
            dict_task_results = {**dict_task_results,
                                 "test_acc_per_class": {str(i): acc for i, acc in enumerate(test_acc_per_class)}}

        wandb.log(dict_task_results)

        print("\n Time to run a task is:", timeit.default_timer() - starttime, "\n")

    if config.replay in ["frequency", "freq_acc"]:
        dict_frequency = ReplayBuffer.dict_stats
        wandb.log({"classes_freq": np.array(list(dict_frequency.keys())),
                   "freq": np.array(list(dict_frequency.values())) / ReplayBuffer.nb_batches,
                   "first_instance": np.array(list(ReplayBuffer.dict_first_instance.values()))})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, default="./Archives")
    parser.add_argument("--data_dir", type=str, default="../Datasets")
    parser.add_argument("--project", type=str, default="Scole")
    parser.add_argument("--wandb_api_key", type=str, default=None)

    parser.add_argument("--dataset", type=str, default="MNIST",
                        choices=["MNIST", "CIFAR10", "CIFAR100", "CUB200", "KMNIST", "fashion", "Car196", "Aircraft",
                                 "CIFAR100Lifelong", "Tiny"])
    parser.add_argument("--num_tasks", type=int, default=5, help="Task number")
    parser.add_argument("--prob_reduction", type=int, default=0,
                        help="reduce probability of visiting a class already visited")
    parser.add_argument("--num_classes", type=int, default=10, help="Task class in the full scenario") 
    parser.add_argument("--model", type=str, default="baseResnet", choices=["alexnet", "resnet", "googlenet", "vgg"])
    parser.add_argument("--classes_per_task", type=int, default=2, help="number of classes wanted in each task")
    parser.add_argument("--nb_epochs", type=int, default=1, help="nb epoch to train")
    parser.add_argument("--nb_epoch_val", type=int, default=1, help="nb epoch to train val probe")
    parser.add_argument("--eval_on", type=str, default="val", choices=["val", "test"])
    parser.add_argument("--nb_layers", type=int, default=20, help="nb layers in resnet", choices=[20, 32, 44, 56])
    parser.add_argument("--optim", default="Adam", type=str,
                        choices=["Adadelta", "Adagrad", "AdamW", "SparseAdam", "Adamax", "ASGD", "LBFGS", "NAdam",
                                 "RMSprop", "Rprop", "SGD", "Adam"])
    parser.add_argument("--setup", default="online", type=str,
                        choices=["online", "preset", "incremental", "incremental_fc"])
    parser.add_argument('--replay', default="None", type=str,
                        choices=["None", "default", "frequency", "freq_acc", "random"])
    parser.add_argument('--replay_budget', default=1.0, type=float)
    parser.add_argument("--low_frequency", default=0.01, type=float)
    parser.add_argument("--high_frequency", default=0.1, type=float)
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--momentum", default=0.0, type=float)
    parser.add_argument("--entropy_decrease", default=0, type=float)          
    parser.add_argument("--rand_transform", default="None", type=str, choices=["None", "perturbations"])
    parser.add_argument("--seed", default="1664", type=int)
    parser.add_argument("--forgetting", type=bool, default=False, help="flag to assess if forgetting still happens")
    parser.add_argument("--masking", default="None", type=str, choices=["None", "group"])
    parser.add_argument("--head", default="linear", type=str, choices=["linear", "weightnorm"])
    parser.add_argument("--scenario", default="default", type=str,
                        choices=["default", "incremental", "classical_cl_repeated",
                                 "classical_cl_repeated_permuted"])
    parser.add_argument("--batch_size", default=64, type=int, help="Batch size")
    parser.add_argument("--early_stopping", default=0, type=int, help="early stopping criterion, if 0 no early stopping"
                                                                      " else, it design the number of epochs without"
                                                                      " progress that trigger the end of the task")
    parser.add_argument("--rand_permut", type=int, default=0,
                        help="if 1 perturbs randomly images of tasks before task 1 repeats (works together with classical_cl_repeated), if 2, perturbs all tasks after the first one ")

    parser.add_argument("--reinit_opt", default="No", type=str, help="Reinitialize optimizer for each task")
    parser.add_argument("--class_acc", type=bool, default=False, help="log accuracy for each class separately")
    parser.add_argument('--debug', type=int, default=0)

    parser.add_argument('--checkpoint_dir', type=str, default="/mnt/home/Projects/convergence/checkpoints")

    parser.add_argument('--use_predefined_hps', type=int, default=0,
                        help="if 1 uses predefined hyperparameters found with preliminary hp search")
    parser.add_argument('--severity', type=int, default=0, choices=[-1, 0, 1, 2, 3, 4])
    parser.add_argument('--class_sampling', type=str,
                        choices=['uniform', 'uniform_with_cycles',
                                 'uniform_shifted', 'cl_with_cycles', "iid"], default='uniform')
    parser.add_argument('--cycle_size', type=int, default=100)  # every cycle_size all classes should have been seen
    parser.add_argument('--class_sampling_std', type=float, default=10)
    parser.add_argument('--lr_aneal', type=int, default=0)
    parser.add_argument('--reinit_model', type=int, default=0)
    parser.add_argument('--pretrained_model', type=str, choices=list(encoders.keys()), default=None)
    parser.add_argument('--wrn_width_factor', type=int, default=1)
    parser.add_argument('--wrn_dropout', type=float, default=0.)
    parser.add_argument('--architecture', type=str,
                        choices=['default', 'default2', 'resnet', 'vgg', 'vit_b_16', 'inception'], default='default')
    parser.add_argument("--randomized_order", default="1", type=float,
                        help="start from a fixed sequence of tasks then randomly change some classes.")
    parser.add_argument("--randomized_couples", default="1", type=float,
                        help="define the amount of meet couples among all possible couples.")
    parser.add_argument('--wandb_offline', type=int, default=0)

    config = parser.parse_args()

    if config.wandb_offline:
        print(config.root_dir)
        os.environ["WANDB_MODE"] = "offline"
    
    if config.wandb_api_key is not None:
        os.environ["WANDB_API_KEY"] = config.wandb_api_key

    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    if config.use_predefined_hps:
        if config.masking == "None":
            if config.optim == "Adam":
                config.lr = 0.0001
                config.momentum = 0.0
            elif config.optim == "SGD":
                config.lr = 0.01
                config.momentum = 0.0
        elif config.masking == "group":
            if config.optim == "Adam":
                config.lr = 0.001
                config.momentum = 0.0
            elif config.optim == "SGD":
                config.lr = 0.01
                config.momentum = 0.0
        else:
            raise NotImplementedError

    if config.num_tasks > 1 and config.optim == "Adam" and config.momentum == 0.9:
        print("adam is not controlled by momentum so this experiments does not make sens.")
        sys.exit()

    if config.early_stopping != 0:
        config.nb_epochs = max(config.nb_epochs, 200)

    run_scenario(config)
