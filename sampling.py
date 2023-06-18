import torch
import numpy as np
import itertools



def get_probability(config):
    probability = np.ones(config.num_classes) / config.num_classes
    factor = 0.05
    if config.num_classes == 100:
        factor = 0.01
    if config.entropy_decrease != 0:
        probability = (
                probability
                - (1 / config.num_classes) * np.arange(config.num_classes) * factor
        )
        probability /= probability.sum()
        probability = (
                probability ** config.entropy_decrease
                / (probability ** config.entropy_decrease).sum()
        )
        np.random.seed(config.seed)
        np.random.shuffle(probability)
    return probability

def get_classes(config, task_id, probability):

    if config.randomized_order != 1:
        # randomized order parameter assess how the structure of the stream impact results
        list_couples = list(itertools.combinations(np.arange(config.num_classes), config.classes_per_task))
        nb_couples = len(list_couples)
        index = task_id % nb_couples
        classes = np.array(list_couples[index])

        # change class value with probability config.randomized_order
        id_change_classes = \
        np.where(np.random.randint(0, 99, size=config.classes_per_task) < config.randomized_order * 100)[0]
        if len(id_change_classes) > 0:  # if somme classes need to be changed
            # create random news classes
            new_classes = np.random.randint(0, config.num_classes, size=config.classes_per_task)
            # change the classes selected
            classes[id_change_classes] = new_classes[id_change_classes]
    elif config.randomized_couples != 1:
        if task_id == 0:
            # we set the couples of possible classes
            # it is useful to measure how important it is that all classes meet all classes within a task.
            list_couples = np.array(
                list(itertools.combinations(np.arange(config.num_classes), config.classes_per_task)))

            # shuffle along first dimension (it does not change the unique combinations)
            np.random.shuffle(list_couples)

            # config.randomized_couples is a pourcentage
            nb_select = int(config.randomized_couples * len(list_couples))

            # set static variable
            get_classes.couple_selected = list_couples[:nb_select, :]

        idx = np.random.randint(0, len(get_classes.couple_selected))
        classes = get_classes.couple_selected[idx]

    else:
        if config.scenario == "incremental":
            # if incremental we create 5 long term data distribution
            num_period = 2
            period_size = config.num_tasks // num_period
            nb_classes_per_period = config.num_classes // num_period
            assert nb_classes_per_period * num_period == config.num_classes, print("we will not see all classes")
            assert nb_classes_per_period >= config.classes_per_task, print("we need more classes per periode")
            start_class = (task_id // period_size) * nb_classes_per_period
            end_class = (1 + (task_id // period_size)) * nb_classes_per_period
            list_classes = np.arange(config.num_classes)[start_class:end_class]
            local_probability = probability[start_class:end_class]
            local_probability /= local_probability.sum()  # normalization
            classes = np.random.choice(
                list_classes, p=local_probability, size=config.classes_per_task, replace=False
            )
       
        else:
            if config.forgetting and task_id > config.num_tasks / 2:
                assert config.entropy_decrease == 0, print(
                    "there is no experiments combining entropy decrease and forgetting"
                )
                # we remove half the classes to assess if there is forgetting
                print("forgetting will start")
                classes = list(torch.randperm(config.num_classes // 2)[:config.classes_per_task].long())
            else:
                if config.class_sampling == 'uniform':
                    classes = np.random.choice(np.arange(config.num_classes), p=probability,
                                               size=config.classes_per_task, replace=False)
                elif config.class_sampling == 'iid':
                    classes = np.arange(config.num_classes)
                else:
                    raise NotImplementedError

    return classes
