
import abc
import heapq
import numpy as np
from continuum.tasks import TaskType, TaskSet
from copy import copy
import random

# Idea
# Class that produces a taskset with a mixture between replay and current data
# the memory of the class can be updated with new data (and eventually predictioon).

# advantage: frequency filtering, data difficulty / plasticity compensation.


class RandomReplay(abc.ABC):
    """" this class manage its memory and replay"""
    """"It is supposed to replay data from classes randomly selected"""


    def __init__(self, config):

        #self.size_per_class = config.size_per_class
        self.replay_budget  = config.replay_budget
        self.classes_per_task  = config.classes_per_task
        self.class_seen_sofar = []
        self.dict_occ = {}
        self.nb_batches = 0
        self.sample_per_classes = 200
        self.memory_set = None

    def add_data(self, task_set):

        classes = task_set.get_classes()

        for class_ in classes:
            if class_ in self.dict_occ.keys():
                # we re-new min 5% of samples and more at the beginning
                nb_samples = max(int(self.sample_per_classes / (self.dict_occ[class_] + 1)),
                                 int(self.sample_per_classes / 20))
            else:
                nb_samples = self.sample_per_classes

            y_indexes = np.where(task_set._y==class_)[0]
            np.random.shuffle(y_indexes)
            subset_y_indexes = y_indexes[:nb_samples]

            x, y, _ =  task_set.get_samples(subset_y_indexes)

            if self.memory_set is None:
                self.memory_set = TaskSet(copy(x), copy(y), None, None, data_type=TaskType.TENSOR)
            else:
                y_memory_indexes = np.where(self.memory_set._y == class_)[0]
                if len(y_memory_indexes) == 0:
                    self.memory_set.add_samples(copy(x), copy(y), None)
                else:
                    # we replace a subset of samples
                    np.random.shuffle(y_memory_indexes)
                    subset_memory_y_indexes = y_memory_indexes[:nb_samples]
                    self.memory_set._x[subset_memory_y_indexes] = copy(x)
                    self.memory_set._y[subset_memory_y_indexes] = copy(y)



    def update_stats(self, classes):
        # we assume distribution over y to be balance.
        self.class_seen_sofar = list(np.unique(self.class_seen_sofar + list(classes)))

        for class_ in classes:
            if class_ in self.dict_occ.keys():
                self.dict_occ[class_] += 1
            else:
                self.dict_occ[class_] = 1

    def classes_2_replay(self, current_classes):

        if len(self.class_seen_sofar) == 0:
            return []

        nb_classes_replay = len(np.where(np.random.uniform(low=0.0, high=1.0, size=self.classes_per_task) < self.replay_budget)[0])

        seen_so_far = copy(self.class_seen_sofar)
        random.shuffle(seen_so_far)


        list_classes = []
        while(len(list_classes) != nb_classes_replay):
            class_selected = seen_so_far.pop()
            if class_selected not in current_classes:
                list_classes.append(class_selected)

            if len(seen_so_far) == 0:
                break

        return list_classes


class FrequencyReplay(abc.ABC):
    """" this class manage its memory and replay"""
    """"It is supposed to replay data only in its frequency range"""


    def __init__(self, config):

        #self.size_per_class = config.size_per_class
        self.low_frequency  = config.low_frequency
        self.high_frequency  = config.high_frequency
        self.occurrence_trigger = 3 # number of time a concept needs to be seen to be taken into account.
        self.dict_stats = {}
        self.dict_occ = {}
        self.dict_first_instance = {}
        self.nb_batches = 0
        self.sample_per_classes = 200
        self.memory_set = None

    def add_data(self, task_set):

        classes = task_set.get_classes()

        for class_ in classes:
            if class_ in self.dict_occ.keys():
                # we re-new min 5% of samples and more at the beginning
                nb_samples = max(int(self.sample_per_classes / (self.dict_occ[class_] + 1)),
                                 int(self.sample_per_classes / 20))
            else:
                nb_samples = self.sample_per_classes

            y_indexes = np.where(task_set._y==class_)[0]
            np.random.shuffle(y_indexes)
            subset_y_indexes = y_indexes[:nb_samples]

            x, y, _ =  task_set.get_samples(subset_y_indexes)

            if self.memory_set is None:
                self.memory_set = TaskSet(copy(x), copy(y), None, None, data_type=TaskType.TENSOR)
            else:
                y_memory_indexes = np.where(self.memory_set._y == class_)[0]
                if len(y_memory_indexes) == 0:
                    self.memory_set.add_samples(copy(x), copy(y), None)
                else:
                    # we replace a subset of samples
                    np.random.shuffle(y_memory_indexes)
                    subset_memory_y_indexes = y_memory_indexes[:nb_samples]
                    self.memory_set._x[subset_memory_y_indexes] = copy(x)
                    self.memory_set._y[subset_memory_y_indexes] = copy(y)



    def update_stats(self, classes, acc_per_classes=None):
        # we assume distribution over y to be balance.

        self.nb_batches += 1
        if acc_per_classes is None:
            increments = list(np.ones(len(classes)))
        else:
            increments = list(acc_per_classes)

        for class_, inc in zip(classes, increments):
            assert inc <= 1.0
            if class_ in self.dict_stats.keys():
                self.dict_stats[class_] += inc
                self.dict_occ[class_] += 1
            else:
                self.dict_first_instance[class_] = self.nb_batches
                self.dict_stats[class_] = inc
                self.dict_occ[class_] = 1



    def classes_2_replay(self, current_classes):

        if len(self.dict_stats.keys()) == 0:
            return []

        # select the least frequent classes to replay

        k = len(current_classes)

        # top k least frequent values
        list_value = list(zip(self.dict_stats.values(), self.dict_stats.keys()))
        heapq.heapify(list_value)

        list_classes = []
        while(len(list_classes) != k):
            if len(list_value) == 0:
                break
            class_candidate = heapq.heappop(list_value)[1]
            if class_candidate in current_classes:
                continue
            if self.dict_occ[class_candidate] < self.occurrence_trigger:
                # if a classes has never been seen more than the occurence trigger, it is not taken into account
                # it avoids replaying too rare classes and focus on frequency > low frequency
                continue

            frequency = self.dict_stats[class_candidate] / self.nb_batches
            if frequency > self.high_frequency:
                #the class is too frequent to be replayed (and next ones will be even more)
                break
            else:
                if frequency > self.low_frequency:
                    list_classes.append(class_candidate)
        return list_classes