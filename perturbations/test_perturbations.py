import warnings
from typing import Callable, List, Union

import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from torchvision.transforms.functional import pil_to_tensor, to_pil_image

from continuum.datasets import _ContinuumDataset, CIFAR10, TinyImageNet200
from continuum.scenarios import TransformationIncremental

from perturbations.utils_perturbations import gaussian_noise, shot_noise, impulse_noise, speckle_noise, gaussian_blur, glass_blur, \
    defocus_blur, motion_blur, zoom_blur, fog, frost, snow, spatter, contrast, brightness, saturate, elastic_transform


class Perturbations(TransformationIncremental):
    """Continual Loader, generating datasets for the consecutive tasks.
    Scenario: Permutations scenarios, use same data for all task but with pixels permuted.
    Each task get a specific permutation, such as all tasks are different.
    :param cl_dataset: A continual dataset.
    :param nb_tasks: The scenario's number of tasks.
    :param base_transformations: List of transformations to apply to all tasks.
    :param seed: initialization seed for the permutations.
    :param shared_label_space: If true same data with different transformation have same label
    """

    def __init__(
            self,
            cl_dataset: _ContinuumDataset,
            list_perturbation=[],
            list_severity=[1],
            base_transformations: List[Callable] = None,
            shared_label_space=True
    ):
        if len(list_severity) > 1:
            assert len(list_severity) == len(list_perturbation)
        else:
            list_severity = list_severity * len(list_perturbation)
        trsfs = self.create_function_list(list_perturbation, list_severity)

        super().__init__(
            cl_dataset=cl_dataset,
            incremental_transformations=trsfs,
            base_transformations=base_transformations,
            shared_label_space=shared_label_space
        )

    def create_function_list(self, list_perturbations, list_severity):

        list_pert = []
        for distortion, severity in zip(list_perturbations, list_severity):
            if distortion == "None":
                trsf = PerturbationTransform(None)
                list_pert.append(trsf)
            elif distortion == "gaussian_noise":
                trsf = PerturbationTransform(gaussian_noise, severity)
                list_pert.append(trsf)
            elif distortion == "shot_noise":
                trsf = PerturbationTransform(shot_noise, severity)
                list_pert.append(trsf)
            elif distortion == "impulse_noise":
                trsf = PerturbationTransform(impulse_noise, severity)
                list_pert.append(trsf)
            elif distortion == "speckle_noise":
                trsf = PerturbationTransform(speckle_noise, severity)
                list_pert.append(trsf)
            elif distortion == "gaussian_blur":
                trsf = PerturbationTransform(gaussian_blur, severity)
                list_pert.append(trsf)
            elif distortion == "glass_blur":
                trsf = PerturbationTransform(glass_blur, severity)
                list_pert.append(trsf)
            elif distortion == "defocus_blur":
                trsf = PerturbationTransform(defocus_blur, severity)
                list_pert.append(trsf)
            elif distortion == "motion_blur":
                trsf = PerturbationTransform(motion_blur, severity)
                list_pert.append(trsf)
            elif distortion == "zoom_blur":
                trsf = PerturbationTransform(zoom_blur, severity)
                list_pert.append(trsf)
            elif distortion == "fog":
                trsf = PerturbationTransform(fog, severity)
                list_pert.append(trsf)
            elif distortion == "frost":
                trsf = PerturbationTransform(frost, severity)
                list_pert.append(trsf)
            elif distortion == "snow":
                trsf = PerturbationTransform(snow, severity)
                list_pert.append(trsf)
            elif distortion == "spatter":
                trsf = PerturbationTransform(spatter, severity)
                list_pert.append(trsf)
            elif distortion == "contrast":
                trsf = PerturbationTransform(contrast, severity)
                list_pert.append(trsf)
            elif distortion == "brightness":
                trsf = PerturbationTransform(brightness, severity)
                list_pert.append(trsf)
            elif distortion == "saturate":
                trsf = PerturbationTransform(saturate, severity)
                list_pert.append(trsf)
            elif distortion == "elastic_transform":
                trsf = PerturbationTransform(elastic_transform, severity)
                list_pert.append(trsf)
            else:
                raise NotImplementedError("this distortion {distortion} does not exists.")

        return list_pert

    def get_task_transformation(self, task_index):
        return transforms.Compose(self.trsf.transforms + [self.inc_trsf[task_index]])


class PerturbationTransform:
    """Permutation transformers.
    This transformer is initialized with a seed such as same seed = same permutation.
    Seed 0 means no permutations
    :param seed: seed to initialize the random number generator
    """

    def __init__(self, perturbation, severity=1):
        self.pertubation_function = perturbation
        self.severity = severity
        self.image_size = 64

    def __call__(self, x):
        if self.pertubation_function == None:
            return x

        #mean = x.mean(axis=[1,2])
        #std = x.std(axis=[1,2])

        x = x.view(self.image_size,self.image_size,3)
        #np_x = to_pil_image(x)
        np_x = x.cpu().numpy()
        np_x -= np_x.min()
        np_x /= np_x.max()
        np_x *= 255
        np_x = np_x.astype(np.uint8)
        #pert_img = np_x
        pert_img = Image.fromarray(np_x)
        pert_img = self.pertubation_function(pert_img, severity=self.severity)
        if isinstance(pert_img, np.ndarray):
            pert_img = torch.from_numpy(pert_img).float()
        else:
            pert_img = pil_to_tensor(pert_img).float()

        pert_img -= np_x.min()
        pert_img /= np_x.max()
        pert_img = pert_img.reshape(3, self.image_size, self.image_size)

        # for dim_ in [0,1,2]:
        #     pert_img[dim_] -= (pert_img[dim_].mean()  - mean[dim_])
        #     pert_img[dim_] /= (pert_img[dim_].std() / std[dim_])

        return pert_img

if __name__ == '__main__':
    list_distortion = ["None", "gaussian_noise", "shot_noise", "impulse_noise", "speckle_noise", "gaussian_blur",
                       "defocus_blur", "motion_blur", "zoom_blur", "fog", "snow", "spatter", "contrast",
                       "brightness", "saturate", "elastic_transform", "glass_blur", "frost", ]
    list_test = ["gaussian_noise", "shot_noise", "impulse_noise", "speckle_noise"]
    list_test = list_distortion
    #dataset = CIFAR10("../../Datasets", train=True)
    dataset = TinyImageNet200("../../Datasets", train=True)

    scenario = Perturbations(dataset, list_perturbation=list_test, list_severity=[5])

    for index, taskset in enumerate(scenario):
        taskset.plot(".", f"task_{index}.png", 100)
