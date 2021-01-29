import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from config import get_argument
from dataloader import get_dataloader, get_dataset
from torchvision import transforms


sys.path.insert(0, "../..")
from classifier_models import PreActResNet18
from networks.models import Generator, NetC_MNIST
from utils import progress_bar


class Normalize:
    def __init__(self, opt, expected_values, variance):
        self.n_channels = opt.input_channel
        self.expected_values = expected_values
        self.variance = variance
        assert self.n_channels == len(self.expected_values)

    def __call__(self, x):
        x_clone = x.clone()
        for channel in range(self.n_channels):
            x_clone[:, :, channel] = (x[:, :, channel] - self.expected_values[channel]) / self.variance[channel]
        return x_clone


class Denormalize:
    def __init__(self, opt, expected_values, variance):
        self.n_channels = opt.input_channel
        self.expected_values = expected_values
        self.variance = variance
        assert self.n_channels == len(self.expected_values)

    def __call__(self, x):
        x_clone = x.clone()
        for channel in range(self.n_channels):
            x_clone[:, :, channel] = x[:, :, channel] * self.variance[channel] + self.expected_values[channel]
        return x_clone


class STRIP:
    def _superimpose(self, background, overlay):
        output = cv2.addWeighted(background, 1, overlay, 1, 0)
        if len(output.shape) == 2:
            output = np.expand_dims(output, 2)
        return output

    def _get_entropy(self, background, dataset, classifier):
        entropy_sum = [0] * self.n_sample
        x1_add = [0] * self.n_sample
        index_overlay = np.random.randint(0, len(dataset), size=self.n_sample)
        for index in range(self.n_sample):
            add_image = self._superimpose(background, dataset[index_overlay[index]][0])
            add_image = self.normalize(add_image)
            x1_add[index] = add_image

        py1_add = classifier(torch.stack(x1_add).to(self.device))
        py1_add = torch.sigmoid(py1_add).cpu().numpy()
        entropy_sum = -np.nansum(py1_add * np.log2(py1_add))
        return entropy_sum / self.n_sample

    def _get_denormalize(self, opt):
        if opt.dataset == "cifar10":
            denormalizer = Denormalize(opt, [0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
        elif opt.dataset == "mnist":
            denormalizer = Denormalize(opt, [0.5], [0.5])
        elif opt.dataset == "gtsrb":
            denormalizer = None
        else:
            raise Exception("Invalid dataset")
        return denormalizer

    def _get_normalize(self, opt):
        if opt.dataset == "cifar10":
            normalizer = Normalize(opt, [0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
        elif opt.dataset == "mnist":
            normalizer = Normalize(opt, [0.5], [0.5])
        elif opt.dataset == "gtsrb":
            normalizer = None
        else:
            raise Exception("Invalid dataset")
        if normalizer:
            transform = transforms.Compose([transforms.ToTensor(), normalizer])
        else:
            transform = transforms.ToTensor()
        return transform

    def __init__(self, opt):
        super().__init__()
        self.n_sample = opt.n_sample
        self.normalizer = self._get_normalize(opt)
        self.denormalizer = self._get_denormalize(opt)
        self.device = opt.device

    def normalize(self, x):
        if self.normalizer:
            x = self.normalizer(x)
        return x

    def denormalize(self, x):
        if self.denormalizer:
            x = self.denormalizer(x)
        return x

    def __call__(self, background, dataset, classifier):
        return self._get_entropy(background, dataset, classifier)


def strip(opt, mode="clean"):
    if opt.dataset == "mnist":
        opt.input_height = 28
        opt.input_width = 28
        opt.input_channel = 1
    elif opt.dataset == "cifar10":
        opt.input_height = 32
        opt.input_width = 32
        opt.input_channel = 3
    elif opt.dataset == "gtsrb":
        opt.input_height = 32
        opt.input_width = 32
        opt.input_channel = 3
    else:
        raise Exception("Invalid dataset")

    # Prepare pretrained classifier
    if opt.dataset == "mnist":
        netC = NetC_MNIST()
    elif opt.dataset == "cifar10":
        netC = PreActResNet18()
    else:
        netC = PreActResNet18(num_classes=43)
    for param in netC.parameters():
        param.requires_grad = False
    netC.to(opt.device)
    netC.eval()

    if mode != "clean":
        netG = Generator(opt)
        for param in netG.parameters():
            param.requires_grad = False
        netG.to(opt.device)
        netG.eval()

    # Load pretrained model
    ckpt_dir = os.path.join(opt.checkpoints, opt.dataset, opt.attack_mode)
    ckpt_path = os.path.join(ckpt_dir, "{}_{}_ckpt.pth.tar".format(opt.attack_mode, opt.dataset))
    state_dict = torch.load(ckpt_path)
    netC.load_state_dict(state_dict["netC"])
    if mode != "clean":
        netG.load_state_dict(state_dict["netG"])
        netM = Generator(opt, out_channels=1)
        netM.load_state_dict(state_dict["netM"])
        netM.to(opt.device)
        netM.eval()
        netM.requires_grad_(False)

    # Prepare test set
    testset = get_dataset(opt, train=False)
    opt.bs = opt.n_test
    test_dataloader = get_dataloader(opt, train=False)

    # STRIP detector
    strip_detector = STRIP(opt)

    # Entropy list
    list_entropy_trojan = []
    list_entropy_benign = []

    if mode == "attack":
        # Testing with perturbed data
        print("Testing with bd data !!!!")
        inputs, targets = next(iter(test_dataloader))
        inputs = inputs.to(opt.device)
        patterns = netG(inputs)
        patterns = netG.normalize_pattern(patterns)
        batch_masks = netM.threshold(netM(inputs))
        bd_inputs = inputs + (patterns - inputs) * batch_masks

        bd_inputs = netG.denormalize_pattern(bd_inputs) * 255.0
        bd_inputs = bd_inputs.detach().cpu().numpy()
        bd_inputs = np.clip(bd_inputs, 0, 255).astype(np.uint8).transpose((0, 2, 3, 1))
        for index in range(opt.n_test):
            background = bd_inputs[index]
            entropy = strip_detector(background, testset, netC)
            list_entropy_trojan.append(entropy)
            progress_bar(index, opt.n_test)

        # Testing with clean data
        for index in range(opt.n_test):
            background, _ = testset[index]
            entropy = strip_detector(background, testset, netC)
            list_entropy_benign.append(entropy)
    else:
        # Testing with clean data
        print("Testing with clean data !!!!")
        for index in range(opt.n_test):
            background, _ = testset[index]
            entropy = strip_detector(background, testset, netC)
            list_entropy_benign.append(entropy)
            progress_bar(index, opt.n_test)

    return list_entropy_trojan, list_entropy_benign


def main():
    opt = get_argument().parse_args()
    if "2" in opt.attack_mode:
        mode = "attack"
    else:
        mode = "clean"

    lists_entropy_trojan = []
    lists_entropy_benign = []
    for test_round in range(opt.test_rounds):
        list_entropy_trojan, list_entropy_benign = strip(opt, mode)
        lists_entropy_trojan += list_entropy_trojan
        lists_entropy_benign += list_entropy_benign

    # Save result to file
    result_dir = os.path.join(opt.results, opt.dataset)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    result_path = os.path.join(result_dir, opt.attack_mode)
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    result_path = os.path.join(result_path, "{}_{}_output.txt".format(opt.attack_mode, opt.dataset))

    with open(result_path, "a+") as f:
        for index in range(len(lists_entropy_trojan)):
            if index < len(lists_entropy_trojan) - 1:
                f.write("{} ".format(lists_entropy_trojan[index]))
            else:
                f.write("{}".format(lists_entropy_trojan[index]))
        f.write("\n")
        for index in range(len(lists_entropy_benign)):
            if index < len(lists_entropy_benign) - 1:
                f.write("{} ".format(lists_entropy_benign[index]))
            else:
                f.write("{}".format(lists_entropy_benign[index]))

    min_entropy = min(lists_entropy_trojan + lists_entropy_benign)
    # Determining
    print("Min entropy trojan: {}, Detection boundary: {}".format(min_entropy, opt.detection_boundary))
    if min_entropy < opt.detection_boundary:
        print("A backdoored model\n")
    else:
        print("Not a backdoor model\n")


if __name__ == "__main__":
    main()
