import copy
import os
import sys

import torch
import torch.nn as nn
from config import get_arguments


sys.path.insert(0, "../..")
from dataloader import get_dataloader
from networks.models import Generator, NetC_MNIST
from utils import progress_bar


def create_targets_bd(targets, opt):
    if opt.attack_mode == "all2one":
        bd_targets = torch.ones_like(targets) * opt.target_label
    elif opt.attack_mode == "all2all":
        bd_targets = torch.tensor([(label + 1) % opt.num_classes for label in targets])
    else:
        raise Exception("{} attack mode is not implemented".format(opt.attack_mode))
    return bd_targets.to(opt.device)


def create_bd(netG, netM, inputs, targets, opt):
    bd_targets = create_targets_bd(targets, opt)
    patterns = netG(inputs)
    patterns = netG.normalize_pattern(patterns)

    masks_output = netM.threshold(netM(inputs))
    bd_inputs = inputs + (patterns - inputs) * masks_output
    return bd_inputs, bd_targets


def eval(netC, netG, netM, test_dl, opt):
    print(" Eval:")
    acc_clean = 0.0
    acc_bd = 0.0
    total_sample = 0
    total_correct_clean = 0
    total_correct_bd = 0

    for batch_idx, (inputs, targets) in enumerate(test_dl):
        inputs, targets = inputs.to(opt.device), targets.to(opt.device)
        bs = inputs.shape[0]
        total_sample += bs

        # Evaluating clean
        preds_clean = netC(inputs)
        correct_clean = torch.sum(torch.argmax(preds_clean, 1) == targets)
        total_correct_clean += correct_clean
        acc_clean = total_correct_clean * 100.0 / total_sample

        # Evaluating backdoor
        inputs_bd, targets_bd = create_bd(netG, netM, inputs, targets, opt)
        preds_bd = netC(inputs_bd)
        correct_bd = torch.sum(torch.argmax(preds_bd, 1) == targets_bd)
        total_correct_bd += correct_bd
        acc_bd = total_correct_bd * 100.0 / total_sample

        progress_bar(batch_idx, len(test_dl), "Acc Clean: {:.3f} | Acc Bd: {:.3f}".format(acc_clean, acc_bd))
    return acc_clean, acc_bd


def main():
    # Prepare arguments
    opt = get_arguments().parse_args()
    if opt.dataset == "mnist":
        opt.num_classes = 10
    else:
        raise Exception("Invalid Dataset")
    if opt.dataset == "mnist":
        opt.input_height = 28
        opt.input_width = 28
        opt.input_channel = 1
    else:
        raise Exception("Invalid Dataset")

    # Load models
    if opt.dataset == "mnist":
        netC = NetC_MNIST().to(opt.device)
    else:
        raise Exception("Invalid dataset")

    path_model = os.path.join(
        opt.checkpoints, opt.dataset, opt.attack_mode, "{}_{}_ckpt.pth.tar".format(opt.attack_mode, opt.dataset)
    )
    state_dict = torch.load(path_model)
    netC.load_state_dict(state_dict["netC"])
    netC.to(opt.device)
    netC.eval()
    netC.requires_grad_(False)
    netG = Generator(opt)
    netG.load_state_dict(state_dict["netG"])
    netG.to(opt.device)
    netG.eval()
    netG.requires_grad_(False)

    netM = Generator(opt, out_channels=1)
    netM.load_state_dict(state_dict["netM"])
    netM.to(opt.device)
    netM.eval()
    netM.requires_grad_(False)

    # Prepare dataloader
    test_dl = get_dataloader(opt, train=False)

    # Forward hook for getting layer's output
    container = []

    def forward_hook(module, input, output):
        container.append(output)

    hook = netC.relu6.register_forward_hook(forward_hook)

    # Forwarding all the validation set
    print("Forwarding all the validation dataset:")
    for batch_idx, (inputs, _) in enumerate(test_dl):
        inputs = inputs.to(opt.device)
        netC(inputs)
        progress_bar(batch_idx, len(test_dl))

    # Processing to get the "more important mask"
    container = torch.cat(container, dim=0)
    activation = torch.mean(container, dim=[0, 2, 3])
    seq_sort = torch.argsort(activation)
    pruning_mask = torch.ones(seq_sort.shape[0], dtype=bool)
    hook.remove()

    # Pruning times - no-tuning after pruning a channel!!!
    acc_clean = []
    acc_bd = []
    with open(opt.outfile, "w") as outs:
        for index in range(pruning_mask.shape[0]):
            net_pruned = copy.deepcopy(netC)
            num_pruned = index
            if index:
                channel = seq_sort[index]
                pruning_mask[channel] = False
            print("Pruned {} filters".format(num_pruned))

            net_pruned.conv5 = nn.Conv2d(64, 64 - num_pruned, (5, 5), 1, 0)
            net_pruned.linear6 = nn.Linear(16 * (64 - num_pruned), 512)

            # Re-assigning weight to the pruned net
            for name, module in net_pruned._modules.items():
                if "conv5" in name:
                    module.weight.data = netC.conv5.weight.data[pruning_mask]
                    module.bias.data = netC.conv5.bias.data[pruning_mask]
                elif "linear6" in name:
                    module.weight.data = netC.linear6.weight.data.reshape(-1, 64, 16)[:, pruning_mask].reshape(512, -1)
                    module.bias.data = netC.linear6.bias.data
                else:
                    continue
            clean, bd = eval(net_pruned, netG, netM, test_dl, opt)
            outs.write("%d %0.4f %0.4f\n" % (index, clean, bd))


if __name__ == "__main__":
    main()
