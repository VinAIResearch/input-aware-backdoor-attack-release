import argparse


def get_argument():
    parser = argparse.ArgumentParser()

    # Directory option
    parser.add_argument("--checkpoints", type=str, default="../../checkpoints/")
    parser.add_argument("--data_root", type=str, default="../../data/")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--result", type=str, default="./results")
    parser.add_argument("--dataset", type=str, default="mnist")
    parser.add_argument("--attack_mode", type=str, default="all2one")
    parser.add_argument("--temps", type=str, default="./temps")

    # ---------------------------- For Neural Cleanse --------------------------
    # Model hyperparameters
    parser.add_argument("--batchsize", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-1)
    parser.add_argument("--input_height", type=int, default=None)
    parser.add_argument("--input_width", type=int, default=None)
    parser.add_argument("--input_channel", type=int, default=None)
    parser.add_argument("--init_cost", type=float, default=1e-3)
    parser.add_argument("--atk_succ_threshold", type=float, default=99.0)
    parser.add_argument("--early_stop", type=bool, default=True)
    parser.add_argument("--early_stop_threshold", type=float, default=99.0)
    parser.add_argument("--early_stop_patience", type=int, default=25)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--cost_multiplier", type=float, default=2)
    parser.add_argument("--epoch", type=int, default=50)
    parser.add_argument("--num_workers", type=int, default=8)

    parser.add_argument("--target_label", type=int)
    parser.add_argument("--total_label", type=int)
    parser.add_argument("--EPSILON", type=float, default=1e-7)

    parser.add_argument("--to_file", type=bool, default=True)
    parser.add_argument("--n_times_test", type=int, default=5)

    return parser
