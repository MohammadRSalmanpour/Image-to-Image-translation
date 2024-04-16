import argparse


class Parser:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.subparsers = self.parser.add_subparsers()

        self.train = self.subparsers.add_parser("train")
        self.test = self.subparsers.add_parser("test")

        self.init_train_subparser()
        self.init_test_subparser()

    def init_train_subparser(self):
        self.train.add_argument("--epochs", type=int, default=200)
        self.train.add_argument("--batch_size", type=int, default=16)
        self.train.add_argument("--dataset_size", type=int, default=10000)

        self.train.add_argument(
            "--augment",
            choices=["random_flip_left_right", "random_brightness", "random_saturation"],
            nargs="*",
        )

        self.train.add_argument("--checkpoints_dir", default="checkpoints")
        self.train.add_argument(
            "--checkpoints_freq", type=int, default=10,
            help="Create a checkpoint after the specified number of epochs"
        )
        self.train.add_argument(
            "--domain_A_files", required=True,
            help="A dataset of all files matching one or more glob patterns"
        )
        self.train.add_argument(
            "--domain_B_files", required=True,
            help="A dataset of all files matching one or more glob patterns"
        )
        self.train.add_argument(
            "--samples_dir", default="samples",
            help="A directory where generated samples are stored"
        )
        self.train.add_argument(
            "--samples_freq", type=int, default=10,
            help="Generate a few samples after the specified number of epochs"
        )

        self.train.add_argument(
            "--parallel", action="store_true", help="Enable distributed training on multiple GPUs"
        )

    def init_test_subparser(self):
        self.test.add_argument("--checkpoints_dir", default="checkpoints")
        self.test.add_argument(
            "--domain_A_files", required=True,
            help="A directory containing images in the JPG or PNG format"
        )
        self.test.add_argument(
            "--results_dir", default="results",
            help="A directory where translated images are stored"
        )

    def parse_args(self):
        return self.parser.parse_args()
