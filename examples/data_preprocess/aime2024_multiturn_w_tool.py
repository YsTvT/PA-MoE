

import argparse
import os

import datasets

from verl.utils.hdfs_io import copy, makedirs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="~/data/retool_aime2024")
    parser.add_argument("--hdfs_dir", default=None)

    args = parser.parse_args()

    data_path = "BytedTsinghua-SIA/AIME-2024"
    dataset = datasets.load_dataset(data_path, "default")

    train_dataset = dataset["train"]

    def make_map_fn(split):
        def process_fn(example, idx):
            orig_extra_info = example.pop("extra_info")
            extra_info = orig_extra_info.copy()
            extra_info["need_tools_kwargs"] = True
            extra_info["tools_kwargs"] = {
                "code_interpreter": {
                    "create_kwargs": {
                        "ground_truth": example["reward_model"]["ground_truth"],
                    },
                },
            }
            example["extra_info"] = extra_info
            return example

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)
