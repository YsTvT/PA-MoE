

import argparse
import os

import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="~/data/multiturn")
    parser.add_argument("--hdfs_dir", default=None)
    args = parser.parse_args()

    conversations = []

    conversations.append(
        {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is the capital of France?"},
                {"role": "assistant", "content": "The capital of France is Paris."},
                {"role": "user", "content": "And what about Germany?"},
                {"role": "assistant", "content": "The capital of Germany is Berlin."},
            ]
        }
    )

    conversations.append(
        {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Can you explain quantum computing?"},
                {
                    "role": "assistant",
                    "content": "Quantum computing is a type of computing that uses quantum-mechanical phenomena, such as superposition and entanglement, to perform operations on data.",
                },
                {"role": "user", "content": "How is it different from classical computing?"},
                {
                    "role": "assistant",
                    "content": "Classical computing uses bits that are either 0 or 1, while quantum computing uses quantum bits or qubits that can exist in multiple states simultaneously due to superposition.",
                },
            ]
        }
    )

    conversations.append(
        {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Write a simple Python function to calculate factorial."},
                {
                    "role": "assistant",
                    "content": "```python\ndef factorial(n):\n    if n == 0 or n == 1:\n        return 1\n    else:\n        return n * factorial(n-1)\n```\n\nThis is a recursive function to calculate the factorial of a number.",
                },
                {"role": "user", "content": "Can you make it iterative instead?"},
                {
                    "role": "assistant",
                    "content": "```python\ndef factorial(n):\n    result = 1\n    for i in range(1, n+1):\n        result *= i\n    return result\n```\n\nThis is an iterative version of the factorial function.",
                },
            ]
        }
    )

    train_data = conversations[:2]
    test_data = conversations[2:]

    local_dir = os.path.expanduser(args.local_dir)
    os.makedirs(local_dir, exist_ok=True)

    train_df = pd.DataFrame(train_data)
    test_df = pd.DataFrame(test_data)

    train_df.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_df.to_parquet(os.path.join(local_dir, "test.parquet"))

    if args.hdfs_dir is not None:
        try:
            from verl.utils.hdfs_io import copy, makedirs

            makedirs(args.hdfs_dir)
            copy(src=local_dir, dst=args.hdfs_dir)
        except ImportError:
            print("Warning: HDFS support not available. Skipping HDFS copy.")

    print(f"Train dataset size: {len(train_df)}")
    print(f"Test dataset size: {len(test_df)}")
    print(f"Data saved to {local_dir}")

if __name__ == "__main__":
    main()
