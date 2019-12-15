# Running the configuration(s)

import torch
from torch.multiprocessing import cpu_count
import torch.multiprocessing as mp
import argparse
from src.Manager import Manager
import warnings

# The fork method has to be setted in the main method, before any CUDA call
if __name__ == "__main__":
    # Parse arguments
    try:
        mp.set_start_method('forkserver', force=True)
        print("forkserver init")
    except RuntimeError:
        pass

# Disabling the warnings (or messages from PyTorch concerning LSTM's memory blocks)
warnings.filterwarnings('ignore')

# Prompt messages
print(f"PyTorch version {torch.__version__}")
print(f"Number of CPU {cpu_count()}")
print(f"Cuda enabled device {torch.cuda.is_available()}")

parser = argparse.ArgumentParser()

parser.add_argument('-c', '--configs', nargs="+", help="Configurations to run", default=["default.cfg"])
parser.add_argument('-p', '--parallel', action="store_true", help="Run configurations in parallel", default=False)


if __name__ == "__main__":
    args = parser.parse_args()

    managers = [Manager(config) for config in args.configs]

    if args.parallel:
        [manager.start() for manager in managers]
        [manager.join() for manager in managers]
    
    else :
        for manager in managers:
            manager.start()
            manager.join()