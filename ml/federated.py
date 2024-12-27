import random
from typing import Tuple
import copy
import numpy as np
import torch
import torch.nn.functional as F
from config import federated_args, str2bool
from ml.utils.train_utils import train, test

# Load arguments
args = federated_args()
print(f"Script arguments: {args}\n")

# Enable Cuda if available
if torch.cuda.is_available():
    device = args.device
else:
    device = 'cpu'

# ensure reproducibility
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Load dataset
from dataset.load_dataset import load_dataset()
train_data_combined, test_data_combined = load_dataset()

# Print Dataset Details
print("== Predict Energy ==")
in_dim = 1
num_classes = len(torch.unique(torch.as_tensor(trainset.targets)))
print(f'Input Dimensions: {in_dim}')
print(f'Num of Classes: {num_classes}')
print(f'Train Samples: {len(trainset)}')
print(f'Test Samples: {len(testset)}')
print(f'Num of Clients: {args.clients}')
print(f'Train samples per client: {int((len(trainset)/args.clients)*(1-args.test_size))}')
print(f'Test samples per client: {int((len(trainset)/args.clients)*(args.test_size))}')
print("===============")

