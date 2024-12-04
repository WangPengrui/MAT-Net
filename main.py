import random
import torch
import numpy as np
import Trans_mod


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


seed = 57207880
setup_seed(seed=seed)
# Device Configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("\nSelected device:", device, end="\n\n")

tmod = Trans_mod.Train_test(dataset='sy30A', device=device, skip_train=False, save=True, seed=seed)
tmod.run(smry=False)
