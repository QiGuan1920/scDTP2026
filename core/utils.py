import torch
import numpy as np
import random
import warnings

warnings.filterwarnings('ignore')

def set_all_seeds(seed=1000):
    """Set all random seeds to ensure reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"✓ Random seeds set to: {seed}")

def get_device():
    """Get the available device (CPU/GPU)."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    return device

def test_reproducibility_cpu(seed=1000):
    """Test reproducibility in the current environment."""
    set_all_seeds(seed)
    rand1, perm1, numpy_rand1 = torch.randn(5), torch.randperm(10), np.random.random(3)
    
    set_all_seeds(seed)
    rand2, perm2, numpy_rand2 = torch.randn(5), torch.randperm(10), np.random.random(3)
    
    success = torch.equal(rand1, rand2) and torch.equal(perm1, perm2) and np.array_equal(numpy_rand1, numpy_rand2)
    print("✓ Randomness control test passed" if success else "✗ Randomness control test failed")
    return success