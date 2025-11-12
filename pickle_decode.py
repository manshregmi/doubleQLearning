import pickle
import numpy as np
import sys
import types

# Compatibility fix for old NumPy pickles
core_module = types.ModuleType("numpy._core")
multiarray_module = types.ModuleType("numpy._core.multiarray")
core_module.multiarray = multiarray_module

def dummy_scalar(*args, **kwargs):
    return np.array(args[0] if args else 0)

multiarray_module.scalar = dummy_scalar

sys.modules["numpy._core"] = core_module
sys.modules["numpy._core.multiarray"] = multiarray_module

# Load the pickle file
file_path = "q_tables.pkl"
with open(file_path, "rb") as f:
    data = pickle.load(f)

print(type(data))
if hasattr(data, "__len__"):
    print("Length:", len(data))

# Optionally inspect the first few entries
if isinstance(data, dict):
    for k, v in list(data.items())[:3]:
        print("Key:", k)
        print("Value type:", type(v))
        break