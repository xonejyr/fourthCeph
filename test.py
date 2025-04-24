import torch
import torchdata
import dgl
import matplotlib
import tensorflow
from torchdata.datapipes.iter import IterDataPipe
print(f"PyTorch version: {torch.__version__}")
print(f"Torchdata version: {torchdata.__version__}")
print(f"DGL version: {dgl.__version__}")
print(f"Matplotlib version: {matplotlib.__version__}")
print(f"TensorFlow version: {tensorflow.__version__}")
print("All imports successful!")