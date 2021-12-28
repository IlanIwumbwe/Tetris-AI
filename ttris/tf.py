import math
import torch

s = {'2':12, '4': 5}


print(max(s, key=s.get))
