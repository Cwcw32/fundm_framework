import torch
import os

t=torch.cuda.is_available()
f=torch.zeros(5)
os.environ['CUDA_VISIBLE_DEVICES'] = str(4)
f.cuda()
while(1):
    print(t)