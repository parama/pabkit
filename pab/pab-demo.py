"""
Demo of Process-Aware Benchmarking (PAB) toolkit on CIFAR-10 using ResNet18
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse

# Import from pab toolkit
from pab_toolkit import PABTracker, track_training, pgd_attack

def parse_args():
    parser = argparse.ArgumentParser(description='Process-Aware Benchmarking (PAB) Demo')
    parser