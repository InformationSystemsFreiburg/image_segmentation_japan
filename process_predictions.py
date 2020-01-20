import pickle
import torch
import numpy

i = 1
with open(f"./data/predictions_{i}.pkl", "rb") as f:

    prediction = pickle.load(f)
    print(prediction)