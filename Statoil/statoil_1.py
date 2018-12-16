import numpy as np
import pandas as pd
import json

with open('./data/train.json') as f:
    data = json.load(f)

train = pd.read_json("data/train.json", orient="record")
print(train.head())