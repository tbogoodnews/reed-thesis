import random 
import numpy as np
import pandas as pd
from pathlib import Path

avg_model = pd.DataFrame(columns = ["prediction_length", "average", "train_mse", "test_mse", "val_mse"])

get_addys = lambda file_name : [int(x.split()[0]) for x in open(file_name, "r").readlines()[2:]]

train = list(Path("../Data/baleen").rglob("*.trace"))
val = list(Path("../Data/block_Nexus/Trace_files").rglob("*"))

train = train[:round(len(train ) * 0.8)]
test = train[round(len(train ) * 0.8):]
train =  [(get_addys(file)) for file in train]
test =  [(get_addys(file)) for file in test]
val =  [(get_addys(file)) for file in val]

def split_into (addys, length, n) : 
    splits = []
    for start in random.sample(range(1, len(addys) - length), n):
        splits.append(addys[start:start+length])
    return splits

mse = lambda addys, avg : np.mean((np.array(addys) - avg)**2)

hits = lambda addys : [len(a) - len(set(a)) for a in addys]

flatten = lambda xss : [x for xs in xss for x in xs] 

for length in [50, 100, 150, 250, 500, 1000]:
    print("Length:", length)
    train_hits = [hits(split_into(f, length, round(len(f) * 0.05))) for f in train]
    avg_hits = np.mean([np.mean(x) for x in train_hits])
    test_hits = [hits(split_into(f, length, round(len(f) * 0.05))) for f in test]
    valid_hits = [hits(split_into(f, length, round(len(f) * 0.3))) for f in val]

    train_hits = mse(flatten(train_hits), avg_hits)
    test_hits = mse(flatten(test_hits), avg_hits)
    valid_hits = mse(flatten(valid_hits), avg_hits)

    avg_model = pd.concat([avg_model, pd.DataFrame([[length,avg_hits, train_hits, test_hits, valid_hits]], columns=avg_model.columns)])


avg_model.to_csv("Results/avg_model.csv")
print(avg_model.to_latex(index=False, float_format="{:.3f}".format))
