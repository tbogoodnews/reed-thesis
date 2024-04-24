from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
from ripser import Rips
from ripser import ripser
from pathlib import Path
import random


path_to_data = "../Data"
datasets = { "Block_Nexus" : {"path" : "block_Nexus/Trace_files", "file-type" : "txt-a"}, 
             "Baleen" : {"path" : "baleen", "file-type" : "txt-trace"}}

window_length = 150
window_overlap = 30
predict_next = 150

np.random.seed(0)


def cache_hits(data):
    return [len(l) - max(l) for l in data]


def reindex(l): 
	fs = list(set(l)) 
	indexed_addrs = [x for _, x in sorted(zip([l.index(a) for a in fs], fs))]
	return [indexed_addrs.index(a) for a in l]


def make_dataset(window_length = window_length, window_overlap = window_overlap, predict_next = predict_next, max_adress_sample = None):
    df = pd.DataFrame(columns=["dataset", "file", "adresses"])

    def slide_window(data):
        data = data["adresses"]
        outputs = []
        future_hits = []
        for start in range(0, len(data), window_length - window_overlap):
            end = start+window_length
            if end > len(data):
                continue
            outputs.append(reindex(data[start:end]))
            hits = predict_next - len(set(data[end:end+predict_next]))
            future_hits.append(hits)
        return np.array(outputs), future_hits

    for d in datasets:
        path = datasets[d]["path"]
        match d:
            case "Block_Nexus":
                file_paths = list(Path(path_to_data + "/" + path).rglob("*.txt"))
            case "Baleen":
                file_paths = list(Path(path_to_data + "/" + path).rglob("*.trace")) # REMOVE LATER
        for file in file_paths:
            full_path = str(file)
            f = open(full_path, "r") # open file
            match datasets[d]["file-type"]:
                case "txt-a":
                    adresses = [int(x.split("\t")[0]) for x in f.readlines()]
                case "txt-trace":
                    adresses = [int(x.split()[0]) for x in f.readlines()[2:]]
            if max_adress_sample is not None:
                file_len = len(adresses)
                if file_len > max_adress_sample:
                    offset = random.randint(0,file_len - max_adress_sample)
                    adresses = adresses[offset:offset + max_adress_sample]
            df.loc[len(df.index)] = [d, file, adresses] 
    df[["reindexed", "future_hits"]] = df.apply(slide_window, axis = 1, result_type="expand") 
    df = df.drop(["adresses"], axis = 1).explode(["reindexed", "future_hits"])
    df["current_hits"] = cache_hits(df["reindexed"])
    return df.reset_index()

tda_results = pd.DataFrame(columns=["window_length", "overlap", "number_windows", "train_mse", "test_mse", "val_mse"])

n = 150 # Number prediction

num_h_0 = 40
num_h_1 = 20

inf_dummy = 1000 # A number it can't possibly see to replace the inf which random forest dislikes

def prep_as_tda(data):
    x = list(range(0, len(data)))
    data = np.array(list(zip(x,data)))
    transformed =  ripser(data)['dgms']
    h_zeros = np.transpose(transformed[0])[1].tolist()
    h_ones = transformed[1]
    count_repeats = h_zeros.count(1)
    count_sqrt_2 = h_zeros.count(1.4142135381698608)
    sample_h_0 = sorted(np.random.choice(np.transpose(transformed[0])[1], num_h_0))
    count_h_1 = len(h_ones)

    if count_h_1 == 0:
        sample_h_1 = np.zeros((num_h_1, 2)) # If nothing, add nothing
    else:
        sample_h_1 = transformed[1][np.random.choice(len(transformed[1]), num_h_1)]

    sample_h_1 = list(sample_h_1.flatten())
    vals = np.array([count_repeats] + [count_sqrt_2] + [count_h_1] + sample_h_0 + sample_h_1)
    return np.nan_to_num(vals, posinf=inf_dummy)

for window_length in range(50, 251, 50):
    for overlap in range(20, 61, 10):
        if overlap >= window_length:
            continue
        # Setup dataset
        previous_hits = make_dataset(window_length=window_length, window_overlap=overlap, predict_next = 150, max_adress_sample = 20000)
        feature_prep = np.array([prep_as_tda(x) for x in previous_hits["reindexed"]])
        for number_windows in range(1, 6):
            print("window_length", window_length, "overlap", overlap, "number_windows", number_windows)
            keep = np.equal(previous_hits["file"].to_numpy()[number_windows:], previous_hits["file"].to_numpy() [:(number_windows * -1)])
            features = feature_prep
            features = np.array([np.concatenate(features[l:l+number_windows]) for l in range(len(features) - number_windows)])
            features = features[keep]
            labels = previous_hits["future_hits"].to_numpy()[number_windows :][keep]
            train_test =  previous_hits["dataset"].to_numpy()[number_windows:][keep] == "Baleen"
            split_index = round(sum(train_test) * 0.8)
            train_features = features[train_test][:split_index]
            test_features = features[train_test][split_index:]
            val_features = features[np.logical_not(train_test)]
            train_labels = labels[train_test][:split_index]
            test_labels = labels[train_test][split_index:]
            val_labels = labels[np.logical_not(train_test)]
            rf = RandomForestRegressor(n_estimators = 1000, random_state = 42, n_jobs=-1)
            # Train the model on training data
            rf.fit(train_features, train_labels)
            train_predictions = rf.predict(train_features)
            test_predictions = rf.predict(test_features)
            val_predictions = rf.predict(val_features)
            train_error = np.mean((train_predictions - train_labels)**2)
            test_error = np.mean((test_predictions - test_labels)**2)
            val_error = np.mean((val_predictions - val_labels)**2)
            tda_results.loc[len(tda_results.index)] = [window_length, overlap, number_windows, train_error, test_error, val_error] 
            tda_results.to_csv("Results/tda.csv")
            
tda_results
