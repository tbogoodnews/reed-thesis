from timeit import default_timer as timer
import pandas as pd
import numpy as np
from ripser import Rips
from ripser import ripser
from plotnine import *


tda_speed = pd.DataFrame(columns=["length", "time"])
num_trials = 500
gen_data = lambda length : reindex(list(np.random.randint(1.5, size = length)))
for length in [5] + list(range(25, 350, 25)) :
    for _ in range(num_trials):
        sample = gen_data(length=length)
        rips = Rips()
        x = list(range(0, length))
        data = np.array(list(zip(x,sample)))
        start = timer()
        ripser(data)['dgms']
        end = timer()
        tda_speed.loc[len(tda_speed.index)] = [length, end - start]


(
    ggplot(tda_speed, aes(x = "length", y = "time", group = "length")) +
    geom_boxplot(notch=False, outlier_shape = "") +
    scale_y_log10() +
    labs(x = "Length of Trace", y = "Seconds (Log scale)")
)
