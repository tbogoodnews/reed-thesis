from plotnine import *
from mizani.formatters import percent_format
import pandas as pd
from pathlib import Path

len_label = lambda s : "Length: " + str(s)

Hits = pd.DataFrame(columns = ["Dataset", "Length", "Hits"])

for ds in ["Baleen", "Nexus 5"]:
    if ds == "Baleen":
        result = list(Path("../Data/baleen").rglob("*.trace"))
    else:
        result = list(Path("../Data/block_Nexus/Trace_files").rglob("*"))
    for length in [150,500, 1000]:
        num_hits = []
        for file in result:
            adresses = [int(x.split()[0]) for x in open(file, "r").readlines()[2:]]
            for i in range( 0, len(adresses) - length, length // 2):
                slice = adresses[i:i+length]
                num_hits.append(length - len(set(slice)))
        df = pd.DataFrame({'Dataset': len(num_hits) * [ds], 'Length' : len(num_hits) * [length], "Hits" : num_hits})
        Hits = pd.concat([Hits, df], ignore_index = True)
Hits = Hits.astype({'Length': 'int32', 'Hits': 'int32'})

(
    ggplot(Hits.sample(1000000), aes(x="Hits", colour="Dataset"))
    + geom_density()
    + facet_grid(cols = "Length", scales="free", labeller=labeller(cols=len_label))
    + labs(x="Repeated adresses")
    + scale_y_continuous(labels=percent_format())
 + theme(axis_title_y=element_blank())
 + theme(axis_ticks_major_y=element_blank())
 + theme(figure_size=(9, 6))
             )
