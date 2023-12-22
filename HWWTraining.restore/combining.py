# Imports

import pandas as pd
import uproot as ur


from all_training_utils import events_to_training, build_loss, simple_deep_dense_net

print("Starting Events")
with ur.open("/home/kye/projects/ctb-stelzer/kye/HWWTraining/Data/nTuples-cHj3-cHW-12M.root") as file:
    events = file["HWWTree_emme"].arrays(library="pandas").copy()

cols_to_keep = list(events.columns[0:48]) + list(events.columns[79:87]) + list(events.columns[-1:])
events = events[cols_to_keep].copy()
print("Finished with events")

with ur.open("/home/kye/projects/ctb-stelzer/kye/HWWTraining/Data/top.root") as file:
    top = file["HWWTree_emme"].arrays(library="pandas").copy()
with ur.open("/home/kye/projects/ctb-stelzer/kye/HWWTraining/Data/WW.root") as file:
    ww = file["HWWTree_emme"].arrays(library="pandas").copy()
with ur.open("/home/kye/projects/ctb-stelzer/kye/HWWTraining/Data/Zjets.root") as file:
    zjets = file["HWWTree_emme"].arrays(library="pandas").copy()

print("building background")

background = pd.concat((top, ww, zjets), axis=0)
print(background.columns)
background = pd.concat((background[background.columns[0:-1]], *([background[background.columns[-2:-1]]] * 16), background[background.columns[-1:]]), axis=1)
print(background.columns)
print(events.columns)
background.columns = events.columns
print(background.columns)

print("Building entire dataset")

all_events = pd.concat((events, background), axis=0)

print("saving")

all_events.to_pickle("/home/kye/projects/ctb-stelzer/kye/HWWTraining/Data/all_data.pkl")

print(events.shape)



# for index, col in enumerate(events[cols_to_keep].columns):
#     print(f"{index}: {col}")