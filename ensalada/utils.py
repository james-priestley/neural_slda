import numpy as np 
import numba as nb

def to_trial_labels(trial):
    trial_labels = []
    if trial.baited:
        trial_labels.append("baited")
    trial_labels.append(trial.side)
    trial_labels.append(trial.context)
    #trial_labels.append(f"position_{trial.position}")
    trial_labels.append(f"odor_{trial.odor}")
    
    return trial_labels

@nb.jit(nopython=True)
def raster2documents(raster, min_length=0):
    documents = []
    n_cells, n_times = raster.shape
    for t in range(n_times):
        doc = ""
        for i in range(n_cells):
            doc += (" " + str(i))*raster[i, t]
        
        if len(doc) > min_length:
            documents.append(doc)
    
    return documents

def zip_rasters_labels(rasters, trial_labels, label_pre_post=False):
    
    assert len(rasters) == len(trial_labels)
    
    all_labeled_documents = []
    
    for raster, trial_label in zip(rasters, trial_labels):
        documents = raster2documents(raster)
        labeled_documents = [(doc, trial_label) for doc in documents]
        all_labeled_documents.extend(labeled_documents)
        
    return all_labeled_documents