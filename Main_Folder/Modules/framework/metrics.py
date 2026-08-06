                    # the metrics functions on the loop of registration
from typing import Optional
import numpy as np
import torch




# mattes mutual information ( MMI)
def histogram_mutual_information(image1, image2):
    hgram, x_edges, y_edges = np.histogram2d(image1.ravel(), image2.ravel(), bins=100)
    pxy = hgram / float(np.sum(hgram))
    px = np.sum(pxy, axis=1)
    py = np.sum(pxy, axis=0)
    px_py = px[:, None] * py[None, :]
    nzs = pxy > 0
    return np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))


# utils for metrics
def metric_outputs_update(start_dict : dict,
                          new_data: Optional[dict] = None,
                          )->dict:
    '''
    update input dict for a general type of metric to store output results

    Args:
        start_dict (dict): the starting point
        new_data ( Optional[dict], default to None): the new data

    Returns:
        dict: the update data list
    '''
    # ======== TRASFORM DATA TYPE ==========
    for k, v in start_dict.items():
        if isinstance(v, torch.Tensor):
            v=v.detach().cpu()
        if not isinstance(v, list):
            start_dict[k] = [v]

    if new_data is None:
        return start_dict
    
    for k, v in new_data.items():
        if isinstance(v, torch.Tensor):
            v= v.detach().cpu()

        if k not in start_dict:
            start_dict[k]=[v]
        else:
            start_dict[k].append(v)
    
    return start_dict
