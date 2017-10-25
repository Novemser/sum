import numpy as np


def aggregate(logging_outputs, key, default=0, avg=True):
    """
    Aggregate logging outputs
    """
    if avg:
        res = np.mean(list(log.get(key, default) for log in logging_outputs) )
    else:
        res = sum(log.get(key, default) for log in logging_outputs) 
    return res