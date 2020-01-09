
import numpy as np

def accuracy(predictions, targets):
    total_num = len(predictions)
    hit_num = int(np.sum(predictions == targets))
    return {"total_num": total_num,
            "hit_num": hit_num,
            "accuracy": 1.0 * hit_num / total_num}