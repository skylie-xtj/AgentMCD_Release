import os
import random
import numpy as np
import torch
import numpy as np
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)

def fix_seeds(seed=101):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed) # In order to disable hash randomization and make the experiment reproducible.
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

def metrics(y_label, y_predict):
    scores = {}
    if y_predict is None or y_label is None:
        print(y_predict, y_label)
    # scores['auc'] = round(roc_auc_score(y_label, y_predict, average='macro'), 5)
    y_predict = np.around(np.array(y_predict)).astype(int)
    scores['f1'] = round(f1_score(y_label, y_predict, average='macro'), 5)
    scores['recall'] = round(recall_score(y_label, y_predict, average='macro'), 5)
    scores['precision'] = round(precision_score(y_label, y_predict, average='macro'), 5)
    scores['acc'] = round(accuracy_score(y_label, y_predict), 5)

    return scores
