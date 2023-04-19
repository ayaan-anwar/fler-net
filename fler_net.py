import datetime
import json
import pickle
import os

import deap_utils

from feature_extractor import FeatureExtractor
from base import BaseModel

if __name__ == '__main__':
    dataset = 'deap'
    label = 'val'
    ts = datetime.utcnow().strftime("%Y-%m-%d")

    # Config
    with open('config/deap.json') as f:
        conf = json.load(f)
    
    # Load data
    (sub_data, sub_labels) = deap_utils.load_deap_data('DEAP/')

    fe = FeatureExtractor(conf)
    fe.extract_features(sub_data, sub_labels)
    
    del sub_data
    del sub_labels
    del fe

    basemodel = BaseModel(label, 'DEAP_Features/')
    model, results = basemodel.train()

    # Results
    if not os.path.exists("Results/"):
        os.makedirs(f"Results/{dataset}/{ts}")
    
    model.save(f"Results/{dataset}/{ts}/model_{label}.h5")
    with open(f"Results/{dataset}/{ts}/history_{label}.pkl", 'wb') as f:
        pickle.dump(results.history, f)
