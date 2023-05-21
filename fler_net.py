import datetime
import json
import pickle
import os
import sys

import deap_utils
import sam40_utils

from feature_extractor import FeatureExtractor
from base import BaseModel
from federated import FederatedModel

if __name__ == '__main__':
    #--- Run Arguments ---#
    # TODO: Add argparser to handle these
    #---      START    ---#
    dataset = 'sam40'
    label = 'val'
    data_dir = 'Data/SAM40/'
    feature_dir = 'Data/SAM40_Features/'
    ts = datetime.datetime.utcnow().strftime("%Y-%m-%d")
    extract_features = False
    train_base = False
    train_fed = True
    #---      END    ---#

    if extract_features == True:
        # Config
        with open(f'config/{dataset}.json') as f:
            conf = json.load(f)
        
        # Load data
        if dataset == 'deap':
            (sub_data, sub_labels) = deap_utils.load_deap_data(data_dir)
        elif dataset == 'sam40':
            (sub_data, sub_labels) = sam40_utils.load_sam40_data(data_dir)
        else:
            print('Unrecognized dataset')
            sys.exit(-1)

        fe = FeatureExtractor(conf)
        fe.extract_features(sub_data, sub_labels, feature_dir)
        
        del sub_data
        del sub_labels
        del fe

    if train_base == True:
        basemodel = BaseModel(label, feature_dir)
        model, results = basemodel.train()

        # Results
        if not os.path.exists(f"Results/{dataset}/{ts}/"):
            os.makedirs(f"Results/{dataset}/{ts}")

        model.save(f"Results/{dataset}/{ts}/model_{label}.h5")
        with open(f"Results/{dataset}/{ts}/history_{label}.pkl", 'wb') as f:
            pickle.dump(results.history, f)
        del model
        del results

    if train_fed == True:
        fedmodel = FederatedModel('val', 4, feature_dir)
        results = fedmodel.train()
        # Results
        if not os.path.exists(f"Results/{dataset}/{ts}"):
            os.makedirs(f"Results/{dataset}/{ts}")
        with open(f"Results/{dataset}/{ts}/fed_history_{label}.pkl", 'wb') as f:
            pickle.dump(results, f)
