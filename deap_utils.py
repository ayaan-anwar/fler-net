import _pickle as cPickle

VALENCE = 0
AROUSAL = 1
DOMINANCE = 2
NUM_SUBJECTS = 32
NUM_SITUATIONS = 40
EEG_CHANNELS = 32


def load_deap_data(deap_path: str) -> tuple:
    sub_data = {}
    sub_labels= {}

    for sub in range(1, NUM_SUBJECTS + 1):
        (data, labels) = get_subject_data_deap(deap_path=deap_path, sub_id=sub)
        subject_data = {}
        subject_labels = []
        print(f"Subject {sub:02d}")
        for sit in range(0, NUM_SITUATIONS):
            subject_data[sit + 1] = data[sit][:EEG_CHANNELS][:]
            val = labels[sit][VALENCE]
            aro = labels[sit][AROUSAL]
            dom = labels[sit][DOMINANCE]
            subject_labels.append([val, aro, dom])
            print(f"\t> Situation {sit:02d}")
        sub_data[sub] = subject_data
        sub_labels[sub] = subject_labels
        print(f"Done...\n---")
    
    return (sub_data, sub_labels)

def get_subject_data_deap(deap_path: str, sub_id: int) -> tuple:
    path = deap_path + "/data_preprocessed_python/s%02d.dat" % (sub_id)
    sub_dict = cPickle.load(open(path, 'rb'), encoding='iso-8859-1')
    return (sub_dict["data"], sub_dict["labels"])