import pickle


######### pickle

def save_pickle(x, filepath):
    with open(filepath, 'wb') as handle:
        pickle.dump(x, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_pickle(filepath):
    with open(filepath, 'rb') as handle:
        return pickle.load(handle)