import numpy as np

def preprocess(state):
    snr = np.array(state["snr"]) / 30.0
    cqi = np.array(state["cqi"]) / 15.0
    queue = np.array(state["queue"]) / 100.0

    return np.concatenate([snr, cqi, queue])
