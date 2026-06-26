import numpy as np

def mean_median_mode(x):
    x = np.asarray(x)

    mean = np.mean(x)
    median = np.median(x)

    values, counts = np.unique(x, return_counts=True)
    mode = values[np.argmax(counts)]

    return mean, median, mode