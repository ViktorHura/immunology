import matplotlib.pyplot as plt
import numpy as np
import logging
import os


def plot_losses(epochs, losses, title="Training loss", ytitle="loss", xtitle="epoch"):
    x = np.array(range(epochs))
    y = np.array(losses)

    plt.plot(x, y)

    plt.title(title)
    plt.ylabel(ytitle)
    plt.xlabel(xtitle)


def setupLogger(path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    logger.addHandler(ch)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    fh = logging.FileHandler(path, mode="w")
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)
    return logger