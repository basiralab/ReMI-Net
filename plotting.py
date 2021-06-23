import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matplotlib._color_data import BASE_COLORS
from sklearn.manifold import TSNE

def plot_cbt(img, fold_num=1, timepoint=0):
    img = np.repeat(np.repeat(img, 10, axis=1), 10, axis=0)
    plt.imshow(img)
    plt.title(f"CBT at Fold {fold_num} - Time {timepoint}")
    plt.axis('off')
    plt.colorbar()
    plt.show()