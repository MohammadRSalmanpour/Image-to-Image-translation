import os
import pandas as pd
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')  # or whatever other backend that you want to solve Segmentation fault (core dumped)
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
import random

parent_dir = r'F:\Project All\Image to Image Translation (US to MRI)\Evaluate\Combination'
# Load AUC and Accuracy files
df = pd.read_excel(os.path.join(parent_dir, 'auc_2.xlsx'))

# Create the bar plot
models = ['RandF', 'PCA+RandF', 'ResNet50']
fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(12,8))
random_colors = random.choices(list(mcolors.CSS4_COLORS.keys()), k=18)
colors_list = random_colors

font_size_labels = 12
font_size_title = 18

plt.title("Prediction Results for AUC Metric", fontdict={'fontsize': font_size_title, 'fontweight': 'bold'})
for i, model in enumerate(models):
    ax[i].grid()
    ax[i].bar(df['Combination'], df[str(model)], color=colors_list, yerr=df[str(model)+'_std'])
    ax[i].set_ylabel('AUC', fontdict={'fontsize': font_size_labels, 'fontweight': 'bold'})
    ax[i].set_xlabel('Combination', fontdict={'fontsize': font_size_labels, 'fontweight': 'bold'})
    ax[i].set_title(model, fontdict={'fontsize': font_size_title, 'fontweight': 'bold'})
    ax[i].set_xticks(ticks=range(18), labels=df['Combination'], rotation ='vertical', fontdict={'fontweight': 'bold'})
    ax[i].set_yticks(np.arange(0, 1.1, step=0.1))
    #ax.legend(title='Fruit Color')

fig.tight_layout()
#plt.title("Prediction Results for Accuracy Metric", fontdict={'fontsize': 20, 'fontweight': 'bold'})
plt.savefig('F:\Project All\Image to Image Translation (US to MRI)\Evaluate\Combination\AUC_BarPlot_2.jpg', dpi=1200)
plt.show()