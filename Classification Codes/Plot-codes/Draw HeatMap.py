import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

plt.rcParams['xtick.labelsize'] = 15
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.edgecolor'] = 'black' 
plt.rcParams['axes.linewidth'] = 5
dpi = 600


path = r'F:\Project All\Image to Image Translation (US to MRI)\Evaluate\Spearman_Correlation test on Radiomics features.xlsx'
spearman_g1 = pd.read_excel(path, 'Group 1',index_col=False).round(2)
spearman_g2 = pd.read_excel(path, 'Group 2', index_col=False).round(2)
spearman_g3 = pd.read_excel(path, 'Group 3', index_col=False).round(2)

def frame_image(img, frame_width, color=(255, 0, 0)):
    b = frame_width # border size in pixel
    ny, nx = img.shape[0], img.shape[1] # resolution / number of pixels in x and y
    framed_img = Image.new('RGB', (b+ny+b, b+nx+b), color) # RGB color tuple
    framed_img = np.array(framed_img.getdata()).reshape(framed_img.size[0], framed_img.size[1], 3)
    framed_img[b:-b, b:-b] = img
    return framed_img

	
cbar_kws = {
            "shrink":1,
           }
plt.figure(figsize = (5,5))
g1 = spearman_g1.set_index('FeatureNames').style.background_gradient(cmap ='Greens').set_properties(**{'font-size': '20px'})
sns_g1 = sns.heatmap(spearman_g1.set_index('FeatureNames'), cmap='Greens', linewidths=0.1, annot=True, cbar_kws=cbar_kws)
plt.title('i) Group 1')
plt.savefig(r'F:\Project All\Image to Image Translation (US to MRI)\Evaluate\Group 1_{}dpi.jpg'.format(dpi), dpi=dpi, bbox_inches="tight")


cbar_kws = {
            "shrink":0.26,
           }
plt.figure(figsize = (5,20))
g2 = spearman_g2.set_index('FeatureNames').style.background_gradient(cmap ='Greens').set_properties(**{'font-size': '20px'}) 
sns_g2 = sns.heatmap(spearman_g2.set_index('FeatureNames'), cmap='Greens', linewidths=0.1, annot=True, cbar_kws=cbar_kws)
plt.title('ii) Group 2')
plt.savefig(r'F:\Project All\Image to Image Translation (US to MRI)\Evaluate\Group 2_{}dpi.jpg'.format(dpi), dpi=dpi, bbox_inches="tight")


cbar_kws = {
            "shrink":0.2,
           }
plt.figure(figsize = (5,26))
g3 = spearman_g3.set_index('FeatureNames').style.background_gradient(cmap ='Greens').set_properties(**{'font-size': '20px'}) 
sns_g3 = sns.heatmap(spearman_g3.set_index('FeatureNames'), cmap='Greens', linewidths=0.1, annot=True, cbar_kws=cbar_kws)
plt.title('iii) Group 3')
plt.savefig(r'F:\Project All\Image to Image Translation (US to MRI)\Evaluate\Group 3_{}dpi.jpg'.format(dpi), dpi=dpi, bbox_inches="tight")