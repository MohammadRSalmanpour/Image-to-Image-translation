import os
import xlrd
import numpy as np
import xlsxwriter
import matplotlib as mpl
mpl.use('TkAgg')  # or whatever other backend that you want to solve Segmentation fault (core dumped)
import matplotlib.pyplot as plt
import pandas as pd


def draw_box_plot(data_list, method_names, path):
    filenames = ['MAE', 'PMAE', 'MSE', 'PSNR', 'SSIM']
    expressions = [' (lower is better)', ' (lower is better)', ' (lower is better)', ' (higher is better)', ' (higher is better)']
    colors = ['red', 'green', 'blue', 'aquamarine', 'olive', 'brown', 'grey', 'purple']

    for idx, data in enumerate(data_list):
        fig1, ax1 = plt.subplots(figsize=(2.5*len(method_names), 5))
        box = ax1.boxplot(np.transpose(data), patch_artist=True, showmeans=True, sym='r+', vert=True)

        # connect mean values
        y = data.mean(axis=1)
        ax1.plot(range(1, len(method_names)+1), y, 'r--')

        for patch, color in zip(box['boxes'], colors):
            patch.set(facecolor=color, alpha=0.5, linewidth=1)

        # scatter draw datapoints
        x_vals, y_vals = [], []
        for i in range(data_list[0].shape[0]):
            # move x coordinate to not overlapping
            x_vals.append(np.random.normal(i + 0.7, 0.04, data.shape[1]))
            y_vals.append(data[i, :].tolist())

        for x_val, y_val, color in zip(x_vals, y_vals, colors):
            ax1.scatter(x_val, y_val, s=5, c=color, alpha=0.5)

        ax1.yaxis.grid()  # horizontal lines
        ax1.set_xticklabels([method_name for method_name in method_names], fontsize=14, fontweight="bold")
        for tick in ax1.yaxis.get_major_ticks():
            tick.label.set_fontsize(14)
            
        plt.setp(box['medians'], color='black')
        plt.title(filenames[idx] + expressions[idx], fontsize=14, fontweight="bold")
        plt.ylabel(filenames[idx], fontsize=18, fontweight="bold")
        #plt.show()
        plt.savefig(os.path.join(path, filenames[idx] + '.jpg'), dpi=300)
        plt.close()
		

def main(path, methods, display_names, num_tests=120):
    # measures = ['MAE', 'RMSE', 'MAPE','PSNR', 'SSIM']
    mae_overall = np.zeros((len(methods), num_tests), dtype=np.float64)
    mape_overall = np.zeros((len(methods), num_tests), dtype=np.float64)
    mse_overall = np.zeros((len(methods), num_tests), dtype=np.float64)
    psnr_overall = np.zeros((len(methods), num_tests), dtype=np.float64)
    ssim_overall = np.zeros((len(methods), num_tests), dtype=np.float64)
    

    for method_idx, method in enumerate(methods):
        print('method_idx: {}'.format(method_idx))
        workbook = xlrd.open_workbook(os.path.join(path, method, 'test.xls'))
        worksheet = workbook.sheet_by_name('test')
        #worksheet = pd.read_excel(os.path.join(path, method, 'test.xlsx'))

        for row_idx in range(1, worksheet.nrows-2):
            mae_overall[method_idx, row_idx-1] = float(worksheet.cell(row_idx, 1).value)
            mape_overall[method_idx, row_idx-1] = float(worksheet.cell(row_idx, 2).value)
            mse_overall[method_idx, row_idx-1] = float(worksheet.cell(row_idx, 3).value)
            psnr_overall[method_idx, row_idx-1] = float(worksheet.cell(row_idx, 4).value)
            ssim_overall[method_idx, row_idx-1] = float(worksheet.cell(row_idx, 5).value)
            

    # draw boxplot
    draw_box_plot([mae_overall, mape_overall,  mse_overall, psnr_overall, ssim_overall], display_names, path)

# main	
main_path = '/home/mohammad/Desktop/GAN/Algorithms/Algorithm Results (New)'

methods_ = ['2D-Pix2Pix', '3D-AutoEncoder', '2D-DualGAN', '2D-DiscoGAN', '3D-UNET', '2D-GcGAN', '2D-CycleGAN', '3D-CycleGAN']
display_names_ = ['2D-Pix2Pix', '3D-AutoEncoder', '2D-DualGAN', '2D-DiscoGAN', '3D-UNET', '2D-GcGAN', '2D-CycleGAN', '3D-CycleGAN']

main_path = r'F:\Project All\Image to Image Translation (US to MRI)\Evaluate\BoxPlots'
main(main_path, methods_, display_names_)