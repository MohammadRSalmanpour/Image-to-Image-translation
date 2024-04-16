import os
import pandas as pd
import numpy as np

parent_dir = r'F:\Project All\Image to Image Translation (US to MRI)\Evaluate\Results (New)'
list_dirs = os.listdir(parent_dir)

column_lists = ['Algorithm', 'train mae mean', 'train pmae mean', 'train mse mean', 'train psnr mean', 'train ssim mean', 
                'train mae std', 'train pmae std', 'train mse std', 'train psnr std', 'train ssim std',
                'val mae mean', 'val pmae mean', 'val mse mean', 'val psnr mean', 'val ssim mean',
                'val mae std', 'val pmae std', 'val mse std', 'val psnr std', 'val ssim std',
                'test mae mean', 'test pmae mean', 'test mse mean', 'test psnr mean', 'test ssim mean',
                'test mae std', 'test pmae std', 'test mse std', 'test psnr std', 'test ssim std']

df_results = pd.DataFrame(columns=column_lists)
results = []
for algorithm in list_dirs:
    data_path = os.path.join(parent_dir, algorithm)
    print(algorithm)
    for mode in ['train', 'val', 'test']:
        #csv_file = os.listdir(os.path.join(data_path, mode))
        df = pd.read_csv(os.path.join(data_path, mode+'.csv'))
        print(mode)
        mae_mean = df['MAE'].mean()
        mape_mean = df['MAPE'].mean()
        mse_mean = df['MSE'].mean()
        psnr_mean = df['PSNR'].mean()
        ssim_mean = df['SSIM'].mean()
        
        mae_std = df['MAE'].std()
        mape_std = df['MAPE'].std()
        mse_std = df['MSE'].std()
        psnr_std = df['PSNR'].std()
        ssim_std = df['SSIM'].std()
        
        results.extend([mae_mean, mape_mean, mse_mean, psnr_mean, ssim_mean,
                        mae_std, mape_std, mse_std, psnr_std, ssim_std])
    
    results = list(np.around(np.array(results),2))
    results.insert(0, algorithm)
    print(results)
    df_results.loc[len(df_results)] = results
    results = []
	
df_results.to_excel(os.path.join(parent_dir, 'Results.xlsx'))