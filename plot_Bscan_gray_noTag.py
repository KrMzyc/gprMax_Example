import argparse
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from gprMax.exceptions import CmdInputError
from .outputfiles_merge import get_output_data
from scipy.ndimage import gaussian_filter

def add_gaussian_noise(image, mean=0, std=5):
    """给图像添加高斯噪声。

    Args:
        image (array): 输入图像数组。
        mean (float): 高斯噪声的均值。
        std (float): 高斯噪声的标准差。

    Returns:
        array: 添加了高斯噪声后的图像数组。
    """
    noisy_image = image + np.random.normal(mean, std, image.shape)
    return np.clip(noisy_image, -200, 400)  # 将值限制在0和1之间

def mpl_plot(filename, outputdata, dt, rxnumber, rxcomponent, save_path=None):
    """创建B扫描的matplotlib图并保存。

    Args:
        filename (string): 输出文件的文件名（包括路径）。
        outputdata (array): A扫描数组，即B扫描数据。
        dt (float): 模型的时间分辨率。
        rxnumber (int): 接收器输出编号。
        rxcomponent (str): 接收器输出字段/电流分量。
        save_path (string): 保存图像的路径，如果为None，则不保存。

    Returns:
        plt (object): matplotlib图对象。
    """

    (path, filename) = os.path.split(filename)
    filename_base=filename.replace("_merged.out","")
    # 创建图形，设置图像大小为100x150像素，去除坐标轴和图例
    fig, ax = plt.subplots(num=filename + ' - rx' + str(rxnumber), 
                           figsize=(100/80, 150/80), facecolor='w', edgecolor='w')
    ax.set_axis_off()
   
    # Apply Gaussian filter for smoothing
    outputdata_smoothed = gaussian_filter(outputdata, sigma=1)
    #outputdata[abs(outputdata) < 10] *= 10
    # Apply sharpening filter
    sharpening_factor = -80  # Increase sharpening strength
    outputdata_sharpened = outputdata + (outputdata - outputdata_smoothed) * sharpening_factor
    
    # 添加高斯噪声
    outputdata_noisy = add_gaussian_noise(outputdata_sharpened)

    # 显示图像
    img = ax.imshow(outputdata_noisy, 
                    extent=[0, outputdata_noisy.shape[1], outputdata_noisy.shape[0] * dt, 0], 
                    interpolation='nearest', aspect='auto', cmap='gray', 
                    vmin=-np.amax(np.abs(outputdata_noisy)), vmax=np.amax(np.abs(outputdata_noisy)))
    plt.ylim(0.25e-8, outputdata_noisy.shape[0] * dt)
    plt.gca().invert_yaxis()
    # 保存图像
    if save_path:
        save_filename = os.path.join(save_path, f"{filename_base}_small.png")
        plt.savefig(save_filename, bbox_inches='tight', pad_inches=0)

    return plt

if __name__ == "__main__":
    # 指定保存图像的路径，如果为None，则不保存
    save_path = "D:\\gprMAX\\gprMax-master\\image_result\\images"  # 修改为实际路径

    # 解析命令行参数
    parser = argparse.ArgumentParser(description='绘制B扫描图像。', 
                                     usage='cd gprMax; python -m tools.plot_Bscan outputfile output')
    parser.add_argument('outputfile', help='输出文件的名称，包括路径')
    parser.add_argument('rx_component', help='要绘制的输出分量的名称', 
                        choices=['Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz', 'Ix', 'Iy', 'Iz'])
    args = parser.parse_args()

    # 打开输出文件并读取输出数量（接收器数量）
    f = h5py.File(args.outputfile, 'r')
    nrx = f.attrs['nrx']
    f.close()

    # 检查是否有接收器
    if nrx == 0:
        raise CmdInputError('在{}中找不到接收器'.format(args.outputfile))

    for rx in range(1, nrx + 1):
        outputdata, dt = get_output_data(args.outputfile, rx, args.rx_component)
        plthandle = mpl_plot(args.outputfile, outputdata, dt, rx, args.rx_component, save_path=save_path)

    plthandle.show()
