
import pandas
import json
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import matplotlib.animation as animation
from matplotlib.pyplot import MultipleLocator

def moving_average(interval, windowsize):
 
    window = np.ones(int(windowsize)) / float(windowsize)
    re = np.convolve(interval, window, 'same')
    return re

plt.style.use('ggplot')

map_names = ['2c_vs_64zg','3s_vs_4z','3m','3s_vs_5z','27m_vs_30m','2s3z']
title_names = [name.replace("_vs_"," vs. ") for name in map_names]
for map_name, title_name in zip(map_names,title_names):
    plt.figure()
    ###################################PPO###################################
    exp_names = ['mappo_step'] 
    label_names = ["MAPPO"]
    color_names = ['red']

    save_dir = './walltime/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    max_steps = []
    for exp_name, label_name, color_name in zip(exp_names, label_names, color_names):
        print(exp_name)
        data_dir =  './walltime/' + map_name + '/' + map_name + '_' + exp_name + '.csv'

        df = pandas.read_csv(data_dir)
        
        key_cols = [c for c in df.columns if 'MIN' not in c and 'MAX' not in c]
        key_step = [n for n in key_cols if n == 'Step']
        key_win_rate = [n for n in key_cols if n != 'Step']

        all_step = np.array(df[key_step])
        all_win_rate = np.array(df[key_win_rate])

        print("original shape is ")
        print(all_step.shape)
        print(all_win_rate.shape)

        df_final = df[key_cols].dropna()
        step = df_final[key_step]
        win_rate = df_final[key_win_rate]

        print("drop nan shape is")
        print(np.array(step).shape)
        print(np.array(win_rate).shape)

        max_step = step.max()['Step']
        print("max step is {}".format(max_step))

        if "ppo" in exp_name and max_step < 4.96e6:
            print("error: broken data! double check!")

        if max_step < 4e6:
            max_step = 2e6
        elif max_step < 9e6:
            max_step = 5e6
        else:
            max_step = 10e6

        print("final step is {}".format(max_step))
        max_steps.append(max_step)

        df_final = df_final.loc[df_final['Step'] <= max_step] 

        x_step = np.array(df_final[key_step]).squeeze(-1)
        y_seed = np.array(df_final[key_win_rate])

        median_seed = np.median(y_seed, axis=1)
        std_seed = np.std(y_seed, axis=1)
        plt.plot(x_step, median_seed, label = label_name, color=color_name)
        plt.fill_between(x_step,
            median_seed - std_seed,
            median_seed + std_seed,
            color=color_name,
            alpha=0.1)
    print("########################QMIX##########################")
    exp_name = 'qmix_step'
    label_name = "QMIX"
    color_name = 'blue'
    print(exp_name)
    data_dir =  './walltime/' + map_name + '/' + map_name + '_' + exp_name + '.csv'

    df = pandas.read_csv(data_dir)

    key_cols = [c for c in df.columns if 'MIN' not in c and 'MAX' not in c]
    key_step = [n for n in key_cols if n == 'Step']
    key_win_rate = [n for n in key_cols if n != 'Step']


    one_run_max_step = []
    qmix_x_step = []
    qmix_y_seed = []
    for k in key_win_rate:
        print("one run original shape is ")
        print(np.array(df[k]).shape)

        df_final = df[[k, 'Step']].dropna()
        step = df_final[key_step]
        win_rate = df_final[k]
        
        print("one run drop nan shape is")
        print(np.array(step).shape)
        print(np.array(win_rate).shape)

        max_step = step.max()['Step']
        print("one run max step is {}".format(max_step))

        if max_step < 2e6:
            print("error: broken data! double check!")
            continue

        if max_step < 4e6:
            max_step = 2e6
        elif max_step < 9e6:
            max_step = 5e6         
        else:
            max_step = 10e6
        
        one_run_max_step.append(max_step)
        print("final step is {}".format(max_step))

        df_final = df_final.loc[df_final['Step'] <= max_step] 
        qmix_x_step.append(np.array(df_final[key_step]).squeeze(-1))
        qmix_y_seed.append(np.array(df_final[k]))
        print("data shape is {}".format(np.array(df_final[k]).shape))

    # pick max qmix step
    qmix_max_step = np.min(one_run_max_step)
    max_steps.append(qmix_max_step)

    # adapt sample frequency
    sample_qmix_s_step = []
    sample_qmix_y_seed = []
    final_max_length = []
    for x, y in zip(qmix_x_step, qmix_y_seed):
        final_max_length.append(len(x))
        sample_qmix_s_step.append(x)
        sample_qmix_y_seed.append(y)

    # truncate numpy
    max_common_length = np.min(final_max_length)
    print("max common qmix length is {}".format(max_common_length))
    final_qmix_x_step = []
    final_qmix_y_seed = []
    for x, y in zip(sample_qmix_s_step, sample_qmix_y_seed):
        final_qmix_x_step.append(x[:max_common_length])
        final_qmix_y_seed.append(y[:max_common_length])

    x_step = np.mean(final_qmix_x_step, axis=0)
    y_seed = np.array(final_qmix_y_seed)

    median_seed = np.median(y_seed, axis=0)
    std_seed = np.std(y_seed, axis=0)
    plt.plot(x_step, median_seed, label=label_name, color=color_name)
    plt.fill_between(x_step,
        median_seed - std_seed,
        median_seed + std_seed,
        color=color_name,
        alpha=0.1)

    plt.tick_params(axis='both',which='major') 
    final_max_step = np.min(max_steps)
    print("final max step is {}".format(final_max_step))
    x_major_locator = MultipleLocator(int(final_max_step/5))
    x_minor_Locator = MultipleLocator(int(final_max_step/10)) 
    y_major_locator = MultipleLocator(0.2)
    y_minor_Locator = MultipleLocator(0.1)
    ax=plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    ax.yaxis.set_major_locator(y_major_locator)
    ax.xaxis.set_minor_locator(x_minor_Locator)
    ax.yaxis.set_minor_locator(y_minor_Locator)
    ax.xaxis.get_major_formatter().set_powerlimits((0,1))
    #ax.xaxis.grid(True, which='minor')
    plt.xlim(0, final_max_step)
    plt.ylim([0, 1.1])
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel('Timesteps', fontsize=20)
    plt.ylabel('Win Rate', fontsize=20)
    plt.legend(loc='best', numpoints=1, fancybox=True, fontsize=20)

    plt.savefig(save_dir + map_name + "_step.png", bbox_inches="tight")