import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import seaborn as sns
# sns.set_palette('deep')
sns.set(font_scale=1.5, rc={'text.usetex': True})

plt.style.use('default')

if __name__ == "__main__":
    results_dir = 'output/tracktor'

    frame_skips = [1, 2, 3, 5, 6, 10, 15, 30]
    trackers = ['Tracktor', 'Tracktor++']

    results = {}
    for t in trackers:
        results[t] = []

        for f_s in frame_skips:
            dataset = f"MOT17_{30 // f_s}_FPS"
            eval_path = os.path.join(results_dir, dataset, t, 'eval.txt')

            f = open(eval_path, 'r')
            linelist = f.readlines()
            f.close

            idf1 = float(linelist[0].split(',')[0])
            mota = float(linelist[0].split(',')[-1])

            results[t].append([idf1, mota])

    fontsize = 16
    tickfontsize = 12

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    fig, ax1 = plt.subplots()
    plt.tick_params(labelsize=tickfontsize)

    # for t, data in results.items():
    #     sns.lineplot(x=30 // np.array(frame_skips), y=np.array(data)[:, 0], label=f'{t} - IDF1')
    #     sns.lineplot(x=30 // np.array(frame_skips),
    #                  y=np.array(data)[:, 1], label=f'{t} - MOTA')

    x = 30 // np.array(frame_skips)
    y = []
    linestyle = ['solid', 'dashed']
    for ls, (t, data) in zip(linestyle, results.items()):
        # y.append(np.array(data)[:, 0])
        # y.append(np.array(data)[:, 1])
        plt.plot(x, np.array(data)[:, 0], label=f'{t} - IDF1', ls=ls)
        plt.plot(x, np.array(data)[:, 1], label=f'{t} - MOTA', ls=ls)

    plt.xlabel('FPS', fontsize=tickfontsize)

    # df = pd.DataFrame({'Tracktor-IDF1': y[0],
    #                    'Tracktor-MOTA': y[1],
    #                    'Tracktor++-IDF1': y[2],
    #                    'Tracktor++-MOTA': y[3],
    #                    'FPS': 30 // np.array(frame_skips)})
    # df = pd.DataFrame({'Metrics': np.c_[y[0], y[1], y[2], y[3]],
    #                    'FPS': 30 // np.array(frame_skips)})
    # ax = sns.lineplot(data=pd.melt(df, ['FPS']), x='FPS', y='value')
    ax1.set(xticks=30 // np.array(frame_skips))
    ax1.set_ylim([30, 65])
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.set_aspect(aspect=0.5)

    # plt.grid()

    # ax1.set_aspect(aspect=0.4)
    # ax1.patch.set_edgecolor('black')
    # ax1.patch.set_facecolor('white')

    legend = plt.legend(loc='lower right', fontsize=tickfontsize)
    frame = legend.get_frame()
    frame.set_facecolor('white')

    plt.savefig('mot17_low_fps_tracktor.pdf',
                format='pdf', bbox_inches='tight')
