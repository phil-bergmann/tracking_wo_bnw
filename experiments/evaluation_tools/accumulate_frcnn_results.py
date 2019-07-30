import os
import csv
import numpy as np


if __name__ == "__main__":
    results_dir = 'output/tracktor/MOT17'

    trackers = ['Tracktor++', 'Tracktor-no-FPN++', 'eHAF17',
                'FWT', 'jCC', 'MOTDT17', 'MHT_DAM']

    for t in trackers:
        eval_path = os.path.join(results_dir, t, 'eval_online.txt')

        results = []
        # print(eval_path)
        with open(eval_path, mode='r') as file:
            reader = csv.reader(file)
            for r in reader:
                if 'FRCNN' in r[0]:
                    r.pop(0)
                    r[5] = r[5].replace(' %', '')
                    r[6] = r[6].replace(' %', '')
                    results.append([float(i) for i in r])

        results = np.array(results)
        # print(results)
        # & Method & MOTA $\uparrow$ & IDF1 $\uparrow$ & MT $\uparrow$ & ML $\downarrow$ & FP $\downarrow$ & FN $\downarrow$ & ID Sw.
        print(f'{t} & {results[:, 0].mean():.2f} & '
              f'{results[:, 1].mean():.2f} & '
              f'{results[:, 5].mean():.2f} & '
              f'{results[:, 6].mean():.2f} & '
              f'{int(results[:, 7].sum())} & '
              f'{int(results[:, 8].sum())} & '
              f'{int(results[:, 9].sum())} \\\\')
