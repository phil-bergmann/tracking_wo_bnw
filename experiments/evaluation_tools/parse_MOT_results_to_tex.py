import pandas as pd

RESULTS_PATH = 'MOTResults.txt'

with open(RESULTS_PATH, 'r') as f:
    f_content = f.readlines()

start_ixs = list(range(0, len(f_content), 4))
end_ixs = list(range(0, len(f_content) + 4, 4))[1:]

metrics_res = {}

for start, end in zip(start_ixs, end_ixs):
    scene_name = f_content[start].strip()
    metric_vals = f_content[end - 2].split()
    metric_vals = [val.replace('|', '').strip() for val in metric_vals]
    metrics_res[scene_name] = metric_vals

metrics_names =f_content[1].replace('|', '').replace('\n', '').split()

print(metrics_names)

for seq_name, data in metrics_res.items():
    print(f"{seq_name} & "
          f"{data[metrics_names.index('MOTA')]} & "
          f"{data[metrics_names.index('IDF1')]} & "
          f"{float(data[metrics_names.index('MT')]) / float(data[metrics_names.index('GT')]) * 100:.1f} & "
          f"{float(data[metrics_names.index('ML')]) / float(data[metrics_names.index('GT')]) * 100:.1f} & "
          f"{data[metrics_names.index('FP')]} & "
          f"{data[metrics_names.index('FN')]} & "
          f"{data[metrics_names.index('IDs')]} \\\\")
