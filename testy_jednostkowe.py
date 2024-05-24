from scipy.stats import ttest_rel
import numpy as np



results = np.loadtxt("results.csv", delimiter=",")


cls_names = ["our_LR", "our_MLP", "sk_LR", "sk_MLP"]



t_stat = np.zeros((4, 4))
p_val = np.zeros((4, 4))
better = np.zeros((4, 4), dtype=bool)
stat_significant = np.zeros((4, 4), dtype=bool)
stat_better = np.zeros((4, 4), dtype=bool)

aplha = 0.05


for col in range(4):
    for row in range(4):
        if col == row:
            t_stat[row, col] = None
            p_val[row, col] = None
            better[row, col] = False
            stat_significant[row, col] = False
            stat_better[row, col] = None

        else:
            t_stat[row, col], p_val[row, col] = ttest_rel(results[:, row], results[:, col])

            if t_stat[row, col] > 0:
                better[row, col] = True
            else:
                better[row, col] = False

            if(aplha > p_val[row, col]):
                stat_significant[row, col] = True
            else:
                stat_significant[row, col] = False

            stat_better[row, col] = better[row, col] * stat_significant[row, col]
            if stat_better[row, col]:
                print(f"{cls_names[row], np.round(np.mean(results[:, row]), 3)} jest lepszy statystycznie od {cls_names[col], np.round(np.mean(results[:, col]), 3)}\n")


# print(t_stat, "\n#########\n")
# print(p_val, "\n#########\n")
# print(better, "\n#########\n")
# print(stat_significant, "\n#########\n")
# print(stat_better, "\n#########\n")

import tabulate


print("t_stat")
print(tabulate.tabulate(t_stat, tablefmt="fancy_grid"), "\n")

print("p_val")
print(tabulate.tabulate(p_val, tablefmt="fancy_grid"), "\n")

print("better")
print(tabulate.tabulate(better, tablefmt="fancy_grid"), "\n")

print("stat_significant")
print(tabulate.tabulate(stat_significant, tablefmt="fancy_grid"), "\n")

print("stat_better")
print(tabulate.tabulate(stat_better, tablefmt="fancy_grid"), "\n")

cls_results = [np.round(np.mean(results[:, row]), 3) for row in range(results.shape[1])]

cls_final = [(cls_names[i], cls_results[i]) for i in range(len(cls_results))]

print(tabulate.tabulate(cls_final, headers=["Klasyfikator", "Znaczenie statystyczne"], tablefmt="fancy_grid"))


