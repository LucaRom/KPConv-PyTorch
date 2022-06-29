from ply import read_ply
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, f1_score, ConfusionMatrixDisplay, jaccard_score
import os
import numpy as np
import laspy

#ply_path = "/mnt/Data/00_Donnees/02_Codes/Github/KPConv-PyTorch/test/Log_2022-06-01_03-19-50/predictions/E2870_N52560.ply"
las_path = "../raw_nb/original_las"
list_las = [x for x in os.listdir(las_path) if x.endswith(".las")]

print(las_path)
print(list_las)

#target_labels = ['unclassified', 'ground', 'building', 'water', 'bridge deck']
#target_labels_num = [0, 1, 2, 3, 4, 5, 6]

#outfile = open('iou_manu.txt', 'w')

for las in list_las:
    full_las_path = os.path.join(las_path, las)
    my_las = laspy.read(full_las_path)

    true = my_las.classification
    #preds = my_ply['preds']

    true_labels = np.unique(true)

    #iter_num = 0
    #adjusted_values_lst = []
    #for class_num in target_labels_num:
    #    if iter_num < len(preds_labels):
    #        if class_num == preds_labels[iter_num]:
    #            adjusted_values_lst.append(jaccard_values[iter_num])
    #            iter_num += 1
    #        elif class_num not in preds_labels: # Extra safety for debug
    #            adjusted_values_lst.append(0)
    #        else:
    #            print("This is bad...")
    #    else:
    #        adjusted_values_lst.append(0)

    #adjusted_values_lst.insert(0, ply) # Add name of file in front
    #outfile.write(ply + ','.join(str(j) for j in adjusted_values_lst + '\n'))
    #outfile.write(','.join(str(j) for j in adjusted_values_lst) + '\n')

    print("Classes pour le fichier :", las)
    #print(classification_report(true, preds, target_names=target_labels))
    print("Labels for this class : ", true_labels)
    #print("IoU ajuste : ", adjusted_values_lst)

#outfile.close()
print("fin")
#cm = confusion_matrix(true, preds)
#disp = ConfusionMatrixDisplay.from_predictions(true, preds, display_labels=target_labels, values_format="d")

#disp.plot()
#plt.show()
