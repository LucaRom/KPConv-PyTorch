from http.client import METHOD_NOT_ALLOWED

from matplotlib import matplotlib_fname
from ply import read_ply
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import confusion_matrix, classification_report, f1_score, ConfusionMatrixDisplay, jaccard_score, multilabel_confusion_matrix

# Chemin vers les fichier ply
#ply_path = "/mnt/Data/00_Donnees/02_Codes/Github/KPConv-PyTorch/test/Log_2022-06-01_03-19-50/predictions/E2870_N52560.ply"
ply_path = "./test/Log_2022-06-01_03-19-50/predictions"

list_ply = [x for x in os.listdir(ply_path) if x.endswith(".ply")]
#print(list_ply)

#target_labels = ['unclassified', 'ground', 'building', 'water', 'bridge deck']
target_labels_num = [0, 1, 2, 3, 4, 5, 6] # stjohns
#target_labels_num = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] # NB

# Creation du fichier txt
#outfile = open('iou_manu.txt', 'w')
outfile = open('omi_manu.txt', 'w')

# Debut de la boucle pour les fichier
#list_ply = list_ply[:1] # list of list with one element (debug)

#for ply in list_ply:
for ply in list_ply:
    full_ply_path = os.path.join(ply_path, ply)
    my_ply = read_ply(full_ply_path, triangular_mesh=False)
    
    true = my_ply['targets']
    preds = my_ply['preds']

    preds_labels = np.unique(preds) # Labels de la prediction

    # Calcul IoU
    #jaccard_values = jaccard_score(true, preds, average=None, zero_division=0)

    # Calcul confusion matrix
    mcm = multilabel_confusion_matrix(true, preds)

    # Loop pour ajouter les labels non-present
    iter_num = 0
    adjusted_values_lst = []
    adjusted_mcm_lst = []
    omission_lst = []
    for class_num in target_labels_num:
        if iter_num < len(preds_labels):
            if class_num == preds_labels[iter_num]:
                # jaccard iou
                #adjusted_values_lst.append(jaccard_values[iter_num])

                # manual iou from cm
                tn, fp, fn, tp = mcm[iter_num].ravel() # attention a l'ordre
                print('tn, fp, fn, tp', tn, fp, fn, tp)
                class_iou = tp / (tp+fn+fp)
                adjusted_mcm_lst.append(class_iou)

                # omission from cm
                if fn == 0: # In case of division par zero
                    class_omi = 0
                else : 
                    class_omi = fn / (fn + tp)
                omission_lst.append("%.2f"%class_omi) # append and round to 2 decimals

                iter_num += 1
            elif class_num not in preds_labels: # Extra safety for debug
                adjusted_values_lst.append(0)
                adjusted_mcm_lst.append(0)
                omission_lst.append(0)
            else:
                print("This is bad...")
        else:
            adjusted_values_lst.append(0)
            adjusted_mcm_lst.append(0)
            omission_lst.append(0)

    #adjusted_values_lst.insert(0, ply) # Add name of file in front
    omission_lst.insert(0, ply) # Add name of file in front
    #outfile.write(','.join(str(j) for j in adjusted_values_lst) + '\n')
    outfile.write(','.join(str(j) for j in omission_lst) + '\n')

    #print("Classification pour le fichier :", ply)
    #print(classification_report(true, preds, target_names=target_labels))
    print("Labels for this class : ", preds_labels)
    #print("IoU ajuste : ", adjusted_values_lst)
    print("New IoU : ", adjusted_mcm_lst)
    print("List of omission : ", omission_lst)

#outfile.close()
print("fin")
