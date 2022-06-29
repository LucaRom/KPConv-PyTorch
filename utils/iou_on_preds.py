from http.client import METHOD_NOT_ALLOWED

from matplotlib import matplotlib_fname
from ply import read_ply
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, f1_score, ConfusionMatrixDisplay, jaccard_score, multilabel_confusion_matrix


#ply_path = "/mnt/Data/00_Donnees/02_Codes/Github/KPConv-PyTorch/test/Log_2022-06-01_03-19-50/predictions/E2870_N52560.ply"
ply_path = "D:/00_Donnees/02_Codes/Github/KPConv-PyTorch/test/Log_2022-06-01_03-19-50/predictions/NL_StJohns_20210205_NAD83CSRS_UTMZ22_1km_E3140_N52960_CQL1_CLASS.ply"
my_ply = read_ply(ply_path, triangular_mesh=False)

true = my_ply['targets']
preds = my_ply['preds']

target_labels = ['unclassified', 'ground', 'building', 'water', 'bridge deck']

unique = np.unique(true)
print(unique)

#print(classification_report(true, preds, target_names=target_labels))
#print(jaccard_score(true, preds, average=None))

#cm = confusion_matrix(true, preds)
cmm = multilabel_confusion_matrix(true, preds)
#disp = ConfusionMatrixDisplay.from_predictions(true, preds, display_labels=target_labels, values_format="d")
cm = cmm[0]
cmm2 = cmm[0]

TN = cm[0][0]
FN = cm[1][0]
TP = cm[1][1]
FP = cm[0][1]

mtn = cmm2[0][0]
mfn = cmm2[1][0]
mtp = cmm2[1][1]
mfp = cmm2[0][1]

# Calculate prediction / omission
omission = mfn / (mfn + mtp)
print(omission)
print(mfn, mtp)


# print(cmm)
# print(cmm[0])
# print(TN, FN, TP, FP)
# print(mtn, mfn, mtp, mfp)
# #print(TN, FN, TP, FP)

#disp.plot()
#plt.show()