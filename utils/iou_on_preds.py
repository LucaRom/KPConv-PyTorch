from ply import read_ply
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, f1_score, ConfusionMatrixDisplay, jaccard_score


ply_path = "/mnt/Data/00_Donnees/02_Codes/Github/KPConv-PyTorch/test/Log_2022-06-01_03-19-50/predictions/E2870_N52560.ply"
my_ply = read_ply(ply_path, triangular_mesh=False)

true = my_ply['targets']
preds = my_ply['preds']

target_labels = ['unclassified', 'ground', 'building', 'water', 'bridge deck']

print(classification_report(true, preds, target_names=target_labels))
print(jaccard_score(true, preds, average=None))

#cm = confusion_matrix(true, preds)
#disp = ConfusionMatrixDisplay.from_predictions(true, preds, display_labels=target_labels, values_format="d")

#disp.plot()
#plt.show()