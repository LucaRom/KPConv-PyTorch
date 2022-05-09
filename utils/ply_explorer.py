from ply import read_ply
import numpy as np

'''
0 = Unassigned  (1)
1 = ground      (2) 
2 = building    (6)
3 = noise       (7)
4 = water       (9)
5 = bridge deck (17)
6 = high noise  (18) # Cette classe est absente dans certains scénario
'''

'''
complete Las Standard list
0 = Never Classified
1 = Unassigned  
2 = Ground
3 = Low Vegetation
4 = Medium Vegetation
5 = High Vegetation
6 = Building
7 = Noise
8 = Model Key / Reserved
9 = Water
10 = Rail
11 = Road surface
12 = Overlap/Reserved
13 = Wire - Guard
14 = Wire - Conductor
15 = Transmission Tower
16 = Wire Connector
17 = Bridge Deck 
18 = High Noise  
'''

#ply_path = "/mnt/data/test/NL_StJohns_20210205_NAD83CSRS_UTMZ22_1km_E2870_N52560_CQL1_CLASS.ply"

ply_path = "/mnt/data/input_0.020/E3290_N52720.ply"


my_ply = read_ply(ply_path, triangular_mesh=False)

# Noms des colonnes
my_ply.dtype.names

#Nombre de points par fichier
#classification = my_ply['scalar_Classification']
classification = my_ply['class']

#label_counts_tuple = classification.unique(return_counts=True)
label_counts_tuple = np.unique(classification, return_counts=True)

#print(list(zip(label_counts_tuple[0].numpy(), label_counts_tuple[1].numpy())))
print(list(zip(label_counts_tuple[0], label_counts_tuple[1])))

#print(my_ply)
print("debug")

