import pdal
import os


path = 'F:/00_Donnees_SSD/02_CCCOT/results_stjohns_random/Log_2022-06-01_03-19-50/predictions'

list_ply = [x[:-4] for x in os.listdir(path) if x.endswith('.ply')]
#list_ply = [list_ply[0]] # temp for tests

# ply to las
for ply in list_ply:
    ply_file = os.path.join(path, ply + '.ply')
    las_file = os.path.join(path, ply + '.las')

    pipeline = pdal.Reader.ply(filename=ply_file).pipeline()
    
    print(pipeline.execute())
    # print(pipeline.arrays)

    preds_array = pipeline.arrays[0]["preds"]

    pipeline |= pdal.Filter.assign()

    pipeline |= pdal.Writer.las(filename=las_file, forward='all', extra_dims='all').pipeline()
    print(pipeline.execute())