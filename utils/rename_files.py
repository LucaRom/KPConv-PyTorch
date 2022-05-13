# importing os module
# Quick rename fonction for st_johns files
import os
 
# Give the folder path 
folder = "/mnt/data/original_las/train"

#list des fichiers
print(os.listdir(folder))

# for count, filename in enumerate(os.listdir(folder)):
#     src = f"{folder}/{filename}"
#     new_name =f"{folder}/{filename[41:-15]}" + '.las'

#     #print("old name : ", src, " New name : ", new_name)
#     os.rename(src, new_name)

 