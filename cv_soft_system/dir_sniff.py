import sys
import os

database_path = str(sys.argv[1])
database_folders = os.listdir(sys.argv[1])

list_file = []
list_id = []

#print (database_folders)

if len(database_folders) == 0:
    print ("empty folder")
    sys.exit()

for i in range(0, len(database_folders)):
# Get pictures inside person folder
    files = os.listdir(database_path + database_folders[i] + "/") 
    list_file.extend(files)
    for j in range(0, len(files)):
        list_id.append(i)



    
