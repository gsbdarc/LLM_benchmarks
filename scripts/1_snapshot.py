import importlib
import sys
import helper

importlib.reload(helper)

data_path='/zfs/projects/students/ltdarc-usf-intern-2025/data'

index = helper.make_index(data_path)

print("Index created, number of rows:", len(index))

print(helper.check_index(index))