import importlib
import sys
sys.path.append('/zfs/projects/students/ltdarc-usf-intern-2025/code')

import Validity_Functions

importlib.reload(Validity_Functions)

data_path='/zfs/projects/students/ltdarc-usf-intern-2025/data'

index = Validity_Functions.make_index(data_path)

print("Index created, number of rows:", len(index))

print(Validity_Functions.check_index(index))