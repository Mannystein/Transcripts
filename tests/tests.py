import numpy as np
import symbolix as sb
from tqdm import tqdm

i = 0
print("Testing...")
for i in tqdm(range(10)):
    arr1 = np.random.random(10000)
    arr2 = np.random.random(10000)
    dv1 = sb.delay_vectors(arr1,1,6)
    symbols1 = sb.find_symbols(arr1,1,6)
    trs = sb.get_transcripts(arr1,arr2,1,6)
    ocs = sb.ocs_from_symbol_series(trs)
    JS = sb.JS_C(arr1,arr2,1,6)
    SKL = sb.SKL_C(arr1,arr2,1,6)

p = sb.circulance(arr1,None)
print("")
print(p)
print("")
print("All tests successful!")




