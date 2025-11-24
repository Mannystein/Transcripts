import numpy as np
import transcripts as tr

test1 = np.random.random(10)
test2 = np.random.random(10)
test3 = np.random.random(10)
dv1 = tr.delay_vectors(test1,1,3)
symbs2 = tr.find_symbols(test1,1,dim=3)
ocs2 = tr.ocs_from_symbol_series(symbs2)


print(test1)
print(dv1)
print(symbs2)
print(ocs2)
