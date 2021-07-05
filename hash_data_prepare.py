import numpy as np
import hash_utils

np.random.seed(0)
n_query = 10000
data, query, gallery = hash_utils.get_cifar10_datasets('data', n_query)

with open('data/hash_train.py', 'w') as f:
    np.save(f, data)

with open('data/hash_query.py', 'w') as f:
    np.save(f, query)

with open('data/hash_gallery.py', 'w') as f:
    np.save(f, gallery)
