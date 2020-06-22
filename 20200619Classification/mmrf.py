import sys
import numpy as np
import pandas as pd 
from cvae import cvae 
from sklearn.cluster import KMeans 

#x = np.random.rand(10,25)
x = pd.read_csv("../data/salmonE74cDNA_counts_baseline.csv", index_col=0)
x = x.T
x = x + 1
x = x.apply(np.log2)
x = x.values 


inSize = 25973

x = np.random.randn(100,20000)
embedder = cvae.CompressionVAE(x,
        dim_latent = 2,
        cells_encoder=[inSize, int(inSize/2), int(inSize/4), int(inSize/8), int(inSize/16), int(inSize/32),
            int(inSize/64), int(inSize/128), int(inSize/256), int(inSize/512), int(inSize/1024), int(inSize/2048),
            int(inSize/4096),2],
        batch_size=1,
        batch_size_test=1,
        )
embedder.train(learning_rate=1e-4,
               num_steps=2,)


z = embedder.embed(x)
print(z.shape)
for ij in z:
    print(ij)
sys.exit()


kmeans = KMeans(n_clusters=4, random_state=0).fit(z)
embedder.visualize(z, labels=[int(label) for label in kmeans.labels_], filename="t.png")
