import time, numpy as np, pandas as pd
from app import predict

df = pd.read_csv("test.csv")
queries = df["query_text"].sample(200, random_state=0).tolist()
predict(queries[0])
times = []
for q in queries:
    s = time.perf_counter()
    predict(q)
    times.append((time.perf_counter() - s) * 1000)
print("Average latency: %.1f ms | median %.1f ms | p95 %.1f ms" % (np.mean(times), np.median(times), np.percentile(times, 95)))
