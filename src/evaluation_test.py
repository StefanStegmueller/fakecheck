from evaluation import precision_at_k, mean_average_precision

data = []
for i in range(1):
    data.append((i, 0.5, 1))
    data.append((i, 0.2, 0))
    data.append((i, 0.6, 0))

for k in [1,2,3]:
    p_at_k = precision_at_k(data, k)
    print('P@{}: {}'.format(k, p_at_k))

print('MAP: {}'.format(mean_average_precision(data)))
