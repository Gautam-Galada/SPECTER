def aggregation(m1, m2):
    weights1 = m1.get_weights()
    weights2 = m2.get_weights()
    try:
        if len(weights1) == len(weights2):
            weights_sum = [w1 + w2 for w1, w2 in zip(weights1, weights2)]
            aggregated = weights_sum
    except:
        raise TypeError("Weights are of different size")
    return aggregated


