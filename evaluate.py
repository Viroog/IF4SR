import numpy as np


# top10&top20
def get_ndcg_hit(total_rec_score):
    ndcg10, hit10, ndcg20, hit20 = [], [], [], []
    for rec_score in total_rec_score:
        prediction = (-rec_score).argsort(1).argsort(1)
        # target的排名
        prediction = prediction[:, 0]

        for rank in prediction:
            if rank < 10:
                ndcg10.append(1 / np.log2(rank + 2))
                hit10.append(1)
            else:
                ndcg10.append(0)
                hit10.append(0)

            if rank < 20:
                ndcg20.append(1 / np.log2(rank + 2))
                hit20.append(1)
            else:
                ndcg20.append(0)
                hit20.append(0)

    return np.mean(ndcg10), np.mean(hit10), np.mean(ndcg20), np.mean(hit20)
