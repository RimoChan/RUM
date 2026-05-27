from typing import Iterable
from collections import Counter

import torch
from torch import Tensor


def _组batch(bb: list[dict]):
    大b = {}
    for b in bb:
        for k, v in b.items():
            大b.setdefault(k, []).append(v)
    融合大b = {}
    for k, v in 大b.items():
        if isinstance(v[0], bool):
            融合大b[k] = v[0]
        else:
            融合大b[k] = torch.cat(v)
            # print(融合大b[k].shape)
    return 融合大b


def arb(it: Iterable[dict], batch_size: int) -> Iterable[dict]:
    所有桶 = {}
    for i, b in enumerate(it):
        if i > 20000:   # 防内存泄漏
            所有桶 = {}
        b = {k: v for k, v in b.items() if isinstance(v, bool | Tensor)}
        a = []
        for k in sorted(b.keys()):
            v = b[k]
            if isinstance(v, Tensor):
                shape = ','.join(map(str, v.shape))
                a.append(f'{k}={shape}')
            else:
                a.append(f'{k}={v}')
        特征 = '_'.join(a)
        所有桶.setdefault(特征, []).append(b)
        if len(所有桶[特征]) == batch_size:
            yield _组batch(所有桶.pop(特征))


if __name__ == '__main__':
    import random
    import pickle
    from pathlib import Path
    def 源():
        while True:
            a = [*Path(r's:/RUM缓存_超').glob('*.pkl')]
            random.shuffle(a)
            for i in a:
                with open(i, 'rb') as f:
                    batch = pickle.load(f)
                yield batch
    tt = []
    for i, b in enumerate(arb(源(), batch_size=2)):
        tt.append(b['学nega'])
        if i % 10 == 0:
            print('外', i, Counter(tt))
