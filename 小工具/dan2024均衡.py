import os
import json
import orjson
import random
import logging
import tarfile
from collections import Counter, defaultdict
from rimo_storage.cache import disk_cache

from tqdm import tqdm
from PIL import Image


random.seed(1)

需要的key = ('tag_string_general', 'tag_string_artist', 'tag_string_character', 'created_at', 'rating')

全角色 = defaultdict(lambda: 0)
全标签 = defaultdict(lambda: 0)
坏图 = tqdm(desc='坏图')

输出文件夹 = 'X:/image_balance大_2024'


def 角色阈值():
    if not 全角色:
        return 10
    return min(v for k, v in Counter(全角色).most_common(500)) + 5


@disk_cache()
def 画师和角色统计():
    a画师 = []
    a角色 = []
    for n in tqdm(range(150), desc='画师和角色统计'):
        meta_d = orjson.loads(open(f"Y:/danbooru2024/_tags缓存/{n}.json").read())
        for k, v in meta_d.items():
            if v['fav_count'] < 5:
                continue
            if v['tag_string_artist']:
                a画师.append(v['tag_string_artist'])
            a角色.extend(v['tag_string_character'].split())
    return a画师, a角色


_t1, _t2 = 画师和角色统计()
要的画师 = [k for k, v in Counter(_t1).most_common(32000)]
要的角色 = [k for k, v in Counter(_t2).most_common(3000)]
print(要的画师.index('fuzichoco'), [v for k, v in Counter(_t1).most_common(32000)][-1])
print(要的角色.index('momoi_(maid)_(blue_archive)'))
with open(f'画师top32000.json', 'w', encoding='utf8') as f:
    f.write(json.dumps(要的画师, ensure_ascii=False))


os.mkdir(输出文件夹)


meta_list = []
def 写(tar, name, meta):
    tar.extract(name, path=输出文件夹)
    try:
        img = Image.open(f"{输出文件夹}/{name}")
        img.load()
    except Exception:
        logging.exception('坏！')
        坏图.update(1)
        try:
            os.remove(f"{输出文件夹}/{name}")
        except Exception:
            None
    else:
        meta_list.append({"file_name": name} | meta)
        if random.random() < 0.1:
            写meta()


def 写meta():
    with open(f'{输出文件夹}/metadata.jsonl', 'w', encoding='utf8') as f:
        for i in meta_list:
            f.write(json.dumps(i)+'\n')


def 计():
    统计 = {
        '角色most_common': Counter(全角色).most_common(1000),
        '来源': 来源,
    }
    print(统计)
    with open(f'{输出文件夹.removesuffix("/")}.json', 'w', encoding='utf8') as f:
        f.write(json.dumps(统计, ensure_ascii=False))


来源 = defaultdict(lambda: 0)
for n in range(80):
    meta_d = json.load(open(f"Y:/danbooru2024/_tags缓存/{n}.json"))
    tar = tarfile.open(name=f"Y:/danbooru2024/images/{str(n).zfill(4)}.tar")
    for i, name in enumerate(tqdm(tar.getnames(), desc=f'第{n}包')):
        if i % 10 == 0:
            _角色阈值 = 角色阈值()
        if not name.endswith('png') and not name.endswith('jpg') and not name.endswith('jpeg') and not name.endswith('.webp'):
            continue
        id = name.split('.', 2)[0]
        meta = {k: meta_d[id][k] for k in 需要的key}
        if random.random() < 0.012:
            来源['全局随机'] += 1
            写(tar, name, meta)
            continue
        cs = meta['tag_string_character'].split()
        if not meta['tag_string_artist']:
            continue
        if 'comic' in meta['tag_string_general']:
            continue
        if int(int(meta['created_at'][:4]) < 2020) and random.random() > 0.4:
            continue
        if 'boy' in meta['tag_string_general'] and random.random() > 0.6:
            continue
        if meta['tag_string_artist'] not in 要的画师:
            if random.random() < 0.8:
                continue
        if not cs:
            continue
        if any([全角色[c] > _角色阈值 for c in cs]):
            continue
        if all([c not in 要的角色 for c in cs]):
            continue
        for c in cs:
            全角色[c] += 1
        来源['角色正确'] += 1
        写(tar, name, meta)
    计()
