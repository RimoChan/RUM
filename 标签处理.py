import re
import random
from datetime import datetime


rating_map = {
    's': 'sensitive',
    'g': 'general',
    'q': 'questionable',
    'e': 'explicit',
}


def 人数标签(s: str) -> bool:
    if s in ('1girl', '1boy'):
        return True
    return bool(re.fullmatch('[0-9]\+?(girl|boy)s$', s))


def 分离人数标签(原始tags: list[str]) -> tuple[list[str], list[str]]:
    人 = [i for i in 原始tags if 人数标签(i)]
    剩下的 = [i for i in 原始tags if not 人数标签(i)]
    return 人, 剩下的


def 计算时间标签(iso_time: str) -> list[str]:
    year = datetime.fromisoformat(iso_time).year
    year_tag = 'newest'
    for t, y in [('oldest', 2017), ('old', 2019), ('modern', 2020), ('recent', 2022)]:
        if year <= y:
            year_tag = t
            break
    t = random.random()
    if t < 0.7:
        return [year_tag]
    elif t < 0.8:
        return [f'year {year}']
    elif t < 0.9:
        return [year_tag, f'year {year}']
    else:
        return [f'year {year}', year_tag]
