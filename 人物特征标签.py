from waifu_sensor.v3 import _人阵, _人均值阵, 人标签


人物特征标签表: dict[str, list[str]] = {}
目标特征 = [均 for 人, 均 in zip(_人阵, _人均值阵, strict=True)]
for 人, 均 in zip(_人阵, _人均值阵, strict=True):
    人物特征标签表[人.replace(' ', '_')] = [k.replace(' ', '_') for k, v in zip(人标签, 均) if v > 0.5 and ('background' not in k)]
