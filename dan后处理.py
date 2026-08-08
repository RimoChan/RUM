import json
import random
from torchvision import transforms
from torchvision.transforms.functional import crop
from 人物特征标签 import 人物特征标签表
from 标签处理 import 计算时间标签, 分离人数标签, rating_map


train_transforms = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)


画师top500 = json.load(open('画师.json'))
画师好 = json.load(open('好画师.json'))


def _抽取画师():
    if random.random() < 0.5:
        return random.choice(画师top500)
    else:
        return random.choice(画师好)


class dan后处理:
    def __init__(self, drop_tag_rate: float, drop_char_feature_rate: float, size: tuple[int, int], 学人rate: float):
        self.drop_tag_rate = drop_tag_rate
        self.drop_char_feature_rate = drop_char_feature_rate
        self.学人rate = 学人rate
        self.size = size

    def 计算prompt(self, d: dict):
        for k, v in [*d.items()]:
            assert len(v) == 1
            d[k] = v[0]

        学人 = (random.random() < self.学人rate) and d['tag_string_character'].split()

        原始tags = d['tag_string_general'].split()
        人标签, 剩下的标签 = 分离人数标签(原始tags)
        random.shuffle(剩下的标签)

        if 学人:
            保留标签数 = min(random.randint(1, 5), len(剩下的标签))
        else:
            保留标签数 = len(剩下的标签) * (1 - self.drop_tag_rate)
            if 保留标签数 > 15:
                保留标签数 = (15 * 1 + 保留标签数) / 2
        剩下的标签 = random.sample(剩下的标签, min(40, int(保留标签数)))
        random.shuffle(剩下的标签)

        角色标签 = d['tag_string_character'].split()
        for 角色 in 角色标签:
            for 签 in 人物特征标签表.get(角色, []):
                if random.random() < self.drop_char_feature_rate and 签 in 剩下的标签:
                    剩下的标签.remove(签)
        
        for 坏标签 in ['oekaki', 'monochrome', 'greyscale', 'greyscale_with_colored_background', 'artist_name', 'signature', 'twitter_username', 'patreon_username', 'speech_bubble', 'thought_bubble', 'limited_palette', 'sketch', 'typo', 'character_name', 'english_text', 'dated', 'copyright_name', 'signature', 'comic', 'qr_code', 'watermark']:
            if 坏标签 in 剩下的标签 and random.random() < 0.9:
                剩下的标签.remove(坏标签)

        if random.random() < 0.2:
            画师标签 = [_抽取画师(), _抽取画师()]
        else:
            画师标签 = [_抽取画师()]

        时间标签 = 计算时间标签(d['created_at'])

        rating标签 = [rating_map[d['rating']]]

        if random.random() < 0.2:
            画师标签 = []
        if random.random() < 0.5:
            时间标签 = []
        if random.random() < 0.5:
            rating标签 = []

        if 学人:
            if random.random() < 0.1:
                画师标签 = []
            时间标签 = []
            rating标签 = []

        新tags = 人标签 + 角色标签 + rating标签 + 剩下的标签 + 画师标签 + 时间标签
        新tags = [i for i in 新tags if i]
        d['prompts'] = ', '.join(新tags).replace('_', ' ')

        新tags改 = 人标签 + [f'character {i}' for i in 角色标签] + [f'rating {i}' for i in rating标签] + 剩下的标签 + [f'artist {i}' for i in 画师标签] + 时间标签
        新tags改 = [i for i in 新tags改 if i]
        d['prompts改'] = ', '.join(新tags改).replace('_', ' ')

        return {k: [v] for k, v in d.items()}

    def preprocess_train(self, examples):
        images = [image.convert("RGB") for image in examples['image']]
        original_sizes = []
        resized_sizes = []
        all_images = []
        crop_top_lefts = []
        for image in images:
            目标边长 = random.randint(*self.size)
            r = ((目标边长 * 目标边长) / (image.height * image.width)) ** 0.5
            train_resize = transforms.Resize((int(image.height*r), int(image.width*r)), interpolation=transforms.InterpolationMode.LANCZOS)
            original_sizes.append((image.height, image.width))
            image = train_resize(image)
            目标size = (image.height // 64 * 64, image.width // 64 * 64)
            train_crop = transforms.RandomCrop((image.height // 64 * 64, image.width // 64 * 64))
            y1, x1, h, w = train_crop.get_params(image, 目标size)
            resized_sizes.append(目标size)
            image = crop(image, y1, x1, h, w)
            crop_top_left = (y1, x1)
            crop_top_lefts.append(crop_top_left)
            image = train_transforms(image)
            all_images.append(image)

        examples["original_sizes"] = original_sizes
        examples["resized_sizes"] = resized_sizes
        examples["crop_top_lefts"] = crop_top_lefts
        examples["pixel_values"] = all_images
        del examples['image']
        examples = self.计算prompt(examples)
        return examples
