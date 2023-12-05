from pathlib import Path
from functools import partial

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T, utils
import torch.nn.functional as F
from models.imagen_pytorch import t5
from torch.nn.utils.rnn import pad_sequence

from PIL import Image

from datasets.utils.file_utils import get_datasets_user_agent
import io
import urllib

from models.imagen_pytorch.t5 import t5_encode_text, get_encoded_dim, DEFAULT_T5_NAME
import xml.etree.ElementTree as ET
import os
import json

USER_AGENT = get_datasets_user_agent()

# helpers functions

def exists(val):
    return val is not None

def cycle(dl):
    while True:
        for data in dl:
            yield data

def convert_image_to(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image

# dataset, dataloader, collator

class Collator:
    def __init__(self, image_size, url_label, text_label, image_label, name, channels):
        self.url_label = url_label
        self.text_label = text_label
        self.image_label = image_label
        self.download = url_label is not None
        self.name = name
        self.channels = channels
        self.transform = T.Compose([
            T.Resize(image_size),
            T.CenterCrop(image_size),
            T.ToTensor(),
        ])
    def __call__(self, batch):

        texts = []
        images = []
        for item in batch:
            try:
                if self.download:
                    image = self.fetch_single_image(item[self.url_label])
                else:
                    image = item[self.image_label]
                image = self.transform(image.convert(self.channels))
            except:
                continue

            text = t5.t5_encode_text([item[self.text_label]], name=self.name)
            texts.append(torch.squeeze(text))
            images.append(image)

        if len(texts) == 0:
            return None
        
        texts = pad_sequence(texts, True)

        newbatch = []
        for i in range(len(texts)):
            newbatch.append((images[i], texts[i]))

        return torch.utils.data.dataloader.default_collate(newbatch)

    def fetch_single_image(self, image_url, timeout=1):
        try:
            request = urllib.request.Request(
                image_url,
                data=None,
                headers={"user-agent": USER_AGENT},
            )
            with urllib.request.urlopen(request, timeout=timeout) as req:
                image = Image.open(io.BytesIO(req.read())).convert('RGB')
        except Exception:
            image = None
        return image

def get_report(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    indication = ''
    findings = ''
    impression = ''
    for text in root.find('MedlineCitation').find('Article').find('Abstract').findall('AbstractText'):
        if text.attrib['Label'] == 'FINDINGS':
            findings = text.text if text.text is not None else ''
        elif text.attrib['Label'] == 'IMPRESSION':
            impression = text.text if text.text is not None else ''
        elif text.attrib['Label'] == 'INDICATION':
            indication = text.text if text.text is not None else ''
        report = indication + ' ' + findings + ' ' + impression
    return report

class NLMCXRDataset(Dataset):
    def __init__(
        self,
        image_path,
        text_path,
        image_size,
        mode = 'train', #val, test
        convert_image_to_type = None,
        text_encoder_name = DEFAULT_T5_NAME
    ):
        super().__init__()
        self.image_size = image_size

        self.encode_text = partial(t5_encode_text, name = text_encoder_name, return_attn_mask = True)

        self.image_path = Path(image_path)
        self.text_path = Path(text_path)

        with open(f"/home/guo/git/Unsupervised-Structured-Reporting-via-Cycle-Consistency/src/data/radrestruct/{mode}_ids.json", "r") as f:
            self.ids = json.load(f)

        self.pairs = []
        with open("/home/guo/git/Unsupervised-Structured-Reporting-via-Cycle-Consistency/src/data/radrestruct/id_to_img_mapping_frontal_reports.json", "r") as f:
            id_to_img_mapping = json.load(f)
            for id in self.ids:
                for img in id_to_img_mapping[id]:
                    report = get_report(self.text_path / f'{id}.xml')
                    self.pairs.append((img, report))


        convert_fn = partial(convert_image_to, convert_image_to_type) if exists(convert_image_to_type) else nn.Identity()

        self.transform = T.Compose([
            T.Lambda(convert_fn),
            T.Resize(image_size),
            T.RandomHorizontalFlip(),
            T.CenterCrop(image_size),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        img, report = self.pairs[index]
        #open a grayscale image
        img = Image.open(self.image_path / f'{img}.png').convert('L')
        text_embed, text_mask = self.encode_text(report)
        return self.transform(img), text_embed, text_mask

def get_images_dataloader(
    folder,
    *,
    batch_size,
    image_size,
    shuffle = True,
    cycle_dl = False,
    pin_memory = True
):
    ds = Dataset(folder, image_size)
    dl = DataLoader(ds, batch_size = batch_size, shuffle = shuffle, pin_memory = pin_memory)

    if cycle_dl:
        dl = cycle(dl)
    return dl
