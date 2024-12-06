import os
import json
from math import prod

from PIL import Image
from tqdm.auto import tqdm
from matplotlib import pyplot as plt

import spacy
import torch
from transformers import CLIPProcessor, CLIPModel

def geo_mean(iterable):
    return prod(iterable) ** (1 / len(iterable))

def parse_noun(parser, text):
    doc = parser(text)
    return [token.text for token in doc if token.pos_ == "NOUN"]

ohd_test_coco = "/data1/home/ohs/workspace/vlmhallu/clip_hallucination/data/OHD-Caps/test/coco_annotations.json"
test_data = []
with open(ohd_test_coco, "r") as f:
    for line in f:
        test_data.append(json.loads(line))

image_text_pairs = []
for d in tqdm(test_data):
    data_tmp = {}

    img_path = d["file_path"]
    img_path = os.path.join("/data1/home/ohs/dataset/coco/val2014", img_path)
    img = Image.open(img_path)
    data_tmp["image"] = img

    choices = []
    choices.append(d["positive_sample"])
    for v in d["adversarial_samples"].values():
        choices.append(v)
    
    data_tmp["text"] = choices
    image_text_pairs.append(data_tmp)

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval().cuda()
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

spacy_parser = spacy.load("en_core_web_sm")

right_vanila = [False] * len(image_text_pairs)
right_finegrained = [False] * len(image_text_pairs)

for id, d in enumerate(tqdm(image_text_pairs)):
    image = d["image"]
    choices = d["text"]

    choices_with_nouns = []
    for c in choices:
        choices_with_nouns.append([c]+parse_noun(spacy_parser, c))
    choices_flatten = [item for sublist in choices_with_nouns for item in sublist]

    with torch.no_grad():
        inputs = processor(text=choices_flatten, images=image, return_tensors="pt", padding=True)
        inputs = {name: tensor.cuda() for name, tensor in inputs.items()}
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image.squeeze().tolist()
        
        logits_per_choice = []
        logits_finegrained = []
        for i in range(len(choices)):
            tmp_logits = []
            choice_id = sum([len(c) for c in choices_with_nouns[:i]])
            logits_per_choice.append(logits_per_image[choice_id])
            for j in range(len(choices_with_nouns[i])):
                tmp_logits.append(logits_per_image[choice_id+j])

            geo_mean_logits = geo_mean(tmp_logits)
            logits_finegrained.append(geo_mean_logits)
        
        logits_per_choice = torch.tensor(logits_per_choice).unsqueeze(0)
        logits_finegrained = torch.tensor(logits_finegrained).unsqueeze(0)
        probs = logits_per_choice.softmax(dim=1)
        probs_finegrained = logits_finegrained.softmax(dim=1)

        if torch.argmax(probs) == 0:
            right_vanila[id] = True
        if torch.argmax(probs_finegrained) == 0:
            right_finegrained[id] = True

print(f"Accuracy: {sum(right_vanila) / len(right_vanila)}")
print(f"Fine-grained Accuracy: {sum(right_finegrained) / len(right_finegrained)}")