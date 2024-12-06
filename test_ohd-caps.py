import os
import json
from math import prod

from PIL import Image
from tqdm.auto import tqdm

import spacy
import torch
from transformers import CLIPProcessor, CLIPModel

def geo_mean(iterable):
    return prod(iterable) ** (1 / len(iterable))

def parse_noun(parser, text):
    doc = parser(text)
    return [token.text for token in doc if token.pos_ == "NOUN"]

def load_data(file_path, img_base_path):
    """Load and process annotation data into image-text pairs."""
    data = []
    with open(file_path, "r") as f:
        for line in f:
            d = json.loads(line)
            data_tmp = {}

            img_path = os.path.join(img_base_path, d["file_path"])
            img = Image.open(img_path)
            data_tmp["image"] = img

            choices = [d["positive_sample"]] + list(d["adversarial_samples"].values())
            data_tmp["text"] = choices
            data.append(data_tmp)
    return data

def evaluate_model(image_text_pairs, model, processor, parser):
    """Evaluate the model on given image-text pairs."""
    right_vanila = [False] * len(image_text_pairs)
    right_finegrained = [False] * len(image_text_pairs)

    for id, d in enumerate(tqdm(image_text_pairs)):
        image = d["image"]
        choices = d["text"]

        # Prepare text inputs with nouns
        choices_with_nouns = [[c] + parse_noun(parser, c) for c in choices]
        choices_flatten = [item for sublist in choices_with_nouns for item in sublist]

        with torch.no_grad():
            inputs = processor(text=choices_flatten, images=image, return_tensors="pt", padding=True)
            inputs = {name: tensor.cuda() for name, tensor in inputs.items()}
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image.squeeze().tolist()

            # Compute logits per choice
            logits_per_choice = []
            logits_finegrained = []
            for i in range(len(choices)):
                tmp_logits = []
                choice_id = sum([len(c) for c in choices_with_nouns[:i]])
                logits_per_choice.append(logits_per_image[choice_id])
                for j in range(len(choices_with_nouns[i])):
                    tmp_logits.append(logits_per_image[choice_id + j])

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

    return sum(right_vanila) / len(right_vanila), sum(right_finegrained) / len(right_finegrained)

# Load models and parsers
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval().cuda()
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
spacy_parser = spacy.load("en_core_web_sm")

# Load datasets
coco_path = "/data1/home/ohs/workspace/vlmhallu/clip_hallucination/data/OHD-Caps/test/coco_annotations.json"
flickr_path = "/data1/home/ohs/workspace/vlmhallu/clip_hallucination/data/OHD-Caps/test/flickr_annotations.json"

coco_base_path = "/data1/home/ohs/dataset/coco/val2014"
flickr_base_path = "/data1/home/ohs/dataset/flickr30k-images"

image_text_pairs_coco = load_data(coco_path, coco_base_path)
image_text_pairs_flickr30k = load_data(flickr_path, flickr_base_path)

# Evaluate datasets
coco_acc, coco_finegrained_acc = evaluate_model(image_text_pairs_coco, model, processor, spacy_parser)
print(f"COCO Accuracy: {coco_acc}")
print(f"COCO Fine-grained Accuracy: {coco_finegrained_acc}")

flickr_acc, flickr_finegrained_acc = evaluate_model(image_text_pairs_flickr30k, model, processor, spacy_parser)
print(f"Flickr30k Accuracy: {flickr_acc}")
print(f"Flickr30k Fine-grained Accuracy: {flickr_finegrained_acc}")