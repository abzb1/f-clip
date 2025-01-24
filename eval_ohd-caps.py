import argparse
from math import prod

import spacy
from tqdm.auto import tqdm
from loguru import logger

import torch
from transformers import CLIPProcessor, CLIPModel
from datasets import load_dataset

def geo_mean(iterable):
    return prod(iterable) ** (1 / len(iterable))

def parse_noun(parser, text):
    doc = parser(text)
    return [token.text for token in doc if token.pos_ == "NOUN"]

def load_data(dataname):
    return load_dataset("doolayer/OHD-Caps", dataname)["test"]

def evaluate_model(image_text_pairs, model, processor, parser):
    right_vanila = [False] * len(image_text_pairs)
    right_finegrained = [False] * len(image_text_pairs)

    for id, d in enumerate(tqdm(image_text_pairs, dynamic_ncols=True)):
        image = d["image"]
        choices = d["choices"]

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

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--vision_model_name", type=str, default="openai/clip-vit-base-patch32")
    args.add_argument("--text_parser", type=str, default="en_core_web_sm")
    args = args.parse_args()

    model = CLIPModel.from_pretrained(args.vision_model_name).eval().cuda()
    processor = CLIPProcessor.from_pretrained(args.vision_model_name)
    spacy_parser = spacy.load(args.text_parser)

    datasets = ["coco", "flickr30k", "nocaps"]
    for dataname in datasets:
        image_text_pairs = load_data(dataname)
        acc, finegrained_acc = evaluate_model(image_text_pairs, model, processor, spacy_parser)
        logger.info(f"{dataname} Accuracy: {acc*100:.2f}%")
        logger.info(f"{dataname} Fine-grained Accuracy: {finegrained_acc*100:.2f}%")