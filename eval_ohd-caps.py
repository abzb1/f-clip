import argparse

from tqdm.auto import tqdm
from loguru import logger

from datasets import load_dataset

from scorers import ClipScorer, scorers_map

def load_data(dataname):
    return load_dataset("doolayer/OHD-Caps", dataname)["test"]

def evaluate_model(image_text_pairs, scorer):
    right = [False] * len(image_text_pairs)
    
    for id, d in enumerate(tqdm(image_text_pairs, dynamic_ncols=True)):
        right[id] = scorer.score(d)

    return sum(right)/len(right)

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--vision_model_name", type=str, default="openai/clip-vit-large-patch14")
    args.add_argument("--device", type=str, default="cuda")
    args.add_argument("--text_parser", type=str, default="en_core_web_sm")
    args.add_argument("--score_type", type=str, default="fclipgm")
    args = args.parse_args()

    scorer = scorers_map.get(args.score_type, ClipScorer)(args)

    datasets = ["coco", "flickr30k", "nocaps"]
    for dataname in datasets:
        image_text_pairs = load_data(dataname)
        acc = evaluate_model(image_text_pairs, scorer)
        logger.info(f"{dataname} Accuracy: {acc*100:.2f}%")