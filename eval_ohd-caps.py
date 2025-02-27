import json
import asyncio
import aiohttp
import argparse
from io import BytesIO

from PIL import Image
from tqdm.auto import tqdm
from loguru import logger

from scorers import ClipScorer, scorers_map

def load_data(dataname):
    url_label = f"clip_hallucination/data/OHD-Caps/test/{dataname}_annotations.json"
    url_out = f"clip_hallucination/data/OHD-Caps/test/{dataname}_out_label.json"

    datas = {}
    with open(url_label, 'r') as f:    
        for line in f:
            if line:
                d = json.loads(line)
                datas[d["file_path"]] = {"img_url": d["url"]}

    with open(url_out, 'r') as f:
        for line in f:
            d = json.loads(line)
            datas[d["image"]]["choices"] = d["caption"]
            datas[d["image"]]["label"] = d["label"]

    async def fetch_image(session, url):
        async with session.get(url) as response:
            if response.status == 200:
                return Image.open(BytesIO(await response.read())).convert("RGB")
            else:
                return None

    async def process_images(datas):
        async with aiohttp.ClientSession() as session:
            tasks = {k: fetch_image(session, v["img_url"]) for k, v in datas.items()}
            results = await asyncio.gather(*tasks.values())
            
            for (k, v), img in zip(datas.items(), results):
                if img is not None:
                    v["image"] = img

    asyncio.run(process_images(datas))

    return datas

def evaluate_model(image_text_pairs, scorer):
    right = [False] * len(image_text_pairs)
    
    for id, d in enumerate(tqdm(image_text_pairs.values(), dynamic_ncols=True)):
        right[id] = scorer.score(d)

    return sum(right)/len(right)

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--vision_model_name", type=str, default="openai/clip-vit-large-patch14")
    args.add_argument("--device", type=str, default="cuda")
    args.add_argument("--text_parser", type=str, default="en_core_web_sm")
    args.add_argument("--score_type", type=str, default="fclipam")
    args.add_argument("--var_alpha", type=float, default=2e-1)
    args.add_argument("--min_gamma", type=float, default=0.3)
    args = args.parse_args()

    scorer = scorers_map.get(args.score_type, ClipScorer)(args)

    logger.info(f"Evaluating model {scorer}")
    logger.info(f"args: {args}")

    datasets = ["coco"] # flickr30k, nocaps images are not available due to the license, you can download them from the official website
    for dataname in datasets:
        image_text_pairs = load_data(dataname)
        acc = evaluate_model(image_text_pairs, scorer)
        logger.info(f"{dataname} Accuracy: {acc*100:.2f}%")