from math import prod

import spacy

import torch

from transformers import CLIPProcessor, CLIPModel

class ClipScorer:
    def __init__(self, model_args):
        self.device = model_args.device
        self.clip_model = CLIPModel.from_pretrained(model_args.vision_model_name).eval().to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_args.vision_model_name)

    def score(self, image_text_pair):
        image = image_text_pair["image"]
        choices = image_text_pair["choices"]

        with torch.no_grad():
            inputs = self.processor(text=choices, images=image, return_tensors="pt", padding=True)
            inputs = {name: tensor.to(self.device) for name, tensor in inputs.items()}
            outputs = self.clip_model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)

            return torch.argmax(probs) == choices.index(image_text_pair["label"])
        
class FClipScorerGM:
    def __init__(self, model_args):
        self.device = model_args.device
        self.clip_model = CLIPModel.from_pretrained(model_args.vision_model_name).eval().to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_args.vision_model_name)
        self.parser = spacy.load(model_args.text_parser)

        self.eps = 1e-7

    def geo_mean(self, iterable):
        return prod(iterable) ** (1 / len(iterable))

    def parse_noun(self, text):
        doc = self.parser(text)
        return [token.text for token in doc if token.pos_ == "NOUN"]

    def score(self, image_text_pair):
        image = image_text_pair["image"]
        choices = image_text_pair["choices"]

        choices_with_nouns = [[c] + self.parse_noun(c) for c in choices]
        choices_flatten = [item for sublist in choices_with_nouns for item in sublist]

        with torch.no_grad():
            inputs = self.processor(text=choices_flatten, images=image, return_tensors="pt", padding=True)
            inputs = {name: tensor.to(self.device) for name, tensor in inputs.items()}
            outputs = self.clip_model(**inputs)
            logits_per_image = outputs.logits_per_image.squeeze()
            
            logits_finegrained = []
            for i in range(len(choices)):
                tmp_logits = []
                choice_id = sum([len(c) for c in choices_with_nouns[:i]])
                
                for j in range(len(choices_with_nouns[i])):
                    if logits_per_image[choice_id + j] > 0:
                        tmp_logits.append(logits_per_image[choice_id + j])
                    else:
                        tmp_logits.append(self.eps)

                geo_mean_logits = self.geo_mean(tmp_logits)
                logits_finegrained.append(geo_mean_logits)

            logits_finegrained = torch.tensor(logits_finegrained).unsqueeze(0)
            probs = logits_finegrained.softmax(dim=1)

            return torch.argmax(probs) == choices.index(image_text_pair["label"])

scorers_map = {
    "clip": ClipScorer,
    "fclipgm": FClipScorerGM,
}