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
            inputs = self.processor(text=choices, images=image, return_tensors="pt", padding=True, truncation=True)
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
            inputs = self.processor(text=choices_flatten, images=image, return_tensors="pt", padding=True, truncation=True)
            inputs = {name: tensor.to(self.device) for name, tensor in inputs.items()}
            outputs = self.clip_model(**inputs)
            logits_per_image = outputs.logits_per_image.squeeze()
            
            logits_finegrained = []
            for i in range(len(choices)):
                tmp_logits = []
                choice_id = sum([len(c) for c in choices_with_nouns[:i]])
                
                for j in range(len(choices_with_nouns[i])):
                    if logits_per_image[choice_id + j] > 0:
                        tmp_logits.append(logits_per_image[choice_id + j].item())
                    else:
                        tmp_logits.append(self.eps)

                geo_mean_logits = self.geo_mean(tmp_logits)
                logits_finegrained.append(geo_mean_logits)

            logits_finegrained = torch.tensor(logits_finegrained).unsqueeze(0)
            probs = logits_finegrained.softmax(dim=1)

            return torch.argmax(probs) == choices.index(image_text_pair["label"])

class FClipScorerGMVAR:
    def __init__(self, model_args):
        self.device = model_args.device
        self.clip_model = CLIPModel.from_pretrained(model_args.vision_model_name).eval().to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_args.vision_model_name)
        self.parser = spacy.load(model_args.text_parser)

        self.alpha = model_args.var_alpha
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
            inputs = self.processor(text=choices_flatten, images=image, return_tensors="pt", padding=True, truncation=True)
            inputs = {name: tensor.to(self.device) for name, tensor in inputs.items()}
            outputs = self.clip_model(**inputs)
            logits_per_image = outputs.logits_per_image.squeeze()
            
            logits_finegrained = []
            for i in range(len(choices)):
                tmp_logits = []
                choice_id = sum([len(c) for c in choices_with_nouns[:i]])
                
                for j in range(len(choices_with_nouns[i])):
                    if logits_per_image[choice_id + j] > 0:
                        tmp_logits.append(logits_per_image[choice_id + j].item())
                    else:
                        tmp_logits.append(self.eps)

                geo_mean_logits = self.geo_mean(tmp_logits)
                tmp_logits = tmp_logits[1:]
                var = torch.var(torch.tensor(tmp_logits)) if tmp_logits else torch.tensor(0)
                var = torch.exp(-self.alpha * var)
                logits_finegrained.append(geo_mean_logits * var)

            logits_finegrained = torch.tensor(logits_finegrained).unsqueeze(0)
            probs = logits_finegrained.softmax(dim=1)

            return torch.argmax(probs) == choices.index(image_text_pair["label"])
        
class FClipScorerGMMIN1:
    def __init__(self, model_args):
        self.device = model_args.device
        self.clip_model = CLIPModel.from_pretrained(model_args.vision_model_name).eval().to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_args.vision_model_name)
        self.parser = spacy.load(model_args.text_parser)

        self.gamma = model_args.min_gamma
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
            inputs = self.processor(text=choices_flatten, images=image, return_tensors="pt", padding=True, truncation=True)
            inputs = {name: tensor.to(self.device) for name, tensor in inputs.items()}
            outputs = self.clip_model(**inputs)
            logits_per_image = outputs.logits_per_image.squeeze()
            
            logits_finegrained = []
            for i in range(len(choices)):
                tmp_logits = []
                choice_id = sum([len(c) for c in choices_with_nouns[:i]])
                
                for j in range(len(choices_with_nouns[i])):
                    if logits_per_image[choice_id + j] > 0:
                        tmp_logits.append(logits_per_image[choice_id + j].item())
                    else:
                        tmp_logits.append(self.eps)

                geo_mean_logits = self.geo_mean(tmp_logits)
                tmp_logits = tmp_logits[1:]
                min_logit = torch.tensor(min(tmp_logits)) if tmp_logits else torch.tensor(tmp_logits[0])
                logits_finegrained.append(self.gamma*geo_mean_logits * (1-self.gamma)*min_logit)

            logits_finegrained = torch.tensor(logits_finegrained).unsqueeze(0)
            probs = logits_finegrained.softmax(dim=1)

            return torch.argmax(probs) == choices.index(image_text_pair["label"])

class FClipScorerGMMIN2:
    def __init__(self, model_args):
        self.device = model_args.device
        self.clip_model = CLIPModel.from_pretrained(model_args.vision_model_name).eval().to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_args.vision_model_name)
        self.parser = spacy.load(model_args.text_parser)

        self.gamma = model_args.min_gamma
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
            inputs = self.processor(text=choices_flatten, images=image, return_tensors="pt", padding=True, truncation=True)
            inputs = {name: tensor.to(self.device) for name, tensor in inputs.items()}
            outputs = self.clip_model(**inputs)
            logits_per_image = outputs.logits_per_image.squeeze()
            
            logits_finegrained = []
            for i in range(len(choices)):
                tmp_logits = []
                choice_id = sum([len(c) for c in choices_with_nouns[:i]])
                
                for j in range(len(choices_with_nouns[i])):
                    if logits_per_image[choice_id + j] > 0:
                        tmp_logits.append(logits_per_image[choice_id + j].item())
                    else:
                        tmp_logits.append(self.eps)

                _tmp_logits = tmp_logits[1:]
                if _tmp_logits:
                    min_logit = torch.tensor(min(_tmp_logits))
                    tmp_logits.append(min_logit)
                    logits_finegrained.append(self.geo_mean(tmp_logits))
                else:
                    logits_finegrained.append(self.geo_mean(tmp_logits))

            logits_finegrained = torch.tensor(logits_finegrained).unsqueeze(0)
            probs = logits_finegrained.softmax(dim=1)

            return torch.argmax(probs) == choices.index(image_text_pair["label"])

class FClipScorerAM:
    def __init__(self, model_args):
        self.device = model_args.device
        self.clip_model = CLIPModel.from_pretrained(model_args.vision_model_name).eval().to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_args.vision_model_name)
        self.parser = spacy.load(model_args.text_parser)

        self.eps = 1e-7

    def arith_mean(self, iterable):
        return sum(iterable) / len(iterable)

    def parse_noun(self, text):
        doc = self.parser(text)
        return [token.text for token in doc if token.pos_ == "NOUN"]

    def score(self, image_text_pair):
        image = image_text_pair["image"]
        choices = image_text_pair["choices"]

        choices_with_nouns = [[c] + self.parse_noun(c) for c in choices]
        choices_flatten = [item for sublist in choices_with_nouns for item in sublist]

        with torch.no_grad():
            inputs = self.processor(text=choices_flatten, images=image, return_tensors="pt", padding=True, truncation=True)
            inputs = {name: tensor.to(self.device) for name, tensor in inputs.items()}
            outputs = self.clip_model(**inputs)
            logits_per_image = outputs.logits_per_image.squeeze()
            
            logits_finegrained = []
            for i in range(len(choices)):
                tmp_logits = []
                choice_id = sum([len(c) for c in choices_with_nouns[:i]])
                
                for j in range(len(choices_with_nouns[i])):
                    if logits_per_image[choice_id + j] > 0:
                        tmp_logits.append(logits_per_image[choice_id + j].item())
                    else:
                        tmp_logits.append(self.eps)

                logits_finegrained.append(self.arith_mean(tmp_logits))

            logits_finegrained = torch.tensor(logits_finegrained).unsqueeze(0)
            probs = logits_finegrained.softmax(dim=1)

            return torch.argmax(probs) == choices.index(image_text_pair["label"])
        
class FClipScorerAMVAR:
    def __init__(self, model_args):
        self.device = model_args.device
        self.clip_model = CLIPModel.from_pretrained(model_args.vision_model_name).eval().to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_args.vision_model_name)
        self.parser = spacy.load(model_args.text_parser)

        self.alpha = model_args.var_alpha
        self.eps = 1e-7

    def arith_mean(self, iterable):
        return sum(iterable) / len(iterable)

    def parse_noun(self, text):
        doc = self.parser(text)
        return [token.text for token in doc if token.pos_ == "NOUN"]

    def score(self, image_text_pair):
        image = image_text_pair["image"]
        choices = image_text_pair["choices"]

        choices_with_nouns = [[c] + self.parse_noun(c) for c in choices]
        choices_flatten = [item for sublist in choices_with_nouns for item in sublist]

        with torch.no_grad():
            inputs = self.processor(text=choices_flatten, images=image, return_tensors="pt", padding=True, truncation=True)
            inputs = {name: tensor.to(self.device) for name, tensor in inputs.items()}
            outputs = self.clip_model(**inputs)
            logits_per_image = outputs.logits_per_image.squeeze()
            
            logits_finegrained = []
            for i in range(len(choices)):
                tmp_logits = []
                choice_id = sum([len(c) for c in choices_with_nouns[:i]])
                
                for j in range(len(choices_with_nouns[i])):
                    if logits_per_image[choice_id + j] > 0:
                        tmp_logits.append(logits_per_image[choice_id + j].item())
                    else:
                        tmp_logits.append(self.eps)

                arith_mean = self.arith_mean(tmp_logits)

                tmp_logits = tmp_logits[1:]
                var = torch.var(torch.tensor(tmp_logits)) if tmp_logits else torch.tensor(0)
                var = torch.exp(-self.alpha * var)
                logits_finegrained.append(arith_mean * var)

            logits_finegrained = torch.tensor(logits_finegrained).unsqueeze(0)
            probs = logits_finegrained.softmax(dim=1)

            return torch.argmax(probs) == choices.index(image_text_pair["label"])

scorers_map = {
    "clip": ClipScorer,
    "fclipgm": FClipScorerGM,
    "fclipgmvar": FClipScorerGMVAR,
    "fclipgmm1": FClipScorerGMMIN1,
    "fclipgmm2": FClipScorerGMMIN2,
    "fclipam": FClipScorerAM,
    "fclipamvar": FClipScorerAMVAR
}