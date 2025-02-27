# F-CLIPScore
Finegrained CLIPScore to detect object hallucination

This repository contains the source code for the following research paper:
[Vision-Encoders (Already) Know What They See: Mitigating Object Hallucination via Simple Fine-Grained CLIPScore]()

## Environment setup

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Evaluate

For vanila CLIPScore
```bash
python eval_ohd-caps.py --score_type clip
```

For F-CLIPScore
```bash
python eval_ohd-caps.py --score_type fclipam
```

Further constrained version of F-CLIPScore is implemented in [scorers.py](scorers.py)