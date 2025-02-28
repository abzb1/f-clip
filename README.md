# F-CLIPScore
Finegrained CLIPScore to detect object hallucination

This repository contains the source code for the following research paper:
[Vision-Encoders (Already) Know What They See: Mitigating Object Hallucination via Simple Fine-Grained CLIPScore](https://arxiv.org/pdf/2502.20034)

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

cf. Further constrained version of F-CLIPScore is implemented in [scorers.py](scorers.py)

## Result
![image](https://github.com/user-attachments/assets/15d9258f-8dc6-4d87-9816-d21d1974ca58)

## Citation
```
@misc{oh2025visionencodersalreadyknowsee,
      title={Vision-Encoders (Already) Know What They See: Mitigating Object Hallucination via Simple Fine-Grained CLIPScore}, 
      author={Hongseok Oh and Wonseok Hwang},
      year={2025},
      eprint={2502.20034},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2502.20034}, 
}
```
