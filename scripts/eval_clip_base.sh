python eval_ohd-caps.py \
    --vision_model_name "openai/clip-vit-base-patch32" \
    --text_parser "en_core_web_sm" \
    --coco_path "/data1/home/ohs/workspace/vlmhallu/clip_hallucination/data/OHD-Caps/test/coco_annotations.json" \
    --flickr30k_path "/data1/home/ohs/workspace/vlmhallu/clip_hallucination/data/OHD-Caps/test/flickr_annotations.json" \
    --nocaps_path "/data1/home/ohs/workspace/vlmhallu/clip_hallucination/data/OHD-Caps/test/nocaps_annotations.json" \
    --coco_img_path "/data1/home/ohs/dataset/coco/val2014" \
    --flickr30k_img_path "/data1/home/ohs/dataset/flickr30k-images" \
    --nocaps_img_path "/data1/home/ohs/dataset/nocaps_v0.1/validation"