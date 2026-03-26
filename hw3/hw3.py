from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from TAM.tam import TAM
from TAM.qwen_utils import process_vision_info
from pathlib import Path

import os


def tam_for_qwen25(image_path, prompt, save_dir='vis_results'):
    os.makedirs(save_dir, exist_ok=True)
    
    model_name = "Qwen/Qwen2.5-VL-3B-Instruct"

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )

    processor = AutoProcessor.from_pretrained(model_name)

    messages = [{"role": "user", "content": [{"type": "image", "image": image_path}, {"type": "text", "text": prompt}]}]
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    inputs = inputs.to(model.device)
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        use_cache=True,
        output_hidden_states=True,
        return_dict_in_generate=True
    )
    
    generated_ids = outputs.sequences
    logits = [model.lm_head(feats[-1]) for feats in outputs.hidden_states]

    special_ids = {'img_id': [151652, 151653],
                   'prompt_id': [151653, [151645, 198, 151644, 77091]], 
                   'answer_id': [[198, 151644, 77091, 198], -1]}
    
    vision_shape = (inputs['image_grid_thw'][0, 1] // 2, inputs['image_grid_thw'][0, 2] // 2)
    vis_inputs = [[video_inputs[0][i] for i in range(0, len(video_inputs[0]))]] if isinstance(image_path, list) else image_inputs
    
    
    raw_map_records = []
    for i in range(len(logits)):
        img_map = TAM(
            generated_ids[0].cpu().tolist(),
            vision_shape,
            logits,
            special_ids,
            vis_inputs,
            processor,
            os.path.join(save_dir, str(i) + '.jpg'),
            i,
            raw_map_records,
            False)
        
if __name__ == '__main__':
    prompt = 'Describe the picture, your limit is 40-50 words'
    image_path = 'TAM/imgs/satellite_data'
    
    img = Path(image_path) / "image.png"

    tam_for_qwen25(str(img), prompt, save_dir="imgs/vis_res")
