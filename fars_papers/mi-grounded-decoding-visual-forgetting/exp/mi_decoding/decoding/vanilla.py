# Standard greedy decoding for Qwen2.5-VL models.
# generate_vanilla() takes model+processor+(image,question), returns generated text.
import torch
from mi_decoding.models.load_model import prepare_inputs


def generate_vanilla(model, processor, image, question, max_new_tokens=1000,
                     system_prompt=None):
    inputs = prepare_inputs(processor, image, question, model_device=model.device,
                            system_prompt=system_prompt)
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )

    generated_ids = output_ids[0, input_len:]
    generated_text = processor.decode(generated_ids, skip_special_tokens=True)
    return generated_text
