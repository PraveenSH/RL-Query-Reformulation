import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

MODEL_ID = "prhegde/t5-query-reformulation-RL"

tokenizer = T5Tokenizer.from_pretrained(MODEL_ID)
model = T5ForConditionalGeneration.from_pretrained(MODEL_ID)
model.eval()

input_sequence = "how to bake great cookie"
input_ids = tokenizer(input_sequence, return_tensors="pt").input_ids
print(f'Input: {input_sequence}')

nsent = 4
with torch.no_grad():
    for i in range(nsent):
        output = model.generate(input_ids, max_length=35, num_beams=1, do_sample=True, repetition_penalty=1.8)
        target_sequence = tokenizer.decode(output[0], skip_special_tokens=True)
        print(f'Target: {target_sequence}')

