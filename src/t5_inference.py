import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

MODEL_ID = "/home/ubuntu/QueryReformulationRL/models/QueryReformulationRL"

tokenizer = T5Tokenizer.from_pretrained(MODEL_ID)
input_sequence = "how to bake great cookie"
input_ids = tokenizer(input_sequence, return_tensors="pt").input_ids
nsent = 4

print(f'Input: {input_sequence}')

model = T5ForConditionalGeneration.from_pretrained(MODEL_ID)

model.eval()

with torch.no_grad():
    for i in range(nsent):
        output = model.generate(input_ids, max_length=35, num_beams=1, do_sample=True, repetition_penalty=1.8)
        target_sequence = tokenizer.decode(output[0], skip_special_tokens=True)
        print(f'Target: {target_sequence}')

