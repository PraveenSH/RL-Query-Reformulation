import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained("google-t5/t5-base")
model = T5ForConditionalGeneration.from_pretrained("google-t5/t5-base")

model.eval()
with torch.no_grad():
 input_ids = tokenizer("translate English to German: The house is wonderful.", return_tensors="pt").input_ids
 outputs = model.generate(input_ids, max_length=40, num_beams=5, early_stopping=True)

print(tokenizer.decode(outputs[0]))
