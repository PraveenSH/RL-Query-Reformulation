import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Load the model and tokenizer
torch.cuda.empty_cache()


MODEL_ID = "google-t5/t5-base"
tokenizer = T5Tokenizer.from_pretrained(MODEL_ID)
input_sequence = "A medical term that means nerve root disease is"
input_ids = tokenizer(input_sequence, return_tensors="pt").input_ids
nsent = 4

print(f'Input: {input_sequence}')

model = T5ForConditionalGeneration.from_pretrained(MODEL_ID)
checkpoint = torch.load("/home/ubuntu/QueryReformulationRL/models/pretrained.pt_1_45000")
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

with torch.no_grad():
    for i in range(nsent):
        output = model.generate(input_ids, max_length=35, num_beams=1, do_sample=True)
        target_sequence = tokenizer.decode(output[0], skip_special_tokens=True)
        print(f'Target: {target_sequence}')

print("**************************")
print("**************************")

exit()
model = T5ForConditionalGeneration.from_pretrained(MODEL_ID)
checkpoint = torch.load("/home/ubuntu/QueryReformulationRL/models/reward_trained.pt_0_1000")
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

with torch.no_grad():
    for i in range(nsent):
        output = model.generate(input_ids, max_length=35, num_beams=1, do_sample=True)
        target_sequence = tokenizer.decode(output[0], skip_special_tokens=True)
        print(f'Target: {target_sequence}')


