import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import T5ForConditionalGeneration, T5Tokenizer, AdamW, get_linear_schedule_with_warmup
from torch import nn
from sentence_transformers import SentenceTransformer
from reward_model import similarity_reward, paraphrase_reward
from nltk.corpus import stopwords
import torch.nn.functional as F


#Define constants
embed_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
loss_output = open("/home/ubuntu/QueryReformulationRL/losses.txt", 'w')
save_path = "/home/ubuntu/QueryReformulationRL/models/reward_trained.pt"
max_seq_len = 35
batch_size = 10
num_traj = 5

# Load the dataset from a TSV file
def load_dataset(file_path):
    df = pd.read_csv(file_path, sep='\t')
    results = df['input'].values.tolist()
    return results



def reward_function(input_seq, output_seq):
    sim_reward = similarity_reward(input_seq, output_seq, embed_model)

    
    stop_words = set(stopwords.words('english'))
    par_reward = paraphrase_reward(input_seq, output_seq, stop_words, embed_model)

    reward = [0.1 * sr + 0.9 * pr for sr, pr in zip(sim_reward, par_reward)]
    return reward


class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]



class ReinforceLoss(nn.Module):

  def __init__(self, model, reward_function, tokenizer):
    super().__init__()
    self.model = model
    self.reward_function = reward_function
    self.tokenizer = tokenizer

  def forward(self, input_sequence):
    # Get the model's output logits for the input and output sequences
    input_ids = self.tokenizer(input_sequence, padding='max_length', max_length=max_seq_len, truncation=True, return_tensors='pt').input_ids
    decoded_input_sequence = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)

    final_loss = 0.0
    for i in range(num_traj):
        output_ids = self.model.generate(input_ids, max_length=max_seq_len, num_beams=1, top_k=0, do_sample=True)
        decoded_output_sequence = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)

        reward = self.reward_function(decoded_input_sequence, decoded_output_sequence)
        reward_tensor = torch.tensor(reward).view(-1, 1)

        logits = self.model(input_ids=input_ids, labels=output_ids).logits
        probs = F.softmax(logits, dim=-1)
        log_max_probs, _ = torch.max(torch.log(probs), dim=-1)
        loss = -torch.mul(log_max_probs, reward_tensor)
        final_loss += loss.mean()
  
    return final_loss / num_traj


def train_with_reward(model, data_loader, optimizer, loss_fn):
    model.train()
    num_epochs = 2
    for epoch in range(num_epochs):
        steps = 0
        for batch_input_sequence in data_loader:
            optimizer.zero_grad()

            try:
                loss = loss_fn(batch_input_sequence)
                loss.backward()
                optimizer.step()
            except Exception as e:
                print(e)

            steps += 1
            if (steps%100) == 0:
                print(steps, loss)

            if (steps%500) == 0:
                torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, save_path+"_"+str(epoch)+"_"+str(steps))


if __name__ == "__main__":

    input_sequences = load_dataset("/home/ubuntu/QueryReformulationRL/data/diamond.tsv")

    MODEL_ID = "google-t5/t5-base"
    tokenizer = T5Tokenizer.from_pretrained(MODEL_ID)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_ID)

    # initialize model with a pretrained checkpoing 
    checkpoint = torch.load("/home/ubuntu/QueryReformulationRL/models/pretrained.pt_1_45000")
    model.load_state_dict(checkpoint['model_state_dict'])

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    loss_fn = ReinforceLoss(model, reward_function, tokenizer)

    dataset = MyDataset(input_sequences)
    data_loader = DataLoader(dataset, batch_size=5, shuffle=True)

    train_with_reward(model, data_loader, optimizer, loss_fn)
