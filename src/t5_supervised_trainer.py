import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import T5ForConditionalGeneration, T5Tokenizer, AdamW, get_linear_schedule_with_warmup


loss_output = open("/home/ubuntu/QueryReformulationRL/losses.txt", 'w')
save_path = "/home/ubuntu/QueryReformulationRL/models/pretrained.pt"
max_seq_len = 35
log_steps = 100
save_steps = 5000
batch_size = 4
num_epochs = 2

# Load the dataset from a TSV file
def load_dataset(file_path):
    df = pd.read_csv(file_path, sep='\t')
    results = df[['input', 'target']].values.tolist()
    return results

# Define the dataset class
class Seq2SeqDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        input_text, target_text = self.data[index]
        input_ids = self.tokenizer(input_text, padding='max_length', max_length=max_seq_len, truncation=True, return_tensors='pt').input_ids
        target_ids = self.tokenizer(target_text, padding='max_length', max_length=max_seq_len, truncation=True, return_tensors='pt').input_ids
        return {'input_ids': input_ids[0], 'target_ids': target_ids[0]}


# Define the training function
def train(model, dataloader, optimizer, scheduler, device, epoch_num):
    model.train()
    total_loss = 0
    steps = 0
    for batch in dataloader:

        input_ids = batch['input_ids'].to(device)
        target_ids = batch['target_ids'].to(device)
        labels = target_ids.to(device)
        labels[labels == tokenizer.pad_token_id] = -100
        outputs = model(input_ids, labels=labels)

        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        steps += 1

        if ((steps%log_steps) == 0):
            print(steps, total_loss/(steps*len(input_ids)))

        if ((steps%save_steps) == 0):
             eval_avg_loss = evaluate(model, eval_dataloader, device)
             print(f'Evaluation Loss: {eval_avg_loss:.4f}')
             torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, save_path+"_"+str(epoch_num)+"_"+str(steps))
             loss_output.write(str(eval_avg_loss)+"\n")


    avg_loss = total_loss / len(dataloader)
    return avg_loss

# Define the evaluation function
def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            target_ids = batch['target_ids'].to(device)
            labels = target_ids.to(device)
            labels[labels == tokenizer.pad_token_id] = -100
            outputs = model(input_ids, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    return avg_loss


if __name__ == "__main__":

    # Load the dataset
    data = load_dataset('/home/ubuntu/QueryReformulationRL/data/diamond.tsv')

    train_index = int(0.97*len(data))
    train_data = data[:train_index]
    eval_data = data[train_index:]

    #model architecture google-t5
    MODEL_ID = "google-t5/t5-base"
    tokenizer = T5Tokenizer.from_pretrained(MODEL_ID)
    train_dataset = Seq2SeqDataset(train_data, tokenizer)
    eval_dataset = Seq2SeqDataset(eval_data, tokenizer)

    # Create the model
    model = T5ForConditionalGeneration.from_pretrained(MODEL_ID)

    # Define the optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=5e-5)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=2000, num_training_steps=len(train_dataset) // 2)

    # Define the training parameters
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create the DataLoader
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=True)

    # Train the model
    for epoch in range(num_epochs):
         train(model, train_dataloader, optimizer, scheduler, device, epoch)
