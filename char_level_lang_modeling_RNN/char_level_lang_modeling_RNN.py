#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
with open('1268-0.txt') as fp: # THE MYSTERIOUS ISLAND by Jules Verne
    text = fp.read()

start_idx = text.find('THE MYSTERIOUS ISLAND')
end_idx = text.find('End of the Project Gutenberg')
text = text[start_idx:end_idx]
char_set = set(text)
print(f'Text length: {len(text)}')
print(f'Unique characters: {len(char_set)}')


# In[2]:


chars_sorted = sorted(char_set)
char2int = {ch:i for i,ch in enumerate(chars_sorted)}
char_array = np.array(chars_sorted)
text_encoded = np.array(
    [char2int[ch] for ch in text],
    dtype=np.int32
)


# In[3]:


import torch
from torch.utils.data import Dataset
seq_length = 40
chunk_size = seq_length + 1
text_chunks = [text_encoded[i:i+chunk_size] 
               for i in range(len(text_encoded)-chunk_size)]

class TextDataset(Dataset):
    def __init__(self, text_chunks):
        self.text_chunks = text_chunks

    def __len__(self):
        return len(self.text_chunks)

    def __getitem__(self, idx):
        text_chunk = torch.tensor(self.text_chunks[idx])
        return text_chunk[:-1].long(), text_chunk[1:].long()

seq_dataset = TextDataset(text_chunks)


# In[4]:


from torch.utils.data import DataLoader
batch_size = 64
torch.manual_seed(0)
seq_dl = DataLoader(seq_dataset, batch_size,
                    shuffle=True, drop_last=True)


# In[56]:


import torch.nn as nn
class RNN(nn.Module):
    def __init__(self, device, vocab_size, embed_dim,
                 rnn_hidden_size):
        super().__init__()
        self.device = device
        self.embedding = nn.Embedding(vocab_size, embed_dim).to(device)
        self.rnn_hidden_size = rnn_hidden_size
        self.rnn = nn.LSTM(embed_dim, rnn_hidden_size, batch_first=True).to(device)
        self.fc = nn.Linear(rnn_hidden_size, vocab_size).to(device)

    def forward(self, x, hidden, cell):
        out = self.embedding(x).unsqueeze(1)
        out, (hidden, cell) = self.rnn(out, (hidden, cell))
        out = self.fc(out).reshape(out.size(0), -1)
        return out, hidden, cell

    def init_hidden(self, batch_size):
        hidden = torch.zeros(1, batch_size, self.rnn_hidden_size).to(self.device)
        cell = torch.zeros(1, batch_size, self.rnn_hidden_size).to(self.device)
        return hidden, cell        


# In[57]:


if torch.cuda.is_available():
    device = torch.device('cuda:0')
elif torch.backends.mps.is_available():
    device = torch.device('mps:0')
else:
    device = torch.device('cpu')

print(device)


# In[58]:


vocab_size = len(char_array)
embed_dim = 256
rnn_hidden_size = 512
torch.manual_seed(0)
model = RNN(device, vocab_size, embed_dim, rnn_hidden_size)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# In[59]:


num_epochs = 10000
torch.manual_seed(0)
model.train()
for epoch in range(1, num_epochs+1):
    hidden, cell = model.init_hidden(batch_size)
    seq_batch, target_batch = next(iter(seq_dl))
    seq_batch = seq_batch.to(device)
    target_batch = target_batch.to(device)
    optimizer.zero_grad()
    loss = 0
    for c in range(seq_length):
        pred, hidden, cell = model(seq_batch[:, c], hidden, cell)
        loss += loss_fn(pred, target_batch[:, c])
    loss.backward()
    optimizer.step()
    loss = loss.item() / seq_length
    if epoch % 500 == 0:
        print(f'Epoch {epoch} Loss {loss:.4f}')


# In[71]:


from torch.distributions.categorical import Categorical
def sample(model, starting_str,
           len_generated_text=500,
           scale_factor=1.0):
    encoded_input = torch.tensor(
        [char2int[ch] for ch in starting_str]
    ).reshape(1, -1).to(device)
    generated_str = starting_str

    model.eval()
    hidden, cell = model.init_hidden(1)
    for c in range(len(starting_str) - 1):
        _, hidde, cell = model(encoded_input[:, c], hidden, cell)

    last_char = encoded_input[:, -1]
    for i in range(len_generated_text):
        logits, hidden, cell = model(
            last_char.view(1), hidden, cell
        )
        logits = torch.squeeze(logits, 0)
        scaled_logits = logits * scale_factor
        m = Categorical(logits=scaled_logits)
        last_char = m.sample().to(device)
        generated_str += str(char_array[last_char])

    return generated_str


# #### Generated text, scale_factor = 1.0

# In[89]:


torch.manual_seed(0)
print(sample(model, starting_str=' ', scale_factor=1.0))


# #### Generated text, scale_factor = 0.5 (more random generation)

# In[90]:


print(sample(model, starting_str=' ', scale_factor=0.5))


# #### Generated text, scale_factor = 5.0 (more deterministic generation)

# In[ ]:


print(sample(model, starting_str=' ', scale_factor=5.0))


# In[88]:


torch.save(model, 'char_level_lang_modeling_RNN.pth')

