#!/usr/bin/env python
# coding: utf-8

# In[13]:


import torch
from torchtext.datasets import IMDB
from torch.utils.data.dataset import random_split

train_dataset = IMDB(split='train')
test_dataset = IMDB(split='test')

torch.manual_seed(0)
train_dataset, valid_dataset = random_split(list(train_dataset), [20000, 5000])


# In[14]:


import re
from collections import Counter, OrderedDict

def tokenizer(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall(
        '(?::|;|=)(?:-)?(?:\)|\(|D|P)', text.lower()
    )
    text = re.sub('[\W]+', ' ', text.lower()) +\
        ' '.join(emoticons).replace('-', '')
    tokenized = text.split()
     
    return tokenized

token_counts = Counter()
for label, line in train_dataset:
    tokens = tokenizer(line)
    token_counts.update(tokens)

len(token_counts)


# In[15]:


from torchtext.vocab import vocab
sorted_by_freq_tuples = sorted(
    token_counts.items(), key=lambda x: x[1], reverse=True
)
ordered_dict = OrderedDict(sorted_by_freq_tuples)
vocab = vocab(ordered_dict)
vocab.insert_token('<pad>', 0)
vocab.insert_token('<unk>', 1)
vocab.set_default_index(1)


# In[16]:


text_pipeline = lambda x: [vocab[token] for token in tokenizer(x)]
label_pipeline = lambda x: 1.0 if x==2 else 0.0

def collate_batch(batch):
    label_list, text_list, lengths = [], [], []
    for _label, _text in batch:
        label_list.append(label_pipeline(_label))
        processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
        text_list.append(processed_text)
        lengths.append(processed_text.size(0))
    label_list = torch.tensor(label_list)
    lengths = torch.tensor(lengths)
    padded_text_list = nn.utils.rnn.pad_sequence(text_list, batch_first=True)
    return padded_text_list, label_list, lengths


# In[17]:


from torch.utils.data import DataLoader

batch_size = 32
train_dl = DataLoader(train_dataset, batch_size, 
                      shuffle=True, collate_fn=collate_batch)
valid_dl = DataLoader(valid_dataset, batch_size, 
                      shuffle=False, collate_fn=collate_batch)
test_dl = DataLoader(test_dataset, batch_size, 
                     shuffle=False, collate_fn=collate_batch)


# In[18]:


import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, rnn_hidden_size, fc_hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, 
                                      embedding_dim=embed_dim,
                                      padding_idx=0)
        self.rnn = nn.LSTM(embed_dim, rnn_hidden_size, batch_first=True)
        self.fc1 = nn.Linear(rnn_hidden_size, fc_hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(fc_hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, text, lengths):
        out = self.embedding(text)
        out = nn.utils.rnn.pack_padded_sequence(
            out, lengths.cpu().numpy(), enforce_sorted=False, batch_first=True
        )
        out, (hidden, cell) = self.rnn(out)
        out = hidden[-1, :, :]
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)

        return out


# In[19]:


vocab_size = len(vocab)
embed_dim = 20
rnn_hidden_size = 64
fc_hidden_size = 64
torch.manual_seed(0)
model = RNN(vocab_size, embed_dim, 
            rnn_hidden_size, fc_hidden_size)
loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# In[39]:


from torch.utils.tensorboard import SummaryWriter

def train(dataloader):
    model.train()
    train_acc = 0
    for text_batch, label_batch, lengths in dataloader:
        pred = model(text_batch, lengths)[:, 0]
        loss = loss_fn(pred, label_batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        train_acc += (
            (pred >= 0.5).float() == label_batch
        ).float().sum().item()
    return train_acc / len(list(dataloader.dataset))

def evaluate(dataloader):
    model.eval()
    eval_acc = 0
    with torch.no_grad():
        for text_batch, label_batch, lengths in dataloader:
            pred = model(text_batch, lengths)[:, 0]
            eval_acc += (
                (pred >= 0.5).float() == label_batch
            ).float().sum().item()
        
    return eval_acc / len(list(dataloader.dataset))


# In[22]:


get_ipython().run_line_magic('load_ext', 'tensorboard')
get_ipython().run_line_magic('tensorboard', "--logdir='./logs'")
writer = SummaryWriter(log_dir='./logs')

torch.manual_seed(0)
num_epochs = 10
for epoch in range(num_epochs):
    train_acc = train(train_dl)
    valid_acc = evaluate(valid_dl)
    writer.add_scalars('Train/Valid Accuracy',
                      {
                          'Train Accuracy': train_acc,
                          'Valid Accuracy': valid_acc
                      }, epoch)


# In[40]:


test_acc = evaluate(test_dl)
print(f'Test Accuracy: {test_acc:.4f}')


# In[41]:


torch.save(model, 'sentiment_analysis_RNN.pth')

