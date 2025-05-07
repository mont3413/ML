#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch.nn as nn
import torch
import numpy as np
import matplotlib.pyplot as plt


# In[24]:


class DCGAN(): # only for 28x28 images
    def __init__(self,
                 device,
                 gen_input_size,
                 n_filters):

        self.device = device

        self.gen_model = self.make_generator_network(gen_input_size,
                                                     n_filters).to(self.device)

        self.disc_model = self.make_discriminator_network(n_filters).to(self.device)

        self.loss_fn = nn.BCELoss()
        self.g_optimizer = torch.optim.Adam(self.gen_model.parameters(), lr=0.001)
        self.d_optimizer = torch.optim.Adam(self.disc_model.parameters(), lr=0.001)

        self.all_d_losses = []
        self.all_g_losses = []
        self.all_d_real = []
        self.all_d_fake = []

    def make_generator_network(self,
                               input_size,
                               n_filters
                              ):
        
        model = nn.Sequential(
            nn.ConvTranspose2d(in_channels=input_size, out_channels=n_filters*4,
                               kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=n_filters*4),
            nn.LeakyReLU(),
            
            nn.ConvTranspose2d(in_channels=n_filters*4, out_channels=n_filters*2,
                               kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=n_filters*2),
            nn.LeakyReLU(),
            
            nn.ConvTranspose2d(in_channels=n_filters*2, out_channels=n_filters,
                               kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=n_filters),
            nn.LeakyReLU(),
            
            nn.ConvTranspose2d(in_channels=n_filters, out_channels=1,
                               kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

        return model

        

    def make_discriminator_network(self,
                                   n_filters
                                  ):

        model = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=n_filters,
                      kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=n_filters),
            nn.LeakyReLU(),
            
            nn.Conv2d(in_channels=n_filters, out_channels=n_filters*2,
                      kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=n_filters*2),
            nn.LeakyReLU(),
            
            nn.Conv2d(in_channels=n_filters*2, out_channels=n_filters*4,
                      kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=n_filters*4),
            nn.LeakyReLU(),

            nn.Conv2d(in_channels=n_filters*4, out_channels=1,
                      kernel_size=4, stride=1, padding=0, bias=False), 
            nn.Sigmoid()           
        )

        return model

    def create_z(self, batch_size, z_size, mode_z='normal'):
        if mode_z == 'uniform':
            input_z = torch.rand(batch_size, z_size, 1, 1)*2 - 1
        elif mode_z == 'normal':
            input_z = torch.randn(batch_size, z_size, 1, 1)
        return input_z

    def d_train(self, x):
        self.disc_model.zero_grad()

        batch_size = x.size(0)

        x = x.to(self.device)
        d_labels_real = torch.ones(batch_size, 1, device=self.device)
        d_prob_real = self.disc_model(x).view(-1, 1).squeeze(0)
        d_loss_real = self.loss_fn(d_prob_real, d_labels_real)

        input_z = self.create_z(batch_size, z_size, mode_z='normal').to(self.device)
        gen_output = self.gen_model(input_z)
        d_labels_fake = torch.zeros(batch_size, 1, device=self.device)
        d_prob_fake = self.disc_model(gen_output).view(-1, 1).squeeze(0)
        d_loss_fake = self.loss_fn(d_prob_fake,d_labels_fake)

        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        self.d_optimizer.step()

        return d_loss.data.item(), d_prob_real.detach(), d_prob_fake.detach()

    def g_train(self, x):
        self.gen_model.zero_grad()

        batch_size = x.size(0)

        input_z = self.create_z(batch_size, z_size, mode_z='normal').to(self.device)
        gen_output = self.gen_model(input_z)
        g_labels_real = torch.ones(batch_size, 1, device=self.device)
        d_prob_fake = self.disc_model(gen_output).view(-1, 1).squeeze(0)

        g_loss = self.loss_fn(d_prob_fake, g_labels_real)
        g_loss.backward()
        self.g_optimizer.step()

        return g_loss.data.item()

    def create_samples(self, input_z, image_size=(28,28,1)):
        batch_size = input_z.size(0)
        gen_output = self.gen_model(input_z)
        images = torch.reshape(gen_output, (batch_size, *image_size))
        return ((images+1)/2.0).detach().cpu().numpy()

    def train(self, num_epochs, log, dataloader, verbose=True):

        for epoch in range(1, num_epochs+1):
            d_losses, g_losses = [], []
            d_vals_real, d_vals_fake = [], []
            for i, (x, _) in enumerate(dataloader): # x - image; _ - label
                d_loss, d_prob_real, d_prob_fake = self.d_train(x)
                g_loss = self.g_train(x)
                d_losses.append(d_loss)
                g_losses.append(g_loss)
                d_vals_real.append(d_prob_real.mean())
                d_vals_fake.append(d_prob_fake.mean())

            self.all_d_losses.append(torch.tensor(d_losses).mean())
            self.all_g_losses.append(torch.tensor(g_losses).mean())
            self.all_d_real.append(torch.tensor(d_vals_real).mean())
            self.all_d_fake.append(torch.tensor(d_vals_fake).mean())

            if verbose and epoch % log == 0:
                print(f'Epoch {epoch}:')
                print(f'  G/D Loss: {self.all_g_losses[-1]:.4f}/{self.all_d_losses[-1]:.4f}')
                print(f'  D-Real/D-Fake probs: {self.all_d_real[-1]:.4f}/{self.all_d_fake[-1]:.4f}\n')


    def plot_loss_curve(self, ax):
        assert self.all_d_losses, 'Model not trained'
        
        ax.plot(self.all_g_losses, label='Generator losses')
        ax.plot(self.all_d_losses, label='Discriminator losses')
        ax.legend()
        ax.grid()
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Loss')

    def plot_disc_values_curve(self, ax):
        assert self.all_d_losses, 'Model not trained'

        ax.plot(self.all_d_real, label='Real: D(x)')
        ax.plot(self.all_d_fake, label='Fake: D(G(z))')
        ax.legend()
        ax.grid()
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Discriminator output')

    def show_generated_samples(self, generated_images, fig):
        num_images = generated_images.shape[0]
        n_rows = np.ceil(num_images / 5).astype(int)
        for i in range(num_images):
            ax = fig.add_subplot(n_rows, 5, i+1)
            ax.set_xticks([])
            ax.set_yticks([])
            image = generated_images[i]
            ax.imshow(image, cmap='Grays')


# In[25]:


if torch.cuda.is_available():
  device = torch.device('cuda:0')
elif torch.backends.mps.is_available():
    device = torch.device('mps:0')
else:
  device = 'cpu'

print(device)


# In[26]:


image_size = (28, 28, 1)
z_size = 100
n_filters = 32


# In[27]:


torch.manual_seed(0)
model = DCGAN(device,
              gen_input_size=z_size,
              n_filters=n_filters
              )


# In[28]:


print(model.gen_model)


# In[29]:


print(model.disc_model)


# In[30]:


import torchvision
from torchvision import transforms
image_path = './'
transform = transforms.Compose([
    transforms.ToTensor(), # [0, 255] --> [0, 1]
    transforms.Normalize(mean=(0.5), std=(0.5)) # [0, 1] --> [-1, 1]
])

fashion_mnist_dataset = torchvision.datasets.FashionMNIST(
    root=image_path, train=True,
    transform=transform, download=True
)


# ## Real Images

# In[31]:


fig, ax = plt.subplots(nrows=2, ncols=5,
                      sharex=True, sharey=True,
                      figsize=(15,6))

ax = ax.flatten()

for i in range(10):
  image = fashion_mnist_dataset[i][0].reshape(*image_size)
  ax[i].imshow((image+1)/2.0, cmap='Grays')


# In[32]:


from torch.utils.data import DataLoader
batch_size = 64
fashion_mnist_dl = DataLoader(fashion_mnist_dataset, batch_size,
                              shuffle=True, drop_last=True)


# In[33]:


num_epochs = 100
log = 10
model.train(num_epochs, log, dataloader=fashion_mnist_dl)


# In[34]:


fig = plt.figure(figsize=(15, 6))
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)
model.plot_loss_curve(ax1)
model.plot_disc_values_curve(ax2)


# ## Generated Images

# In[35]:


fig = plt.figure(figsize=(7, 9))
num_samples = 30
torch.manual_seed(5)
input_z = model.create_z(batch_size=num_samples, z_size=z_size, mode_z='normal').to(device)
generated_images = model.create_samples(input_z, image_size)
model.show_generated_samples(generated_images, fig)

