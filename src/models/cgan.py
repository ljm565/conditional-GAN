import torch
import torch.nn as nn




class Generator(nn.Module):
    def __init__(self, config):
        super(Generator, self).__init__()
        self.height = config.height
        self.width = config.width
        self.class_num = config.class_num
        self.label_dim = config.label_dim
        self.hidden_dim = config.hidden_dim
        self.noise_init_size = config.noise_init_size
        self.color_channel = config.color_channel

        self.emb = nn.Embedding(self.class_num, self.label_dim)
        self.generator = nn.Sequential(
            nn.Linear(self.noise_init_size + self.label_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.width*self.height*self.color_channel),
            nn.Sigmoid()
        )


    def forward(self, x, y):
        batch_size = x.size(0)
        x = torch.cat((x, self.emb(y)), dim=1)
        output = self.generator(x)
        output = output.view(batch_size, -1, self.height, self.width)
        return output



class Discriminator(nn.Module):
    def __init__(self, config):
        super(Discriminator, self).__init__()
        self.height = config.height
        self.width = config.width
        self.class_num = config.class_num
        self.label_dim = config.label_dim
        self.hidden_dim = config.hidden_dim
        self.color_channel = config.color_channel

        self.emb = nn.Embedding(self.class_num, self.label_dim)
        self.discriminator = nn.Sequential(
            nn.Linear(self.width*self.height*self.color_channel + self.label_dim, self.hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(self.hidden_dim, int(self.hidden_dim/4)),
            nn.LeakyReLU(0.2),
            nn.Linear(int(self.hidden_dim/4), 1),
            nn.Sigmoid()
        )


    def forward(self, x, y):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        x = torch.cat((x, self.emb(y)), dim=1)
        output = self.discriminator(x)
        return output