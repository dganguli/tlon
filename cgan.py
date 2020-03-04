import os

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.utils import save_image


class Generator(nn.Module):
    def __init__(self, latent_dim, img_shape):
        super(Generator, self).__init__()

        # set to 10 for 10 digits
        self.label_emb = nn.Embedding(num_embeddings=10, embedding_dim=10)
        self.img_shape = img_shape

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim + 10, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        # Concatenate label embedding and image to produce input
        gen_input = torch.cat((self.label_emb(labels), noise), -1)
        img = self.model(gen_input)
        img = img.view(img.size(0), *self.img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self, img_shape):
        super(Discriminator, self).__init__()

        self.label_embedding = nn.Embedding(10, 10)

        self.model = nn.Sequential(
            nn.Linear(10 + int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1),
        )

    def forward(self, img, labels):
        # Concatenate label embedding and image to produce input
        d_in = torch.cat((img.view(img.size(0), -1), self.label_embedding(labels)), -1)
        return self.model(d_in)


class CGANTrainer:
    def __init__(self,
                 train_loader,
                 save_path='/tmp',
                 latent_dim=100,
                 img_shape=(1, 28, 28),
                 n_epochs=200,
                 learning_rate=0.0002,
                 b1=0.5,
                 b2=0.999):

        self.save_path = save_path
        self.latent_dim = latent_dim
        self.train_loader = train_loader
        self.n_epochs = n_epochs

        # Initialize generator and discriminator
        self.generator = Generator(latent_dim, img_shape)
        self.discriminator = Discriminator(img_shape)

        # loss functions
        self.mse_loss = torch.nn.MSELoss()
        self.train_counter = []
        self.train_losses_generator = []
        self.train_losses_discriminator = []

        # handle gpu
        cuda = True if torch.cuda.is_available() else False

        if cuda:
            print("Sending models to gpu")
            self.generator.cuda()
            self.discriminator.cuda()
            self.mse_loss.cuda()

        self.FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        self.LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

        # optimizers
        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=learning_rate, betas=(b1, b2))
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=learning_rate, betas=(b1, b2))

    def train_generator(self, batch_size):
        self.optimizer_G.zero_grad()

        # generate synthetic images
        z = Variable(self.FloatTensor(np.random.normal(0, 1, (batch_size, self.latent_dim))))
        targets_gen = Variable(self.LongTensor(np.random.randint(0, 10, batch_size)))
        imgs_gen = self.generator(z, targets_gen)

        # try to fool discriminator into thinking these synthetic images are real
        labels_real = Variable(self.FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
        pred = self.discriminator(imgs_gen, targets_gen)
        g_loss = self.mse_loss(pred, labels_real)
        g_loss.backward()
        self.optimizer_G.step()

        return imgs_gen, targets_gen, g_loss

    def train_discriminator(self, imgs_real, targets_real, imgs_gen, targets_gen, batch_size):
        self.optimizer_D.zero_grad()

        # discriminator should predict these images are real
        labels_real = Variable(self.FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
        pred_real = self.discriminator(imgs_real, targets_real)
        d_real_loss = self.mse_loss(pred_real, labels_real)

        # discriminator should predict these images are fake
        labels_fake = Variable(self.FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)
        pred_fake = self.discriminator(imgs_gen.detach(), targets_gen)
        d_fake_loss = self.mse_loss(pred_fake, labels_fake)

        # Total discriminator loss
        d_loss = 0.5 * d_real_loss + 0.5 * d_fake_loss

        d_loss.backward()
        self.optimizer_D.step()

        return d_loss

    def save_samples_of_generated_images(self, n_row, batches_done):
        """Saves a grid of generated digits ranging from 0 to n_classes"""
        z = Variable(self.FloatTensor(np.random.normal(0, 1, (n_row ** 2, self.latent_dim))))
        labels = np.array([num for _ in range(n_row) for num in range(n_row)])
        labels = Variable(self.LongTensor(labels))
        gen_imgs = self.generator(z, labels)
        save_image(gen_imgs.data,
                   os.path.join(self.save_path, "{}.png".format(batches_done)),
                   nrow=n_row,
                   normalize=True)

    def train(self, log_interval=938):
        for epoch in range(self.n_epochs):
            for i, (imgs, targets_real) in enumerate(self.train_loader):
                batch_size = imgs.shape[0]

                # configure inputs and outputs for gpu/cpu
                imgs_real = Variable(imgs.type(self.FloatTensor))
                targets_real = Variable(targets_real.type(self.LongTensor))

                # train generator and collect synthetic images/targets
                imgs_gen, targets_gen, g_loss = self.train_generator(batch_size)

                # train discriminator
                d_loss = self.train_discriminator(imgs_real,
                                                  targets_real,
                                                  imgs_gen,
                                                  targets_gen,
                                                  batch_size)

                # checkpoint models and print losses
                if i % log_interval == 0:
                    print(
                        "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                        % (epoch + 1, self.n_epochs, i, len(self.train_loader), d_loss.item(), g_loss.item())
                    )
                    self.train_counter.append(
                        (i * 64) + ((epoch - 1) * len(self.train_loader.dataset)))
                    self.train_losses_discriminator.append(d_loss.item())
                    self.train_losses_generator.append(g_loss.item())

                    torch.save(self.generator.state_dict(),
                               os.path.join(self.save_path, 'generator.pth')
                               )

                    torch.save(self.discriminator.state_dict(),
                               os.path.join(self.save_path, 'discriminator.pth')
                               )

                    torch.save(self.optimizer_G.state_dict(),
                               os.path.join(self.save_path, 'optimizer_generator.pth')
                               )

                    torch.save(self.optimizer_G.state_dict(),
                               os.path.join(self.save_path, 'optimizer_discriminator.pth')
                               )

            self.save_samples_of_generated_images(n_row=10, batches_done=epoch)


if __name__ == '__main__':
    from data import load_mnist

    train_loader, test_loader = load_mnist('/tmp')
    trainer = CGANTrainer(train_loader, save_path='/tmp')
    trainer.train()
