import itertools
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import generative_model_score


class Encoder(nn.Module):
    def __init__(self, latent_dim, image_shape):
        super(Encoder, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(int(np.prod(image_shape)), 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.mu = nn.Linear(64, latent_dim)
        self.sigma = nn.Linear(64, latent_dim)
        self.latent_dim = latent_dim

    def forward(self, x):
        x_flat = x.view(x.shape[0], -1)
        mu, sigma = self.encode(x_flat)
        z_posterior = self.reparameterize(mu, sigma)
        return z_posterior

    def encode(self, x):
        x = self.model(x)
        mu = self.mu(x)
        sigma = self.sigma(x)
        return mu, sigma

    def reparameterize(self, mu, sigma):
        from torch.autograd import Variable
        batch_size = mu.size(0)
        eps = Variable(torch.FloatTensor(np.random.normal(0, 1, (batch_size, self.latent_dim)))).cuda()
        return eps * sigma + mu


class Decoder(nn.Module):
    def __init__(self, latent_dim, image_shape):
        super(Decoder, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, int(np.prod(image_shape))),
            nn.Tanh(),
        )
        self.image_shape = image_shape

    def forward(self, z_posterior):
        decoded_flat = self.model(z_posterior)
        decoded = decoded_flat.view(decoded_flat.shape[0], *self.image_shape)
        return decoded


class Discriminator(nn.Module):
    def __init__(self, latent_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, z):
        validity = self.model(z)
        return validity


def sample_image(encoder, decoder, x):
    z = encoder(x)
    return decoder(z)


def get_celebA_dataset(batch_size):
    image_path = "../data/"
    transformation = transforms.Compose([
        transforms.Resize((16, 16)),
        transforms.ToTensor(),
    ])
    train_dataset = torchvision.datasets.ImageFolder(image_path + 'celeba', transformation)
    num_train = len(train_dataset)
    indices = list(range(num_train))
    train_indices, test_indices = indices[:10000], indices[200000:]
    train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
    test_sampler = torch.utils.data.SubsetRandomSampler(test_indices)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
    test_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=test_sampler)
    return train_loader, test_loader

def get_cifar1_dataset(batch_size) : 
    import torchvision.transforms as transforms
    import torchvision
    dataset = torchvision.datasets.CIFAR10(root='../../20210306_gan/dataset', #download=True,
                               transform=transforms.Compose([
                                   transforms.Resize((16,16)),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
    data = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    return data

def update_autoencoder(ae_optimizer, X_train_batch, encoder, decoder):
    ae_optimizer.zero_grad()
    z_posterior = encoder(X_train_batch)
    X_decoded = decoder(z_posterior)
    pixelwise_loss = torch.nn.L1Loss()
    r_loss = pixelwise_loss(X_decoded, X_train_batch)
    r_loss.backward()
    ae_optimizer.step()
    return r_loss


def update_discriminator(d_optimizer, X_train_batch, encoder, discriminator, latent_dim):
    d_optimizer.zero_grad()
    batch_size = X_train_batch.size(0)
    from torch.autograd import Variable
    z_prior = Variable(torch.FloatTensor(np.random.normal(0, 1, (batch_size, latent_dim)))).cuda()
    z_posterior = encoder(X_train_batch)
    d_loss = -torch.mean(torch.log(discriminator(z_prior)) + torch.log(1 - discriminator(z_posterior)))
    d_loss.backward()
    d_optimizer.step()
    return d_loss.data


def update_generator(g_optimizer, X_train_batch, encoder, discriminator):
    g_optimizer.zero_grad()
    z_posterior = encoder(X_train_batch)
    g_loss = -torch.mean(torch.log(discriminator(z_posterior)))
    g_loss.backward()
    g_optimizer.step()
    return g_loss.data


def save_losses(epochs, r_losses, d_losses, g_losses):
    X = range(1, epochs + 1)
    fig = plt.figure(figsize=(30, 30))
    plt.subplot(3, 1, 1)
    plt.title("r_losses")
    plt.plot(X, r_losses, color="blue", linestyle="-", label="r_losses")
    plt.subplot(3, 1, 2)
    plt.title("g_losses")
    plt.plot(X, g_losses, color="purple", linestyle="-", label="g_losses")
    plt.subplot(3, 1, 3)
    plt.title("d_losses")
    plt.plot(X, d_losses, color="red", linestyle="-", label="d_losses")
    plt.savefig('aae_celebA/losses.png')
    plt.close()


def save_images(each_epoch, images):
    images = images.numpy()
    images = np.transpose(images, (0, 2, 3, 1))
    plt.figure(figsize=(5, 5))
    for i in range(images.shape[0]):
        plt.subplot(5, 5, i + 1)
        plt.imshow((images[i, :, :, :] * 255).astype('uint8'))
        plt.axis('off')
    plt.savefig('aae_celebA/image_at_epoch_{:04d}.png'.format(each_epoch+1))
    plt.close()


def save_scores_and_print(current_epoch, epochs, r_loss, d_loss, g_loss, precision, recall, fid, inception_score_real,
                          inception_score_fake):
    f = open("./logs/aae_celebA/generative_scores.txt", "a")
    f.write("%d %f %f %f %f %f %f %f %f\n" % (current_epoch, r_loss, d_loss, g_loss, precision, recall, fid, inception_score_real, inception_score_fake))
    f.close()
    print("[Epoch %d/%d] [R loss: %f] [D loss: %f] [G loss: %f] [precision: %f] [recall: %f] [fid: %f] [inception_score_real: %f] [inception_score_fake: %f]"
          % (current_epoch, epochs, r_loss, d_loss, g_loss, precision, recall, fid, inception_score_real, inception_score_fake))


def main():
    inception_model_score = generative_model_score.GenerativeModelScore()

    batch_size = 64
    epochs = 1000
    #train_loader, test_loader = get_celebA_dataset(batch_size)
    train_loader = get_cifar1_dataset(batch_size)
    latent_dim = 10
    image_shape = [3, 16, 16]

    encoder = Encoder(latent_dim, image_shape).cuda()
    decoder = Decoder(latent_dim, image_shape).cuda()
    discriminator = Discriminator(latent_dim).cuda()
    ae_optimizer = torch.optim.Adam(itertools.chain(encoder.parameters(), decoder.parameters()), lr=1e-4)
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=1e-4)
    g_optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-4)

    r_losses = []
    d_losses = []
    g_losses = []
    precisions = []
    recalls = []
    fids = []
    inception_scores_real = []
    inception_scores_fake = []



    import torchvision
    from torch.autograd import Variable
    from torchvision import transforms
    import tqdm

    # put all the real image in inception_model_score
    inception_model_score.model_to('cuda')
    for each_batch in tqdm.tqdm(train_loader) : 
        X_train_batch = each_batch[0].cuda()
        inception_model_score.put_real(X_train_batch)

    inception_model_score.calculate_real_image_statistics()



    for i in range(0, epochs):
        batch_count = 0
        for each_batch in tqdm.tqdm(train_loader):
            batch_count += 1
            X_train_batch = Variable(each_batch[0]).cuda()
            r_loss = update_autoencoder(ae_optimizer, X_train_batch, encoder, decoder)
            d_loss = update_discriminator(d_optimizer, X_train_batch, encoder, discriminator, latent_dim)
            g_loss = update_generator(g_optimizer, X_train_batch, encoder, discriminator)

            sampled_images = sample_image(encoder, decoder, X_train_batch)

            inception_model_score.put_fake(sampled_images)


        inception_model_score.calculate_fake_image_statistics()
        metrics = inception_model_score.calculate_generative_score()
        inception_model_score.clear_fake()

        precision, recall, fid, inception_score_real, inception_score_fake, density, coverage = \
            metrics['precision'], metrics['recall'], metrics['fid'], metrics['real_is'], metrics['fake_is'], metrics['density'], metrics['coverage']

        r_losses.append(r_loss)
        d_losses.append(d_loss)
        g_losses.append(g_loss)
        precisions.append(precision)
        recalls.append(recall)
        fids.append(fid)
        inception_scores_real.append(inception_score_real)
        inception_scores_fake.append(inception_score_fake)
        save_scores_and_print(i + 1, epochs, r_loss, d_loss, g_loss, precision, recall, fid, inception_score_real, inception_score_fake)
    save_losses(epochs, r_losses, d_losses, g_losses)


if __name__ == "__main__":
    main()
