import torch.nn as nn
import torch

class Generator_gan(nn.Module):
    def __init__(self, nz) -> None:
        super(Generator_gan, self).__init__()
        self.nz = nz

        self.main = nn.Sequential(
            nn.ConvTranspose2d(
            nz, 512, kernel_size=4, stride=1, padding=0,bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),

            nn.ConvTranspose2d(
            512, 256, kernel_size=4, stride=2, padding=1,bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),

            nn.ConvTranspose2d(
            256, 128, kernel_size=4, stride=2, padding=1,bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),

            nn.ConvTranspose2d(
            128, 64, kernel_size=4, stride=2, padding=1,bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, True),

            nn.ConvTranspose2d(
            64, 3, kernel_size=4, stride=2, padding=1,bias=False),
            nn.Tanh()
        )
    def forward(self, input):
        return self.main(input)


class Discriminator_gan(nn.Module):
    def __init__(self) -> None:
        super(Discriminator_gan, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=False),
            # nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),

            
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),

            
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )
    def forward(self, input):
        return self.main(input)


if __name__ == "__main__":
    tensor = torch.rand([1, 100, 32, 32])
    image = torch.rand([1,3, 64, 64])
    nz = 100
    model_generator = Generator_gan(nz)
    model_discriminator = Discriminator_gan()

    print(model_generator(tensor))
    print(model_discriminator(image))



