from __future__ import print_function
#%matplotlib inline
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

if __name__ == "__main__":

    # 再現性のためにランダムシードを設定する
    manualSeed = 999
    #manualSeed = random.randint（1、10000）＃新しい結果が必要な場合に使用
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    # データセットのルートディレクトリ
    dataroot = "./data/celeba"

    # データローダーのワーカー数
    workers = 2

    # トレーニングのバッチサイズ
    batch_size = 128

    # トレーニング画像の空間サイズ。
    # すべての画像はトランスフォーマーを使用してこのサイズに変更されます。
    image_size = 64

    # トレーニング画像のチャネル数。カラー画像の場合は「3」
    nc = 3

    # 潜在ベクトル z のサイズ（つまり、ジェネレータ入力のサイズ）
    nz = 100

    # 生成器の feature map のサイズ
    ngf = 64

    # 識別器の feature map のサイズ
    ndf = 64

    # エポック数
    num_epochs = 5

    # 学習率
    lr = 0.0002

    # Adam オプティマイザのBeta1ハイパーパラメータ
    beta1 = 0.5

    # 使用可能なGPUの数。0の場合、CPUモードで実行されます
    ngpu = 1

    # 画像フォルダデータセットは、以下で設定した方法で使用できます。

    # データセットを作成する
    dataset = dset.ImageFolder(root=dataroot,
                            transform=transforms.Compose([
                                transforms.Resize(image_size),
                                transforms.CenterCrop(image_size),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ]))

    # データローダーを作成する
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                            shuffle=True, num_workers=workers)

    # Decide which device we want to run on
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

    # トレーニング画像をプロットする（RuntimeError のためコメントアウト）
    real_batch = next(iter(dataloader))
    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
    # plt.show()

    # G（生成器）とD（識別器）の重みの初期化
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    # Generator (生成器）
    class Generator(nn.Module):
        def __init__(self, ngpu):
            super(Generator, self).__init__()
            self.ngpu = ngpu
            self.main = nn.Sequential(
                # 入力は Z で、畳み込み層に渡されます
                nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
                nn.BatchNorm2d(ngf * 8),
                nn.ReLU(True),
                # サイズ (ngf*8) x 4 x 4
                nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 4),
                nn.ReLU(True),
                # サイズ  (ngf*4) x 8 x 8
                nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 2),
                nn.ReLU(True),
                # サイズ (ngf*2) x 16 x 16
                nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf),
                nn.ReLU(True),
                # サイズ (ngf) x 32 x 32
                nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
                nn.Tanh()
                # サイズ (nc) x 64 x 64
            )

        def forward(self, input):
            return self.main(input)

    # Create the generator
    netG = Generator(ngpu).to(device)

    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (ngpu > 1):
        netG = nn.DataParallel(netG, list(range(ngpu)))

    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.02.
    netG.apply(weights_init)

    # Print the model
    # print(netG)

    class Discriminator(nn.Module):
        def __init__(self, ngpu):
            super(Discriminator, self).__init__()
            self.ngpu = ngpu
            self.main = nn.Sequential(
                # 入力は (nc) x 64 x 64
                nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                # サイズ (ndf) x 32 x 32
                nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),
                # サイズ (ndf*2) x 16 x 16
                nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 4),
                nn.LeakyReLU(0.2, inplace=True),
                # サイズ (ndf*4) x 8 x 8
                nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 8),
                nn.LeakyReLU(0.2, inplace=True),
                # サイズ (ndf*8) x 4 x 4
                nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
                nn.Sigmoid()
            )

        def forward(self, input):
            return self.main(input)

    # Create the Discriminator
    # 識別器を作成します
    netD = Discriminator(ngpu).to(device)

    # Handle multi-gpu if desired
    # 必要に応じてGPUを使用します
    if (device.type == 'cuda') and (ngpu > 1):
        netD = nn.DataParallel(netD, list(range(ngpu)))

    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.2.
    # weights_init関数を適用して、すべての重みを平均「0」、標準偏差「0.02」でランダムに初期化します。
    netD.apply(weights_init)

    # Print the model
    # モデルを出力します
    # print(netD)

    # BCELoss関数を初期化します
    criterion = nn.BCELoss()

    # ジェネレータの進行を視覚化するために使用する潜在ベクトルを作成します
    fixed_noise = torch.randn(64, nz, 1, 1, device=device)

    # トレーニング中に本物のラベルと偽のラベルのルールを設定します
    real_label = 1.
    fake_label = 0.

    # G と D に Adam オプティマイザを設定する
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))


    # 進捗状況を追跡するためのリスト
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0
    print("Starting Training Loop...")
    # エポックごとのループ
    for epoch in range(num_epochs):
        # データローダーのバッチごとのループ
        for i, data in enumerate(dataloader, 0):
            ############################
            # (1) Dネットワークの更新：log(D(x)) + log(1 - D(G(z))) を最大化します
            ###########################
            ## 実在の画像でトレーニングします
            netD.zero_grad()
            # バッチのフォーマット
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            # 実在の写真で D の順伝播させます
            output = netD(real_cpu).view(-1)
            # 損失を計算します
            errD_real = criterion(output, label)
            # 逆伝播でDの勾配を計算します
            errD_real.backward()
            D_x = output.mean().item()

            ## 偽の画像でトレーニングします
            # 潜在ベクトルのバッチを生成します
            noise = torch.randn(b_size, nz, 1, 1, device=device)
            # Gで偽の画像を生成します
            fake = netG(noise)
            label.fill_(fake_label)
            # 生成した偽画像をDで分類します
            output = netD(fake.detach()).view(-1)
            # Dの損失を計算します
            errD_fake = criterion(output, label)
            # 勾配を計算します
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # 実在の画像の勾配と偽画像の勾配を加算します
            errD = errD_real + errD_fake
            # Dを更新します
            optimizerD.step()

            ############################
            # (2) Gネットワ​​ークの更新：log(D(G(z))) を最大化します
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # 偽のラベルは生成器の損失にとって本物です
            # パラメータ更新後のDを利用して、偽画像を順伝播させます
            output = netD(fake).view(-1)
            # この出力に基づいてGの損失を計算します
            errG = criterion(output, label)
            # Gの勾配を計算します
            errG.backward()
            D_G_z2 = output.mean().item()
            # Gを更新します
            optimizerG.step()

            # トレーニング統計を出力します
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                    % (epoch, num_epochs, i, len(dataloader),
                        errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            # 後でプロットするために損失を保存します
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            # fixed_noiseによる G の出力を保存し、生成器の精度を確認します
            if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

            iters += 1
