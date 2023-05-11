import os
import csv
import sys
import yaml
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CyclicLR
import argparse
import tabulate
import logging
import numpy as np
import pytorch_ssim
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from mpl_toolkits.basemap import Basemap
import imageio.v2 as imageio

from torch.utils.data import DataLoader

from torchvision import transforms

from datetime import datetime

import unet
import gunet
from dataset import *
from messenger import Messenger


def log_progress(epoch, epoch_train_loss, epoch_mse_loss, epoch_mae_loss, epoch_ssim_loss):
    global  logger, model_config, test_loader, model, device, directory, run_name
    logger.debug('logging progress')

    iter_testloader = iter(test_loader)
    with torch.no_grad():
        (inputs, targets) = next(iter_testloader)
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)

        inputs = inputs.cpu()
        outputs = outputs.cpu()
        targets = targets.cpu()

        max_j = 0
        max_mean = 0
        for j in range(len(targets)):
            mean = np.mean(np.array(targets[j]))
            if max_mean < mean:
                max_mean = mean
                max_j = j

        m = Basemap(llcrnrlon=11.684135852177782,
                    llcrnrlat=48.13851378158722,
                    urcrnrlon=19.421619437767024,
                    urcrnrlat=51.589943631895736,
                    resolution='i',
                    projection='mill')

        os.makedirs(f"{directory}/images/tmp", exist_ok=True)

        fig = plt.figure(dpi=400)
        axs = ImageGrid(fig, 111, nrows_ncols=(1, 2), axes_pad=0)
        cmap = 'jet'

        def modified_sigmoid(x, k=50, a=0.2):
            return 1 / (1 + np.exp(-k * (x - a)))

        def flatten_image_on_white_bg(image):
            white_bg = Image.new('RGBA', image.size, (255, 255, 255, 255))
            white_bg.paste(image, mask=image.getchannel('A'))
            return white_bg.convert('RGB')

        images = []
        for i in range(0, outputs.shape[1], 1):
            m.drawcountries(linewidth=0.1, ax=axs[0])
            m.drawcountries(linewidth=0.1, ax=axs[1])
            m.imshow(targets[max_j, i], cmap=cmap,
                     vmin=0, vmax=1,
                     ax=axs[0],
                     origin='upper',
                     alpha=modified_sigmoid(targets[max_j, i]))
            m.imshow(outputs[max_j, i], cmap=cmap,
                     vmin=0, vmax=1,
                     ax=axs[1],
                     origin='upper',
                     alpha=modified_sigmoid(outputs[max_j, i]))
            for ax in axs:
                ax.set_axis_off()

            fig.savefig(f"{directory}/images/tmp/{i}.png", bbox_inches='tight', pad_inches=0)

            for ax in axs:
                ax.clear()
        plt.close()

        images = [os.path.join(f"{directory}/images/tmp", f) for f in os.listdir(f"{directory}/images/tmp") if f.endswith('.png')]
        images.sort()

        with imageio.get_writer(f"{directory}/images/{epoch}.gif", mode='I') as writer:
            for filename in images:
                image = imageio.imread(filename)
                writer.append_data(image)

    csv_file = f'{directory}/training.csv'
    header = ['epoch', f'train_{model_config["loss_function"]}_loss', 'val_mse_loss', 'val_mae_loss', 'val_ssim_loss',
              'time']
    data = [epoch, epoch_train_loss, epoch_mse_loss, epoch_mae_loss, epoch_ssim_loss,
            datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
    if not os.path.exists(csv_file):
        with open(csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(header)

    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(data)

    logger.info(f"epoch {epoch + 1}, "
                f'train_{model_config["loss_function"]}_loss = {epoch_train_loss:.10f}, '
                f'val_mse_loss = {epoch_mse_loss:.10f}, '
                f'val_mae_loss = {epoch_mae_loss:.10f}, '
                f'val_ssim_loss = {epoch_ssim_loss:.10f}, '
                f'time = {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    messenger.send(f"run {run_name}",
                   f"epoch {epoch + 1}\n"
                   f'train_{model_config["loss_function"]}_loss = {epoch_train_loss:.10f}\n'
                   f'val_mse_loss = {epoch_mse_loss:.10f}\n'
                   f'val_mae_loss = {epoch_mae_loss:.10f}\n'
                   f'val_ssim_loss = {epoch_ssim_loss:.10f}\n'
                   f'time = {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
                   f"{directory}/images/{epoch}.gif")


def train_model(optimizer, scheduler, criterion):
    global train_loader, model, device, logger
    logger.debug('training')

    model.train()
    running_loss = 0.0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        running_loss += loss.item()
    return running_loss / len(train_loader)


def validate_model(losses):
    global val_loader, model, device, logger
    logger.debug('validating')

    model.eval()
    running_losses = np.zeros((len(losses)))
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            for i, loss in enumerate(losses):
                running_losses[i] += loss(outputs, targets).item()
    return running_losses / len(val_loader)


def train():
    global model, model_config, train_loader, logger, directory
    logger.debug('starting to train')

    os.makedirs(f"{directory}/models", exist_ok=True)

    optimizer = optim.Adam(model.parameters(), lr=model_config['learning_rate'])
    # scheduler = CyclicLR(optimizer, base_lr=model_config['learning_rate_min'],
    #                      max_lr=model_config['learning_rate_max'],
    #                      step_size_up=2 * len(train_loader),
    #                      cycle_momentum=False)
    scheduler = None
    mse_loss = nn.MSELoss()
    ssim_loss = pytorch_ssim.SSIM()
    mae_loss = nn.L1Loss()

    if model_config['loss_function'] == "mse":
        criterion = mse_loss
    else:
        criterion = mae_loss

    best_val_ssim = -float('inf')
    # Training of the model.
    for epoch in range(model_config['epochs']):
        epoch_train_loss = train_model(optimizer, scheduler, criterion)
        epoch_mse_loss, epoch_mae_loss, epoch_ssim_loss = validate_model([mse_loss, mae_loss, ssim_loss])

        log_progress(epoch, epoch_train_loss, epoch_mse_loss, epoch_mae_loss, epoch_ssim_loss)
        if epoch_ssim_loss > best_val_ssim:
            best_val_ssim = epoch_ssim_loss
            torch.save(model.state_dict(), f"{directory}/models/best.pt")
        torch.save(model.state_dict(), f"{directory}/models/last.pt")


def create_dataloaders():
    global train_loader, val_loader, test_loader, model_config, logger
    logger.debug("getting data")

    X_train, y_train, X_val, y_val, X_test, y_test = get_images(dataset_config['data_path'],
                                                                dataset_config['stride_minutes'],
                                                                dataset_config['input_length'],
                                                                dataset_config['output_length'],
                                                                dataset_config['chunk_size'],
                                                                dataset_config['test_frac'],
                                                                dataset_config['val_frac'],
                                                                dataset_config['seed'])
    # X_train, y_train, X_val, y_val, X_test, y_test = get_images_like_phydnet(dataset_config['data_path'],
    #                                                                          dataset_config['stride_minutes'],
    #                                                                          dataset_config['input_length'],
    #                                                                          dataset_config['output_length'],
    #                                                                          dataset_config['chunk_size'],
    #                                                                          dataset_config['test_frac'],
    #                                                                          dataset_config['val_frac'],
    #                                                                          dataset_config['seed'])

    logger.debug("creating dataloaders")

    transform = transforms.Compose([
        transforms.CenterCrop((256, 512)),
        transforms.ToTensor(),
    ])

    train_dataset = CustomDataset(X_train, y_train, transform=transform)
    val_dataset = CustomDataset(X_val, y_val, transform=transform)
    test_dataset = CustomDataset(X_test, y_test, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=model_config['batch_size'], shuffle=True, num_workers=64)
    val_loader = DataLoader(val_dataset, batch_size=model_config['batch_size'], shuffle=False, num_workers=64)
    test_loader = DataLoader(test_dataset, batch_size=model_config['batch_size'], shuffle=True, num_workers=1)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # parser.add_argument("-v", "--verbose", default=False, type=bool)
    parser.add_argument("--model_config", type=str, required=True)
    parser.add_argument("--dataset_config", type=str, required=True)
    parser.add_argument("-n", "--run_name", type=str, required=False,
                        help="if the run name exist, the run will be resumed")
    parser.add_argument("-d", "--device", required=True, choices=['cpu', 'cuda:0', 'cuda:1', 'cuda:2', 'cuda:3'])
    args = parser.parse_args()

    global model_config, dataset_config
    with open(args.model_config) as file:
        model_config = yaml.safe_load(file)
    with open(args.dataset_config) as file:
        dataset_config = yaml.safe_load(file)

    global device
    device = torch.device(args.device)

    global model
    if model_config['model'] == 'unet':
        model = unet.UNet(dataset_config['input_length'],
                          dataset_config['output_length'],
                          model_config['dropout_rate'],
                          model_config['kernel_size']).to(device)
    else:
        model = gunet.GUNet(in_channels=dataset_config['input_length'],
                            out_channels=dataset_config['output_length'],
                            dropout_rate=model_config['dropout_rate'],
                            epsilon=model_config['epsilon'],
                            r=model_config['r']).to(device)
    global run_name
    if args.run_name:
        run_name = args.run_name
    else:
        run_name = model_config['model'] + "_" + datetime.now().strftime('%Y%m%d_%H%M%S')

    global directory
    directory = f"./run/{run_name}"
    os.makedirs(directory, exist_ok=True)

    global logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(f'./run/{run_name}/run.log')
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(message)s', datefmt="%Y.%m.%d %H:%M")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    if os.path.exists("./run/" + args.run_name + "/models/last.pt"):
        model.load_state_dict(torch.load("./run/" + args.run_name + "/models/last.pt", map_location=device))
        logger.debug("model loaded")
    else:
        logger.debug("no previous model found")

    headers = ['Argument', 'Value']
    table = [['run', run_name], ['device', args.device]]
    logger.info("\narguments\n" + tabulate.tabulate(table, headers=headers, tablefmt="fancy_grid"))

    global messenger
    messenger = Messenger("xxx", "xxx", logger)

    create_dataloaders()

    with open(f"./run/{run_name}/dataset_config.yaml", 'w') as file:
        yaml.dump(dataset_config, file)
    with open(f"./run/{run_name}/model_config.yaml", 'w') as file:
        yaml.dump(model_config, file)

    messenger.send(f"run {run_name}", "training started")
    train()
    logger.debug('training finished')
