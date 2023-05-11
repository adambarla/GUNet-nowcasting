import os
import optuna
from optuna.trial import TrialState
import logging
import pytorch_ssim
import argparse
import yaml
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
import imageio
import numpy as np

import unet
import gunet
from dataset import *
from messenger import Messenger

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", type=str, required=True, help="path to config.yaml")
parser.add_argument("-d", "--device", required=True, choices=['cpu', 'cuda:0', 'cuda:1', 'cuda:2', 'cuda:3'])
args = parser.parse_args()

with open(args.config) as file:
    config = yaml.safe_load(file)

run_name = config['model'] + "_" + datetime.now().strftime('%Y%m%d_%H%M%S')
os.mkdir(f"./tuning/{run_name}")
os.mkdir(f"./tuning/{run_name}/trials")

X_train, y_train, X_val, y_val, X_test, y_test = get_images(config['data_path'],
                                                            config['stride_minutes'],
                                                            config['input_length'],
                                                            config['output_length'],
                                                            config['chunk_size'],
                                                            config['test_frac'],
                                                            config['val_frac'],
                                                            config['seed'])
# X_train, y_train, X_val, y_val, X_test, y_test = get_images_like_phydnet(config['data_path'],
#                                                         config['stride_minutes'],
#                                                         config['input_length'],
#                                                         config['output_length'],
#                                                         config['chunk_size'],
#                                                         config['test_frac'],
#                                                         config['val_frac'],
#                                                         config['seed'])


logger = optuna.logging.get_logger(__name__)
logger.setLevel(logging.DEBUG)  # Setup the root logger.
file_handler = logging.FileHandler(f'./tuning/{run_name}/run.log', mode="w")
file_handler.setLevel(logging.DEBUG)
logger.addHandler(file_handler)

messenger = Messenger("xxx", "xxx", logger)


def get_trial_info(trial):
    info = f"Trial: {trial.number}\n" \
           f"Value: {trial.value}\n" \
           f"State: {trial.state}\n" \
           f"Params:\n"
    for key, value in trial.params.items():
        info += f"{key}: {value}\n"
    return info


def send_notification(study, trial):
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
    message = "Study statistics: \n" \
              f"Number of finished trials: {len(study.trials)}\n" \
              f"Number of pruned trials: {len(pruned_trials)}\n" \
              f"Number of complete trials: {len(complete_trials)}\n\n" \
              f"Current {get_trial_info(trial)}\n\n" \
              f"Best {get_trial_info(study.best_trial)}\n"

    messenger.send(f"{study.study_name}", message)


def log_progress(trial, model, device, dataloader_test, epoch, train_loss, val_ssim):
    testloader = iter(dataloader_test)
    with torch.no_grad():
        (inputs, targets) = next(testloader)
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

        line_width = 1
        combined_images = np.concatenate((inputs[max_j][0], np.ones((inputs[max_j][0].shape[0], line_width)),
                                          np.zeros(inputs[max_j][0].shape)), axis=1)
        for i in range(1, len(inputs[max_j])):
            combined_images = np.concatenate((combined_images,
                                              np.ones((line_width, combined_images.shape[1])),
                                              np.concatenate((inputs[max_j][i],
                                                              np.ones((inputs[max_j][0].shape[0], line_width)),
                                                              np.zeros(inputs[max_j][0].shape)), axis=1)), axis=0)

        for i in range(0, len(targets[max_j])):
            combined_images = np.concatenate((combined_images,
                                              np.ones((line_width, combined_images.shape[1])),
                                              np.concatenate((targets[max_j][i],
                                                              np.ones((targets[max_j][i].shape[0], line_width)),
                                                              outputs[max_j][i]), axis=1)), axis=0)

        imageio.imsave(f'./tuning/{run_name}/trials/{trial.number}/images/{epoch + 1}.png',
                       (255 - 255 * combined_images).astype('uint8'))
        gif = []
        for i in range(0, len(targets[max_j])):
            gif.append(np.concatenate(
                (targets[max_j][i], np.ones((targets[max_j][i].shape[0], line_width)), outputs[max_j][i]), axis=1))
        imageio.mimsave(f'./tuning/{run_name}/trials/{trial.number}/images/{epoch + 1}.gif',
                        (255 - 255 * np.array(gif)).astype('uint8'), format='gif')

    logger.info(
        f"study {run_name}, trial {trial.number}, epoch {epoch + 1}, train_loss: {train_loss:.10f}, val_ssim:{val_ssim:.10f}")

    m = ""
    for key, value in trial.params.items():
        m += f"{key}: {value}\n"
    messenger.send(f"study {run_name}, trial {trial.number}",
                   f"epoch: {epoch + 1}\ntrain_loss: {train_loss:.10f}\nval_ssim:{val_ssim:.10f}\n\n{m}",
                   f'./tuning/{run_name}/trials/{trial.number}/images/{epoch + 1}.gif')


def log_callback(study, trial):
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
    message = "Study statistics: \n" \
              f"Number of finished trials: {len(study.trials)}\n" \
              f"Number of pruned trials: {len(pruned_trials)}\n" \
              f"Number of complete trials: {len(complete_trials)}\n\n" \
              f"Current {get_trial_info(trial)}\n\n" \
              f"Best {get_trial_info(study.best_trial)}\n"
    logger.info(message)


def train_model(model, train_loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    return running_loss / len(train_loader)


def validate_model(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item()
    return running_loss / len(val_loader)


def objective(trial):
    best_val_ssim = -float('inf')

    os.mkdir(f"./tuning/{run_name}/trials/{trial.number}/")
    os.mkdir(f"./tuning/{run_name}/trials/{trial.number}/images/")
    # Define hyperparameters to optimize
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    dropout_rate = trial.suggest_float("dropout_rate", 1e-5, 0.5, log=True)
    kernel_size = trial.suggest_int("kernel_size", 2, 8, step=2)
    loss_name = trial.suggest_categorical("loss_function", ["mse", "mae"])

    # Set the loss function based on the suggested value
    if loss_name == "mse":
        criterion = nn.MSELoss()
    elif loss_name == "mae":
        criterion = nn.L1Loss()

    device = torch.device(args.device)

    # Create the model, optimizer, and criterion
    model = unet.UNet(in_channels=config['input_length'],
                        out_channels=config['input_length'],
                        dropout_rate=dropout_rate,
                        k=kernel_size
                        ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    ssim_loss = pytorch_ssim.SSIM()

    # Get the dataloaders
    transform = transforms.Compose([
        transforms.CenterCrop((256, 512)),
        transforms.ToTensor(),
    ])
    train_dataset = CustomDataset(X_train, y_train, transform=transform)
    val_dataset = CustomDataset(X_val, y_val, transform=transform)
    test_dataset = CustomDataset(X_test, y_test, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=64)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=64)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=1)

    val_ssim = 0
    # Training of the model.
    for epoch in range(config['epochs']):
        train_loss = train_model(model, train_loader, optimizer, criterion, device)
        val_ssim = validate_model(model, val_loader, ssim_loss, device)
        trial.report(val_ssim, epoch)
        log_progress(trial, model, device, test_loader, epoch, train_loss, val_ssim)
        if val_ssim > best_val_ssim:
            best_val_ssim = val_ssim
            torch.save(model.state_dict(), f"./tuning/{run_name}/trials/{trial.number}/best.pt")
        torch.save(model.state_dict(), f"./tuning/{run_name}/trials/{trial.number}/last.pt")

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return val_ssim


if __name__ == "__main__":
    logger.info("Start optimization.")
    study = optuna.create_study(direction="maximize", study_name=f"study {run_name}")

    study.optimize(objective, n_trials=50, callbacks=[log_callback, send_notification], gc_after_trial=True)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    logger.info("Study statistics: ")
    logger.info("Number of finished trials: ", len(study.trials))
    logger.info("Number of pruned trials: ", len(pruned_trials))
    logger.info("Number of complete trials: ", len(complete_trials))

    logger.info("Best trial:")
    trial = study.best_trial

    logger.info("Value: ", trial.value)

    logger.info("Params: ")
    for key, value in trial.params.items():
        logger.info("  {}: {}".format(key, value))
