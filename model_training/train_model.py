import numpy as np
import cv2
import torch
import os
import random

letter_path = 'base_letters/'
filenames = os.listdir(letter_path)

batch_size = 64
input_layer = 32 * 32
hidden1 = 500
hidden2 = 250
output_layer = 26 * 2 + 10
learning_rate = 1e-4


def get_label(fname):
    letter = fname.split('_')[2]
    if 'upper' in fname:
        return ord(letter) - 65
    elif 'lower' in fname:
        return ord(letter) - 71
    elif 'number' in fname:
        return ord(letter) + 4


def training_batch(batch_size):
    batch_imgs = random.sample(filenames, batch_size)
    samples = []
    labels = []
    for filename in batch_imgs:
        img = cv2.imread(letter_path + filename)
        samples.append(np.ndarray.flatten(img))
        labels.append(get_label(filename))
    return samples, labels


if __name__ == '__main__':
    random.seed()
    model = torch.nn.Sequential(torch.nn.Linear(input_layer, hidden1),
                                torch.nn.ReLU(),
                                torch.nn.Linear(hidden1, hidden2),
                                torch.nn.Linear(hidden2, output_layer))

    loss_fn = torch.nn.MSELoss(size_average=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for t in range(500):
        samples, labels = training_batch(batch_size=batch_size)
        predictions = model(samples)
        loss = loss_fn(predictions, labels)
        print('EPOCH', t, 'LOSS', loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
