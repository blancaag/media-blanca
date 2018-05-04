import torch
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F


def heatmap(det, X, resolution, patch_size):
    ''' Create a resolution by resolution pixel heatmap with chances of spoof. '''
    c, h, w = X.shape
    # get patches from a uniform range
    hr = float(h - patch_size) / (resolution - 1)
    wr = float(w - patch_size) / (resolution - 1)
    batch = np.zeros((resolution * resolution, 3, patch_size, patch_size))
    for y in range(resolution):
        for x in range(resolution):
            patch = X[:, int(y * hr):int(y * hr + patch_size), int(x * wr):int(x * wr + patch_size)] / 255.0
            batch[y * resolution + x, :, :, :] = patch
    # run batch of patches through the network
    inputs = Variable(torch.from_numpy(batch).float(), volatile=True).cuda()
    return F.softmax(det(inputs)).data.cpu().numpy()

def test_network(network, files):
    ''' Run every file through the network and compare it with the correct label. '''
    network.eval()
    network.train(False)
    y_true = []
    y_pred = []
    y_live = []
    for filename, label in files:
        # load sample from filename
        sample = np.load(filename)
        # save true label
        y_true.append(label)
        # predict label for sample
        result = heatmap(network, sample.transpose(2,0,1), 32, 32)
        y_pred.append(np.mean(result, 0))
        # save live_photo_id
        y_live.append(filename.split('.')[0].split('/')[-1])
    return y_true, y_pred, y_live
