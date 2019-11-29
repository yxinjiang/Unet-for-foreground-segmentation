import torch
import torch.nn.functional as F
from tqdm import tqdm

from dice_loss import dice_coeff


def eval_net(net, dataset, device, n_val):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    tot = 0

    for i, b in tqdm(enumerate(dataset), total=n_val, desc='Validation round', unit='img'):
        img = b['O'].to(device)
        true_mask = b['M'].to(device)

        #img = img.squeeze(dim=0)
        print('input image shape: ',img.shape)
        
        #true_mask = true_mask.squeeze(0)
        print('true_mask shape: ',true_mask.shape)

        mask_pred = net(img).squeeze(0)
        print('mask_pred shape ',mask_pred.shape)

        mask_pred = (mask_pred > 0.5).float()
        if net.num_classes > 1:
            tot += F.cross_entropy(mask_pred.unsqueeze(dim=0), true_mask.unsqueeze(dim=0)).item()
        else:
            tot += dice_coeff(mask_pred, true_mask.squeeze(dim=1)).item()

    return tot / n_val
