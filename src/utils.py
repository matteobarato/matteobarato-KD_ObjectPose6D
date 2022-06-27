import os

import torch


def progress_bar(current, total, msg=None):
    if current >= (total -1) : print(f"Batch {current}/{total} : {msg}")

def resume_checkpoint(net, path, map_location=torch.device('cuda') ):
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/'+path+'.pth', map_location=map_location )
    net.load_state_dict(checkpoint['net'])
    return  checkpoint['acc'], checkpoint['epoch']
    # best_acc = checkpoint['acc']
    # start_epoch = checkpoint['epoch']
