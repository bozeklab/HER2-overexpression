import argparse
import os
import torch.multiprocessing as mp

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from torch.nn import CrossEntropyLoss
from torchvision.models import resnet34
from torchvision import transforms
import pandas as pd

from src.utils import calculate_weights
from src.dataset import DistributedWeightedSampler, ImageDataset
from src.models import ResnetABMIL

def train_step(train_loader, model, criterion, optimizer):
    model.train()
    training_epoch_loss = 0
    for i, batch in enumerate(train_loader):
        img_tensor, target = batch[0].cuda(), batch[1].cuda()
        output = model(img_tensor)
        if isinstance(output, (tuple, list)):
            output = output[0]
        optimizer.zero_grad()
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        training_epoch_loss += loss.item()
    
    training_phase_results = {
        'Loss': training_epoch_loss/( (i+1) ),
        'Learning rate': optimizer.param_groups[0]['lr']}

    return training_phase_results


def validate_step(val_loader, model, criterion):
    model.eval()
    val_epoch_loss = torch.tensor([0.0]).cuda()
    acc = torch.tensor([0.0]).cuda()

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            img_tensor, target = batch[0].cuda(), batch[1].cuda()
            output = model(img_tensor)
            if isinstance(output, (tuple, list)):
                output = output[0]
            loss = criterion(output, target)
            val_epoch_loss += loss.item()
            predicted_classes = torch.max(output, dim = 1)[1]
            acc += (predicted_classes == target).sum()

    if dist.is_nccl_available():
        dist.all_reduce(acc, dist.ReduceOp.SUM)
        dist.all_reduce(val_epoch_loss, dist.ReduceOp.SUM)

    acc /= ( (i+1) * dist.get_world_size() )
    val_epoch_loss /= dist.get_world_size() 
    val_phase_results = {'Loss': val_epoch_loss, 'Accuracy' : acc.item()} 
    return val_phase_results

def main_worker(proc_index, args):

    torch.manual_seed(0)

    if torch.cuda.is_available():
        torch.cuda.set_device(proc_index)
    else:
        raise RuntimeError('CUDA not available!')

    if dist.is_nccl_available():
        dist.init_process_group(
            backend = 'nccl',
            world_size = args.gpus,
            rank = proc_index
        )
    else:
        raise RuntimeError('NCCL backend not available!')

    if args.task == 'ihc-score':
        args.num_classes = 4
    elif args.task == 'her2-status':
        args.num_classes = 2
    else:
        raise ValueError('Task should be ihc-score or her-status')

    if args.model == 'resnet34':
        model = resnet34(pretrained = False, num_classes = args.num_classes).cuda()
    elif args.model == 'abmil':
        model = ResnetABMIL(num_classes = args.num_classes).cuda()
    else:
        raise ValueError('Model should be resnet34 or abmil')  

    model = DistributedDataParallel(model, device_ids=[proc_index], output_device=proc_index)

    train_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
    ])

    train_df = pd.read_csv(args.train_csv)
    train_dataset = ImageDataset(train_df, fn_col = 'filename', lbl_col = args.task, transform = train_transform)
    if args.weighted_sampler_label == 'None':
        args.weighted_sampler_label = args.task
    weights = calculate_weights(torch.tensor(train_df[args.weighted_sampler_label].values))
    train_sampler = DistributedWeightedSampler(weights, num_replicas=args.gpus, rank=proc_index, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, sampler=train_sampler)

    if args.val_csv != 'None':
        val_transform = transforms.Compose([
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ToTensor(),
        ])
        val_df = pd.read_csv(args.val_csv)
        val_dataset = ImageDataset(val_df, fn_col = 'filename', lbl_col = args.task, transform = val_transform)
        val_sampler = DistributedSampler(val_dataset, num_replicas=args.gpus, rank=proc_index, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, sampler=val_sampler)
    
    optimizer = torch.optim.Adam(model.parameters(), lr = args.learning_rate, weight_decay=1e-8)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=args.scheduler_factor, patience=args.scheduler_patience, min_lr=1e-15)

    criterion = CrossEntropyLoss()

    epoch0 = 0
    epoch = epoch0
    while epoch < epoch0 + args.epochs:

        train_phase_results = train_step(train_loader, model, criterion, optimizer)
        val_phase_results = {'Loss': '', 'Accuracy' : ''} 
        if args.val_csv != 'None':
            val_phase_results = validate_step(val_loader, model, criterion)
            acc = val_phase_results['Accuracy']
            scheduler.step(acc)

        if (proc_index == 0): 
            print('Epoch {} finished.'.format(epoch))
            print('Train phase: ', train_phase_results)
            print('Val phase: ', val_phase_results)
            print('\n')

            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'accuracy': val_phase_results['Accuracy']

            }, os.path.join(args.checkpoints_dir,'checkpoint_{}.pth.tar'.format(epoch)))
        epoch += 1

def get_args():
    parser = argparse.ArgumentParser(description='Train HER2 overexpression classifier',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', dest='model', type=str, default='resnet34', help='resnet34 or abmil')
    parser.add_argument('--task', dest='task', type=str, default='her2-status', help='ihc-score or her2-status')
    parser.add_argument('--weighted_sampler_label', dest='weighted_sampler_label', type=str, default='None', help='Additional label in the train .csv to weight the sampling')
    parser.add_argument('--gpus', dest='gpus', type=int, default=4, help='Number of GPUs')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs', dest='epochs')
    parser.add_argument('--learning_rate', dest="learning_rate", type=float, nargs='?', default=0.001, help='Learning rate')
    parser.add_argument('--scheduler_factor', dest="scheduler_factor", type=float, nargs='?', default=0.1, help='Scheduler factor for decreasing learning rate')
    parser.add_argument('--scheduler_patience', dest="scheduler_patience", type=int, nargs='?', default=10, help='Scheduler patience for decreasing learning rate')
    parser.add_argument('--batch_size', type=int, nargs='?', default=32, help='Batch size', dest='batch_size')
    parser.add_argument('--train_csv', dest='train_csv', type=str, default='train.csv', help='.csv file containing the training examples')
    parser.add_argument('--val_csv', dest='val_csv', type=str, default='None', help='.csv file containing the val examples')
    parser.add_argument('--checkpoints_dir', dest='checkpoints_dir', type=str, default='./checkpoints', help='Path to save model checkpoints')
    parser.add_argument('--ip_address', dest='master_addr', type=str, default='localhost', help='IP address of rank 0 node')
    parser.add_argument('--port', dest='master_port', type=str, default='8888', help='Free port on rank 0 node')
    parser.add_argument('--num_workers', dest='num_workers', type=int, default=0, help='Number of workers for loading data')
    parser.add_argument('--img_size', dest='img_size', type=int, default=1024, help='Input image size for the model')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    os.makedirs(args.checkpoints_dir, exist_ok = True)
    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = args.master_port
    mp.spawn(main_worker, nprocs=args.gpus, args=(args,))
