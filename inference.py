import argparse
import os

import torch
from torch.utils.data import DataLoader
from torchvision.models import resnet34
from torchvision import transforms
import pandas as pd
from torch.nn.functional import softmax

from src.utils import load_model_without_ddp
from src.dataset import ImageDataset
from src.models import ResnetABMIL

def get_args():
    parser = argparse.ArgumentParser(description='Test HER2 overexpression classifier',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', dest='model', type=str, default='resnet34', help='resnet34 or abmil')
    parser.add_argument('--task', dest='task', type=str, default='her2-status', help='ihc-score or her2-status')
    parser.add_argument('--batch_size', type=int, nargs='?', default=32, help='Batch size', dest='batch_size')
    parser.add_argument('--test_csv', dest='test_csv', type=str, default='test.csv', help='.csv file containing the test examples')
    parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', type=str, default='./checkpoints/checkpoint_99.pth.tar', help='Path to load the model checkpoint from')
    parser.add_argument('--out_dir', dest='out_dir', type=str, default='./out', help='Path to save the inference results')
    parser.add_argument('--num_workers', dest='num_workers', type=int, default=0, help='Number of workers for loading data')
    parser.add_argument('--img_size', dest='img_size', type=int, default=1024, help='Input image size for the model')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    os.makedirs(args.out_dir, exist_ok = True)

    torch.manual_seed(0)

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA not available!')

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

    model = load_model_without_ddp(model, args.checkpoint_dir)

    test_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
    ])

    test_df = pd.read_csv(args.test_csv)
    fn_col = 'filename'

    test_dataset = ImageDataset(test_df, fn_col = fn_col, lbl_col = args.task, transform = test_transform, return_filename = True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)

    predictions_total = []
    targets_total = []
    filenames_total = []
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            img_tensor, target, fn = batch
            img_tensor, target = img_tensor.cuda(), target.cuda()
            output = model(img_tensor)
            if isinstance(output, (tuple, list)):
                output = output[0]
            targets_total.append(target)
            predictions_total.append(softmax(output, dim = 1))
            filenames_total.append(fn)

    predictions_total = torch.cat(predictions_total, 0).cpu().detach().numpy()
    targets_total = torch.cat(targets_total, 0).cpu().detach().numpy()
    filenames_total = [fn for fns in filenames_total for fn in fns]

    targets_series = pd.Series(data = targets_total, name = 'true_class')
    filenames_series = pd.Series(data = filenames_total, name = 'filename')

    series_list = [pd.Series( data = predictions_total[:,i], name = 'prob_class_{}'.format(i) ) for i in range(predictions_total.shape[1])] \
        + [targets_series] \
        + [filenames_series]

    dataframe = pd.concat(series_list, axis = 1)
    dataframe.to_csv(os.path.join(args.out_dir,'inference_out.csv' ), index = False )
