import os
import argparse

from core.data_loader import CelebaMultiLabelDataset
from core.solver import Solver

from PIL import Image

import torch
from torch.utils import data
from torchvision import transforms

class CelebaDS4Metric(CelebaMultiLabelDataset):
    def __init__(self, root, labels, transform=None):
        super(CelebaDS4Metric, self).__init__(root, labels, transform)

    def __getitem__(self, index):
        fname = self.samples[index]
        labels = self.targets[index]
        img = Image.open(fname).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, labels

class Solver4Gen(Solver):
    def __init__(self, args):
        super(Solver4Gen, self).__init__(args)

    def generate(self, img, style):
        args = self.args

        nets_ema = self.nets_ema
        masks = nets_ema.fan(img)
        newimg = nets_ema.generator(img, style, masks)

    def get_style(self, img, label):
        return self.nets_ema.style_encoder(img, label)
        

def get_data_loader(args):
    transform = transforms.Compose([
        transforms.Resize([args.img_size, args.img_size]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5]),
        ])

    val_ds = CelebaDS4Metric(args.val_img_dir, args.domains, transform)

    #Possibility of increasing batch size and num_workers
    return data.DataLoader(dataset=val_ds, batch_size=1, shuffle=False,
                           num_workers=1, pin_memory=True)


def main(args):
    #GPUS are occupied at the moment, so....
    os.environ['CUDA_VISIBLE_DEVICES'] = ''

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    val_loader1 = get_data_loader(args)
    val_loader2 = get_data_loader(args)
    solver = Solver4Gen(args)

    count = 0
    for _, (img1, labels1) in enumerate(val_loader1):
        img1 = img1.unsqueeze(0).to(device)
        for _, (img2, labels2) in enumerate(val_loader2):
            #Muti-labels go with generator multiple times
            label = torch.LongTensor(1).to(device)
            label[0] = 0 #

            style = solver.get_style(img1, label)
            img2 = img2.unsqueeze(0).to(device)
            solver.generate(img2, style)
            break

           
            
 
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--img_size', type=int, default=256,
                        help='Image resolution')
    parser.add_argument('--val_img_dir', type=str, default='data/celeba_hq/train',
                        help='Directory containing validation images')
    parser.add_argument('--domains', nargs='+',
                        default=["Male", "Smiling", "Pale_Skin"])
    parser.add_argument('--style_dim', type=int, default=64,
                        help='Style code dimension')
    parser.add_argument('--w_hpf', type=float, default=1,
                        help='weight for high-pass filtering')
    parser.add_argument('--latent_dim', type=int, default=16,
                        help='Latent vector dimension')
    parser.add_argument('--multi_gpus', action="store_true")
    parser.add_argument('--wing_path', type=str, default='expr/checkpoints/wing.ckpt')
    parser.add_argument('--mode', type=str, default='generate')
    parser.add_argument('--checkpoint_dir', type=str, default='/home/data/gan_checkpoints',
                        help='Directory for saving network checkpoints')
    args = parser.parse_args()

    args.num_domains = 2 * len(args.domains)

    main(args)

