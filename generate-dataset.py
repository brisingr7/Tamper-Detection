import os
import argparse
import json
import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
import numpy as np
from PIL import Image
from models import CompletionNetwork
from utils import poisson_blend, gen_input_mask


parser = argparse.ArgumentParser()
parser.add_argument('model')
parser.add_argument('config')
parser.add_argument('--max_holes', type=int, default=1)
parser.add_argument('--img_size', type=int, default=240)
parser.add_argument('--hole_min_w', type=int, default=10)
parser.add_argument('--hole_max_w', type=int, default=16)
parser.add_argument('--hole_min_h', type=int, default=10)
parser.add_argument('--hole_max_h', type=int, default=16)


def main(args):

    args.model = os.path.expanduser(args.model)
    args.config = os.path.expanduser(args.config)

    # =============================================
    # Load model
    # =============================================
    with open(args.config, 'r') as f:
        config = json.load(f)
    mpv = torch.tensor(config['mpv']).view(1, 3, 1, 1)
    model = CompletionNetwork()
    model.load_state_dict(torch.load(args.model, map_location='cpu'))
    model.to('cuda')

    # =============================================
    # Predict
    # =============================================
    # convert img to tensor
    rootdir = './indoorCVPR_09/Images'
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            try:
                print(os.path.join(subdir, file))
                img = Image.open(os.path.join(subdir, file))
                img = transforms.Resize(args.img_size)(img)
                img = transforms.RandomCrop(
                    (args.img_size, args.img_size))(img)
                x = transforms.ToTensor()(img)
                x = torch.unsqueeze(x, dim=0)

                # create mask
                mask = gen_input_mask(
                    shape=(1, 1, x.shape[2], x.shape[3]),
                    hole_size=(
                        (args.hole_min_w, args.hole_max_w),
                        (args.hole_min_h, args.hole_max_h),
                    ),
                    max_holes=args.max_holes,
                )

                # inpaint
                model.eval()
                with torch.no_grad():
                    x_mask = x - x * mask + mpv * mask
                    input = torch.cat((x_mask, mask), dim=1)
                    input = input.to('cuda')
                    output = model(input)
                    inpainted = poisson_blend(x_mask, output, mask)
                    # imgs = torch.cat((x, x_mask, inpainted), dim=0)
                    save_image(x, './dataset/input/' + file)
                    save_image(mask, './dataset/mask/' + file)
                    save_image(inpainted, './dataset/output/' + file)
                    # save_image(imgs, args.output_img, nrow=3)
            except:
                continue


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
