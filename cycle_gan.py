import os
import shutil
import itertools

import torch
from torch.utils.data import DataLoader

import torchvision.transforms.v2 as transforms
from PIL import Image

from recap_folder.data_prep import ImageStyleDataset, init_cyclegan_transform
from recap_folder.model_build import CycleGANGenerator, CycleGANDiscriminator
from recap_folder.helper_funcs import *
from recap_folder.engine import train_cyclegan



### Execution of training phase ###

BATCH_SIZE = 1
EPOCH_NUMS = 50
IMAGE_SIZE = 256
IMAGE_SIZE_RESIZED = 286
LAMBDA = 10
LEARN_RATE = 2e-4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = ImageStyleDataset(
    style_dir="/kaggle/input/gan-getting-started/monet_jpg",
    image_dir="/kaggle/input/gan-getting-started/photo_jpg",
    transform=init_cyclegan_transform(IMAGE_SIZE, IMAGE_SIZE_RESIZED)
)

dloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

gen_g = CycleGANGenerator().to(device)
gen_f = CycleGANGenerator().to(device)
dsc_x = CycleGANDiscriminator().to(device)
dsc_y = CycleGANDiscriminator().to(device)

init_weights(gen_g)
init_weights(gen_f)
init_weights(dsc_x)
init_weights(dsc_y)

mae_fn = torch.nn.L1Loss().to(device)
loss_fn = torch.nn.BCEWithLogitsLoss().to(device)

opt_gen = torch.optim.Adam(
    params=itertools.chain(gen_g.parameters(), gen_f.parameters()),
    lr=LEARN_RATE,
    betas=(0.5, 0.999)
)

opt_dsc = torch.optim.Adam(
    params=itertools.chain(dsc_x.parameters(), dsc_y.parameters()),
    lr=LEARN_RATE,
    betas=(0.5, 0.999)
)

train_cyclegan(
    epochs=EPOCH_NUMS,
    train_dataloader=dloader,
    generators=(gen_g, gen_f),
    discriminators=(dsc_x, dsc_y),
    optimizers=(opt_gen, opt_dsc),
    loss_fn=loss_fn,
    mae_fn=mae_fn,
    mae_fn_lambda=LAMBDA,
    device=device
)



### Monet-Style Image Generation ###

photo_dir = "/kaggle/input/gan-getting-started/photo_jpg"
photo_paths = os.listdir(photo_dir)

tfm = transforms.Compose([
    transforms.ToImage(),
    transforms.ToDtype(torch.float32, scale=True),
    transforms.Normalize([0.5], [0.5])
])

os.makedirs("/kaggle/images/")
for i, path in enumerate(photo_paths):
    photo_path = os.path.join(photo_dir, path)
    photo_img = Image.open(photo_path)
    photo_tsr = tfm(photo_img).unsqueeze(0).to(device)

    gen_f.eval()
    with torch.no_grad(): op = gen_f(photo_tsr).squeeze(0).cpu().detach()

    op_unnormalized = op * 0.5 + 0.5
    op_image = transforms.ToPILImage()(op_unnormalized).convert("RGB")
    op_image.save("/kaggle/images/" + str(i+1) + ".jpg")

shutil.make_archive("/kaggle/working/images", "zip", "/kaggle/images")