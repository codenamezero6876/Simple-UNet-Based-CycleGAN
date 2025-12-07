import torch
from typing import Tuple

from recap_folder.helper_funcs import *
from recap_folder.model_build import CycleGANGenerator, CycleGANDiscriminator


def train_cyclegan(
    epochs: int,
    train_dataloader: torch.utils.data.DataLoader,
    generators: Tuple[CycleGANGenerator, CycleGANGenerator],
    discriminators: Tuple[CycleGANDiscriminator, CycleGANDiscriminator],
    optimizers: Tuple[torch.optim.Optimizer, torch.optim.Optimizer],
    loss_fn: torch.nn.Module,
    mae_fn: torch.nn.Module,
    mae_fn_lambda: float,
    device
):
    gen_g, gen_f = generators
    dsc_x, dsc_y = discriminators
    opt_gen, opt_dsc = optimizers

    gen_g.train()
    gen_f.train()
    dsc_x.train()
    dsc_y.train()

    for epoch in range(epochs):
        for i, (tru_style, tru_image) in enumerate(train_dataloader):
            tru_style = tru_style.to(device)
            tru_image = tru_image.to(device)

            set_requires_grad(dsc_x, False)
            set_requires_grad(dsc_y, False)

            opt_gen.zero_grad()

            gen_image = gen_g(tru_style)
            gen_style = gen_f(tru_image)

            cyc_style = gen_f(gen_image)
            cyc_image = gen_g(gen_style)

            sme_style = gen_f(tru_style)
            sme_image = gen_g(tru_image)

            dsc_gen_style = dsc_x(gen_style)
            dsc_gen_image = dsc_y(gen_image)

            total_gen_loss = (
                calc_cyc_loss(cyc_style, tru_style, mae_fn, mae_fn_lambda) +
                calc_cyc_loss(cyc_image, tru_image, mae_fn, mae_fn_lambda) +
                calc_idt_loss(sme_style, tru_style, mae_fn, 0.5*mae_fn_lambda) +
                calc_idt_loss(sme_image, tru_image, mae_fn, 0.5*mae_fn_lambda) +
                calc_gen_loss(dsc_gen_style, loss_fn, device) +
                calc_gen_loss(dsc_gen_image, loss_fn, device)
            )

            total_gen_loss.backward()
            opt_gen.step()

            gen_style = gen_style.detach().clone().requires_grad_(True)
            gen_image = gen_image.detach().clone().requires_grad_(True)

            set_requires_grad(dsc_x, True)
            set_requires_grad(dsc_y, True)

            opt_dsc.zero_grad()

            dsc_tru_style = dsc_x(tru_style)
            dsc_tru_image = dsc_y(tru_image)
            dsc_gen_style = dsc_x(gen_style)
            dsc_gen_image = dsc_y(gen_image)

            total_dsc_loss = (
                calc_dsc_loss(dsc_gen_style, dsc_tru_style, loss_fn, device) +
                calc_dsc_loss(dsc_gen_image, dsc_tru_image, loss_fn, device)
            )

            total_dsc_loss.backward()
            opt_dsc.step()

            if i % 30 == 0:
                print(f"[INFO] Epoch: {(epoch+1):03d} | Gen Loss: {total_gen_loss:.4f} | Dsc Loss: {total_dsc_loss:.4f}")