import torch


def set_requires_grad(model, requires_grad):
    for p in model.parameters(): p.requires_grad_(requires_grad)


def init_weights(net, gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, "weight") and (classname.find("Conv") != -1 or classname.find("Linear") != -1):
            torch.nn.init.normal_(m.weight.data, 0.0, gain)
            if hasattr(m, "bias") and m.bias is not None:
                torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find("BatchNorm2d") != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, gain)
            torch.nn.init.constant_(m.bias.data, 0.0)
    net.apply(init_func)


def calc_gen_loss(gen_values, loss_fn, device="cpu"):
    labels = torch.FloatTensor(gen_values.shape).fill_(1.0).to(device)
    labels.requires_grad_(False)
    return loss_fn(gen_values, labels)


def calc_dsc_loss(gen_values, tru_values, loss_fn, device="cpu"):
    gt_labels = torch.FloatTensor(tru_values.shape).fill_(1.0).to(device)
    fk_labels = torch.FloatTensor(gen_values.shape).fill_(0.0).to(device)
    gt_labels.requires_grad_(False)
    fk_labels.requires_grad_(False)
    gt_loss = loss_fn(tru_values, gt_labels)
    fk_loss = loss_fn(gen_values, fk_labels)
    return (gt_loss + fk_loss) / 2


def calc_cyc_loss(cyc_values, tru_values, loss_fn, factor):
    return factor * loss_fn(cyc_values, tru_values)


def calc_idt_loss(sme_values, tru_values, loss_fn, factor):
    return factor * loss_fn(sme_values, tru_values)