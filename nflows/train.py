# %% [markdown]
# # Glow

# %%
from pathlib import Path

# Import required packages
import torch
import torchvision as tv
import numpy as np
import normflows as nf

from matplotlib import pyplot as plt
from tqdm import tqdm

# from data import OceanData
# %%
from nflows import models
from nflows import data

# Move model on GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 2

# train_ds, test_ds, num_classes, input_shape = data.get_ocean_dataset(
#     data_dir="/mnt/vol_b/datasets/oceans_small_320_320"
# )

# train_ds, test_ds, num_classes, input_shape = data.get_bubbles_dataset(
#     data_dir="/mnt/vol_b/datasets/duck_in_tub"
# )

# train_ds, test_ds, num_classes, input_shape = data.get_flowers_102(batch_size=batch_size)
# train_ds, test_ds, num_classes, input_shape = data.get_stl10(batch_size=8)
train_ds, test_ds, num_classes, input_shape = data.get_mnist(batch_size=32)

model = models.get_glow_model(num_classes=num_classes, input_shape=input_shape).to(device)
# model.load("model_4.pth")

loss_hist = np.array([])

optimizer = torch.optim.Adamax(model.parameters(), lr=1e-3, weight_decay=1e-5)
epochs = 200
for epoch in range(epochs):
    for sample in tqdm(train_ds):

        if isinstance(sample, dict):
            x, y = sample["image"], sample["label"]
        else:
            x, y = sample
        optimizer.zero_grad()
        loss = model.forward_kld(x.to(device), y.to(device))

        if not (torch.isnan(loss) | torch.isinf(loss)):
            loss.backward()
            optimizer.step()

        loss_hist = np.append(loss_hist, loss.detach().to("cpu").numpy())
        del (x, y, loss)
    num_sample = 5
    model.save(f"model_{epoch}.pth")
    # num_classes = 2
    with torch.no_grad():
        y = torch.arange(num_classes).repeat(num_sample).to(device=device)
        x, _ = model.sample(y=y)
        x_ = torch.clamp(x, 0, 1)
        plt.figure(figsize=(10, 10))
        plt.imshow(
            np.transpose(tv.utils.make_grid(x_, nrow=num_classes).cpu().numpy(), (1, 2, 0))
        )
        plt.savefig(f"examples_{epoch}.png")


    del (x, y, x_)

plt.figure(figsize=(10, 10))
plt.plot(loss_hist, label="loss")
plt.legend()
plt.savefig("training loss.png")
from random import choices

plt.figure(figsize=(10, 10))

num_classes = 2
train_ds_iter = iter(train_ds)
sample_truth_images = [next(train_ds_iter)[0][0] for o in  range(num_sample)]
x_ = torch.stack(sample_truth_images)
plt.imshow(
    np.transpose(tv.utils.make_grid(x_, nrow=num_classes).cpu().numpy(), (1, 2, 0))
)
plt.savefig("examples_true.png")
# %%
# # Prepare training data
# batch_size = 2
# data_path = "/mnt/vol_b/datasets/"
# transform = tv.transforms.Compose([tv.transforms.ToTensor(), nf.utils.Scale(255. / 256.), nf.utils.Jitter(1 / 256.), tv.transforms.Resize((320,320))])
# # train_data = tv.datasets.CIFAR10(data_path, train=True,
# #                                  download=True, transform=transform)
# train_data = OceanData(Path(data_path)/"oceans_small_320_320", train=True,
#                                   transform=transform)

# test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)

# train_iter = iter(train_loader)
# # x, y = next(train_iter)

# # %%
# # y

# # %%
# # Train model
# max_iter = 1000


# for i in tqdm(range(max_iter)):
#     try:
#         x, y = next(train_iter)
#     except StopIteration:
#         train_iter = iter(train_loader)

#         x, y = next(train_iter)
#     optimizer.zero_grad()
#     loss = model.forward_kld(x.to(device), y.to(device))

#     if ~(torch.isnan(loss) | torch.isinf(loss)):
#         loss.backward()
#         optimizer.step()

#     loss_hist = np.append(loss_hist, loss.detach().to('cpu').numpy())
#     del(x, y, loss)

# plt.figure(figsize=(10, 10))
# plt.plot(loss_hist, label='loss')
# plt.legend()
# plt.show()

# # %%
# # Model samples
# num_sample = 1

# with torch.no_grad():
#     y = torch.arange(num_classes).repeat(num_sample).to(device)
#     x, _ = model.sample(y=y)
#     x_ = torch.clamp(x, 0, 1)
#     plt.figure(figsize=(10, 10))
#     plt.imshow(np.transpose(tv.utils.make_grid(x_, nrow=num_classes).cpu().numpy(), (1, 2, 0)))
#     plt.savefig("examples.png")

#     del(x, y, x_)


# # %%
# # # Get bits per dim
# # n = 0
# # bpd_cum = 0
# # with torch.no_grad():
# #     for x, y in iter(test_loader):
# #         nll = model(x.to(device), y.to(device))
# #         nll_np = nll.cpu().numpy()
# #         bpd_cum += np.nansum(nll_np / np.log(2) / n_dims + 8)
# #         n += len(x) - np.sum(np.isnan(nll_np))

# #     print('Bits per dim: ', bpd_cum / n)
