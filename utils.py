import matplotlib.pyplot as plt
import torch


def get_current_work_dir():
    import os

    print("The current work dir is:")
    print(os.getcwd())


def check_gpu():
    print("Is CUDA available?", torch.cuda.is_available())
    print("Device names:")
    print("cuda:0", torch.cuda.get_device_name(0))
    try:
        print("cuda:1", torch.cuda.get_device_name(1))
    except:
        pass


def check_cpu():
    import multiprocessing

    print("Available CPU cores", multiprocessing.cpu_count())


def check_installed_packages():
    import subprocess
    import sys
    import pprint

    pp = pprint.PrettyPrinter(depth=6)
    reqs = subprocess.check_output([sys.executable, "-m", "pip", "freeze"])
    installed_packages = [r.decode().split("==")[0] for r in reqs.split()]
    print("Installed packages in environment:")
    pp.pprint(installed_packages)


def check_available_memory():
    import psutil

    print(
        "Available memory:", round(psutil.virtual_memory().available / 1000000000), "GB"
    )


def check_available_disk_space():
    import psutil

    obj_Disk = psutil.disk_usage("/")

    print("Total disc space:", round(obj_Disk.total / (1024.0 ** 3)), "GB")
    print("Used disc space:", round(obj_Disk.used / (1024.0 ** 3)), "GB")
    print("Free disc space:", round(obj_Disk.free / (1024.0 ** 3)), "GB")


def check_system_info():
    get_current_work_dir()
    check_cpu()
    check_available_memory()
    check_available_disk_space()
    check_gpu()
    check_installed_packages()


def print_image(image):
    print("image: ", image)
    plt.imshow(image.permute(1, 2, 0))
    plt.show()


def print_sample(sample, tabular=False, image=False):
    if image and tabular:
        image, tabular, y = sample
        print("tabular data: ", tabular)
        print("y: ", y)
        print_image(image)
    elif tabular:
        tabular, y = sample
        print("tabular data: ", tabular)
        print("y: ", y)
    elif image:
        image, y = sample
        print("y: ", y)
        print_image(image)


def child_counter(model):
    child_counter = 0
    for child in model.children():
        print(" child", child_counter, "is:")
        print(child)
        child_counter += 1


def convert_coordinates(x, y, in_proj="epsg:25833", out_proj="epsg:4326"):
    from pyproj import Proj, transform

    in_proj = Proj(in_proj)
    out_proj = Proj(out_proj)

    print(transform(in_proj, out_proj, x, y))


def annot_min(
    x,
    y,
    ax=None,
    x_label="epoch",
    y_label="loss",
    xytext=(0.94, 0.96),
    arrowcolor="black",
    edgecolor="k",
    textcolor="black"
):
    import numpy as np

    xmin = x[np.argmin(y)]
    ymin = min(y)
    text = f"{x_label}={xmin}, {y_label}={ymin:.3f}"
    if not ax:
        ax = plt.gca()
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec=edgecolor, lw=0.72)
    arrowprops = dict(
        arrowstyle="->", connectionstyle="angle,angleA=0,angleB=90", color=arrowcolor
    )
    kw = dict(
        xycoords="data",
        textcoords="axes fraction",
        arrowprops=arrowprops,
        bbox=bbox_props,
        ha="right",
        va="top",
    )
    ax.annotate(text, xy=(xmin, ymin), xytext=xytext, color=textcolor, **kw)

