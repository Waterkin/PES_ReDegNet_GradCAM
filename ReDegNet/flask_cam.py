import io
import json
import warnings
from torchvision import models
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, jsonify, request, send_file, send_from_directory

app = Flask(__name__)  # 固定写法
app.config["UPLOAD_FOLDER"] = "/dev_data/wlh/PES/"

import cv2
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch
from pytorch_grad_cam import (
    GradCAM,
    HiResCAM,
    ScoreCAM,
    GradCAMPlusPlus,
    AblationCAM,
    XGradCAM,
    EigenCAM,
    FullGrad,
)
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image

# from torchvision.models import resnet50
from torchvision.datasets import CIFAR10
from networks.ResNet import PreActResNet18
import os
import numpy as np
from numpy import asarray

# torch.cuda.set_device(torch.device("cuda:1"))
from PIL import Image
from torch.autograd import Variable
import subprocess

from ISR.models import RDN, RRDN

warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 没用


def gradcam(path, noisy_type="symmetric"):
    model = PreActResNet18(10)
    model.cuda()
    model.load_state_dict(torch.load(f"{noisy_type}.pth"))
    model.eval()
    # print(model)
    target_layers = [model.layer4[-1]]
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    invTrans = transforms.Compose(
        [
            transforms.Normalize(
                mean=[0.0, 0.0, 0.0], std=[1 / 0.2023, 1 / 0.1994, 1 / 0.2010]
            ),
            transforms.Normalize(mean=[-0.4914, -0.4822, -0.4465], std=[1.0, 1.0, 1.0]),
        ]
    )

    rgb_img = Image.open(f"{path}")
    images = transform_test(rgb_img).unsqueeze(0)
    images = images.cuda()
    targets = None
    cam.batch_size = 1
    grayscale_cam = cam(input_tensor=images, targets=targets)
    print(grayscale_cam.shape)

    grayscale_cam = grayscale_cam[0, :]
    input_images = asarray(rgb_img)
    input_images = np.float32(input_images) / 255
    visualization = show_cam_on_image(input_images, grayscale_cam, use_rgb=True)

    path = path.split("/")[-1]

    # image-super-resolution
    # rdn = RDN(
    #    arch_params={"C": 3, "D": 10, "G": 64, "G0": 64, "x": 2}
    # )  # weights="psnr-small"
    # rdn = RRDN(weights="gans")
    # rdn.model.load_weights(
    #    "image-super-resolution/weights/sample_weights/rdn-C6-D20-G64-G064-x2/PSNR-driven/rdn-C6-D20-G64-G064-x2_PSNR_epoch086.hdf5"
    # )
    # visualization = rdn.predict(visualization)
    # Image.fromarray(sr_img)

    cv2.imwrite(f"pic/{path}", visualization)

    os.system(
        f"cp pic/{path} /dev_data/wlh/PES/ReDegNet/CleanTest/testsets/my/pic.jpg  && cd ReDegNet/CleanTest && CUDA_VISIBLE_DEVICES=1 python test_restoration.py -i my && cd ../.. && cp /dev_data/wlh/PES/ReDegNet/CleanTest/testsets/my/pic.jpg pic/{path}"
    )  # 超分辨率+去噪
    # send files to remote
    s = subprocess.Popen(
        [
            "scp",
            f"pic/{path}",
            "wlh@202.38.247.12:/dev_data_2/wlh/Water-MAML/test_flask/",
        ]
    )
    sts = os.waitpid(s.pid, 0)

    return send_file(f"pic/{path}")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":  # 接收传输的图片, 这里传的是路径
        image_file = request.files["file"]  # .args.get("file")  #
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], image_file.filename)
        image_file.save(file_path)
    # zxy add for GET
    else:
        file_path = request.args.get("path")  # 接收其他客户端浏览器发送的请求
        # file = open(image_file, "rb")
    # img_bytes = file.read()
    return gradcam(file_path)
    # return gradcam(image_file) # 目前只读本机图片先


if __name__ == "__main__":
    # app.run() # 原工程的写法，默认只能本机访问
    app.run(host="0.0.0.0", port=5005)  # 使其他主机可以访问服务


# #从外部主机发送图片到服务器，并接收返回结果
# curl -X POST -F 'file=@/dev_data_2/wlh/Water-MAML/grad-cam/dog/origin/batch_3_num_4503.jpg' --output 'test.jpg' http://116.56.134.107:5005/predict

# # 从浏览器发出请求，图片在服务端本地
# http://116.56.134.107:5005/predict?path=/dev_data/wlh/PES/data/cifar-10-batches-py/train/dog/batch_3_num_4503.jpg


# References:
# https://www.tutorialspoint.com/How-to-copy-a-file-to-a-remote-server-in-Python-using-SCP-or-SSH
# superweb999.com/article/356190.html
# https://cloud.tencent.com/developer/article/1669557
# https://blog.theodo.com/2022/05/upgrade-pytorch-for-aws-sagemaker/
# https://www.thegeekdiary.com/how-to-run-scp-without-password-prompt-interruption-in-linux/
# https://flask.palletsprojects.com/en/2.2.x/patterns/fileuploads/
# https://blog.csdn.net/qq_27825451/article/details/102909772
