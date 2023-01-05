import glob
import os
import time

import torch
from PIL import Image
from vizer.draw import draw_boxes
from ssd.config import cfg
from ssd.data.datasets import COCODataset, VOCDataset
import argparse
import numpy as np

from ssd.data.transforms import build_transforms
from ssd.modeling.detector import build_detection_model
from ssd.utils import mkdir
from ssd.utils.checkpoint import CheckPointer

from flask import Flask, request
import cv2
import numpy as np
import base64
import json

app = Flask(__name__)


@torch.no_grad()
def run_demo(cfg, ckpt, score_threshold, images_dir, output_dir, dataset_type):
    if dataset_type == "voc":
        class_names = VOCDataset.class_names
    elif dataset_type == 'coco':
        class_names = COCODataset.class_names
    else:
        raise NotImplementedError('Not implemented now.')
    # device = torch.device(cfg.MODEL.DEVICE)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_detection_model(cfg)
    model = model.to(device)
    checkpointer = CheckPointer(model, save_dir=cfg.OUTPUT_DIR)
    checkpointer.load(ckpt, use_latest=ckpt is None)
    weight_file = ckpt if ckpt else checkpointer.get_checkpoint_file()
    print('Loaded weights from {}'.format(weight_file))

    image_paths = glob.glob(os.path.join(images_dir, '*.jpg'))
    mkdir(output_dir)

    cpu_device = torch.device("cpu")
    transforms = build_transforms(cfg, is_train=False)
    model.eval()
    for i, image_path in enumerate(image_paths):
        start = time.time()
        image_name = os.path.basename(image_path)

        image = np.array(Image.open(image_path).convert("RGB"))
        height, width = image.shape[:2]
        images = transforms(image)[0].unsqueeze(0)
        load_time = time.time() - start

        start = time.time()
        result = model(images.to(device))[0]
        inference_time = time.time() - start

        result = result.resize((width, height)).to(cpu_device).numpy()
        boxes, labels, scores = result['boxes'], result['labels'], result['scores']

        indices = scores > score_threshold
        boxes = boxes[indices]
        labels = labels[indices]
        scores = scores[indices]
        meters = ' | '.join(
            [
                'objects {:02d}'.format(len(boxes)),
                'load {:03d}ms'.format(round(load_time * 1000)),
                'inference {:03d}ms'.format(round(inference_time * 1000)),
                'FPS {}'.format(round(1.0 / inference_time))
            ]
        )
        print('({:04d}/{:04d}) {}: {}'.format(i + 1, len(image_paths), image_name, meters))

        drawn_image = draw_boxes(image, boxes, labels, scores, class_names).astype(np.uint8)
        Image.fromarray(drawn_image).save(os.path.join(output_dir, image_name))

@app.route('/api', methods=['POST'])
def main():
    # 来捕捉客户端传来的数据
    data = request.get_data().decode('utf-8')
    # json.loads将string转换为dict
    data = json.loads(data)
    # 获取dict中'img'标签的数据
    image_b64 = data["img"]
    # 进行base64解码工作 base64->数组
    image_decode = base64.b64decode(image_b64)
    # fromstring实现了字符串到Ascii码的转换
    nparr = np.fromstring(image_decode, np.uint8)
    # 从nparr中读取数据，并把数据转换(解码)成图像格式
    img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    cv2.imwrite('./demo/test.jpg', img_np)


    parser = argparse.ArgumentParser(description="SSD Demo.")
    parser.add_argument(
        "--config-file",
        default="configs/vgg_ssd512_voc0712.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--ckpt", type=str, default='outputs/vgg_ssd512_voc0712/model_final.pth', help="Trained weights.")
    parser.add_argument("--score_threshold", type=float, default=0.7)
    parser.add_argument("--images_dir", default='demo', type=str, help='Specify a image dir to do prediction.')
    parser.add_argument("--output_dir", default='demo/result', type=str, help='Specify a image dir to save predicted images.')
    parser.add_argument("--dataset_type", default="voc", type=str, help='Specify dataset type. Currently support voc and coco.')

    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    print(args)

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    print("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        print(config_str)
    print("Running with config:\n{}".format(cfg))

    run_demo(cfg=cfg,
             ckpt=args.ckpt,
             score_threshold=args.score_threshold,
             images_dir=args.images_dir,
             output_dir=args.output_dir,
             dataset_type=args.dataset_type)

    with open("./demo/result/test.jpg", "rb") as f:  # 转为二进制格式
        base64_data = base64.b64encode(f.read())  # 使用base64进行加密

    return base64_data

if __name__ == '__main__':
    main:app.run('0.0.0.0',debug=True)
