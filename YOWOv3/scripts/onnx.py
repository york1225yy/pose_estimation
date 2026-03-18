from model.TSN.YOWOv3 import build_yowov3
from cus_datasets.build_dataset import build_dataset
from utils.box import non_max_suppression

import torch
from utils.box import draw_bounding_box
import cv2
import numpy as np

def export2onnx(config):
    model   = build_yowov3(config) 
    model.eval()

    dummy_input = torch.randn(1, 3, 16, 224, 224)

    torch.onnx.export(model,
                    dummy_input,
                    "yowov3.onnx",
                    verbose=False,
                    input_names=['clip'],
                    output_names=['image'],
                    export_params=True)
    
    
    print("ONNX export complete: {}".format(onnx_model_path))
    print("To run inference, use export_onnx.py or onnxruntime directly.")