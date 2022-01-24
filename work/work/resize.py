import argparse
import glob
import os.path
from transforms import Normalize, Resize
from PIL import Image
from transforms import Compose, Compose_test
import paddle
import pickle
import paddle.nn as nn
import paddlehub as hub
import cv2
import numpy as np
from src.config import Config
from main import set_model_log_output_dir
from src.MRTR import MRTR
import time
import os
sr_model = hub.Module(name='falsr_c')
with open('/home/aistudio/work/map.pickle','rb') as f:
    w_h_map = pickle.load(f) 

    print('Now Start to generate the super resolution image....')
    start = time.time()
    img_list = glob.glob('/home/aistudio/work/test_result/*')

    # for path in img_list:
    #     sr_model.predict(path)
    sr_model.reconstruct(paths=img_list,output_dir="/home/aistudio/work/test_result/dcscn_output/",use_gpu=True,visualization=True)
    end = time.time()
    time_two = (end - start) / 200
    print(f"The super resolution part of per image time cost is {time_two}s")

    print('Now resize image to orignial size')

    img_list = glob.glob('/home/aistudio/work/test_result/dcscn_output/*')

    start = time.time()
    for path in img_list:
        img = cv2.imread(path)
        resized = cv2.resize(img, w_h_map[path.replace('dcscn_output/','')],interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(path, resized)
    end = time.time()
    time_three = (end - start) / 200
    print(f"Resize time is {time_three}s")