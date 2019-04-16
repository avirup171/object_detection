from __future__ import print_function
import sys
import os
from argparse import ArgumentParser
import cv2
import time
import logging as log
import numpy as np
import io
import detect as dt
from openvino.inference_engine import IENetwork, IEPlugin
from pathlib import Path
sys.path.insert(0, str(Path().resolve().parent.parent))



def build_argparser():
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", help="Path to an .xml file with a trained model.", required=True, type=str)
    parser.add_argument("-i", "--input",
                        help="Path to video file or image. 'cam' for capturing video stream from camera",
                        type=str)
    parser.add_argument("-l", "--cpu_extension",
                        help="MKLDNN (CPU)-targeted custom layers.Absolute path to a shared library with the kernels "
                             "impl.", type=str, default=None)
    parser.add_argument("-pp", "--plugin_dir", help="Path to a plugin folder", type=str, default=None)
    parser.add_argument("-d", "--device",
                        help="Specify the target device to infer on; CPU, GPU, FPGA, MYRIAD or HDDL is acceptable. Sample "
                             "will look for a suitable plugin for device specified (CPU by default)", default="CPU",
                        type=str)
    parser.add_argument("--labels", help="Labels mapping file", default=None, type=str)
    parser.add_argument("-pt", "--prob_threshold", help="Probability threshold for detections filtering",
                        default=0.5, type=float)
    parser.add_argument("-o", "--output_dir", help="If set, it will write a video here instead of displaying it",
                        default=None, type=str)
    return parser

def make_sure_path_exists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        pass

 
def main():
    is_async_mode = True
    args = build_argparser().parse_args()
    object_detection=dt.Detectors(args.device,args.model,args.cpu_extension,args.plugin_dir,is_async_mode)
    resultant_initialisation_object=object_detection.initialise_inference()
    input_stream = args.input
    
    #Start video capturing process
    if args.input==None:
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(input_stream)
    #Frame count
    video_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    out = cv2.VideoWriter("out_path.mp4", 0x00000021, 50.0, (frame_width, frame_height), True)
    cur_request_id = 0
    next_request_id = 1

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            initial_w = cap.get(3)
            initial_h = cap.get(4)
            res_inference=resultant_initialisation_object.process_frame(cur_request_id,next_request_id,frame,initial_h,initial_w,False)
            resultant_frame=resultant_initialisation_object.placeBoxes(res_inference,None,0.5,frame,initial_w,initial_h,False,cur_request_id)
            #out.write(resultant_frame)
            cv2.imshow('frame',resultant_frame)
            key = cv2.waitKey(1)
            if key == 27:
                break
        #out.release()
        cap.release()
    finally:
        del resultant_initialisation_object.exec_net


if __name__ == '__main__':
    sys.exit(main() or 0)
