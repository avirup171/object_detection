from __future__ import print_function
import sys
import os
from argparse import ArgumentParser
import cv2
import time
import logging as log
import numpy as np
import io
from openvino.inference_engine import IENetwork, IEPlugin
from pathlib import Path
sys.path.insert(0, str(Path().resolve().parent.parent))

class Detectors(object):
    #Constructor
    def __init__(self, device, model, cpu_extension, plugin_dir,is_async_mode):
        self.cpu_extension = cpu_extension
        self.plugin_dir = plugin_dir
        self.is_async_mode = is_async_mode
        self.device=device
        self.model=model

    def initialise_inference(self):
        
        model_xml=self.model
        model_bin = os.path.splitext(model_xml)[0] + ".bin"

        # Plugin initialization for specified device and load extensions library if specified
        log.info("Initializing plugin for {} device...".format(self.device))
        plugin = IEPlugin(device=self.device, plugin_dirs=self.plugin_dir)
        if self.cpu_extension and 'CPU' in self.device:
            log.info("Loading plugins for {} device...".format(self.device))
            plugin.add_cpu_extension(self.cpu_extension)
        
        # Read IR
        log.info("Reading IR...")
        net = IENetwork(model=model_xml, weights=model_bin)

        if plugin.device == "CPU":
            supported_layers = plugin.get_supported_layers(net)
            not_supported_layers = [l for l in net.layers.keys() if l not in supported_layers]
            if len(not_supported_layers) != 0:
                log.error("Following layers are not supported by the plugin for specified device {}:\n {}".
                        format(plugin.device, ', '.join(not_supported_layers)))
                log.error("Please try to specify cpu extensions library path in sample's command line parameters using -l "
                        "or --cpu_extension command line argument")
                sys.exit(1)
        

        assert len(net.inputs.keys()) == 1, "Sample supports only single input topologies"
        assert len(net.outputs) == 1, "Sample supports only single output topologies"
        
        input_blob = next(iter(net.inputs))
        out_blob = next(iter(net.outputs))
        log.info("Loading IR to the plugin...")
        exec_net = plugin.load(network=net, num_requests=2)

        if isinstance(net.inputs[input_blob], list):
            n, c, h, w = net.inputs[input_blob]
        else:
            n, c, h, w = net.inputs[input_blob].shape
        del net
        processor= Processor(exec_net,input_blob,out_blob,n,c,h,w)
        return processor


class Processor(object):
    #Constructor
    def __init__(self,exec_net,input_blob,out_blob,n,c,h,w):
        self.exec_net=exec_net
        self.input_blob=input_blob
        self.out_blob=out_blob
        self.n=n
        self.c=c
        self.h=h
        self.w=w

    def process_frame(self,cur_request_id,next_request_id,frame,frame_height,frame_width,is_async_mode):
              
        in_frame = cv2.resize(frame, (self.w, self.h))
        in_frame = in_frame.transpose((2, 0, 1))  # Change data layout from HWC to CHW
        in_frame = in_frame.reshape((self.n, self.c, self.h, self.w))
        
        if is_async_mode:
            self.exec_net.start_async(request_id=next_request_id, inputs={self.input_blob: in_frame})
        else:
            self.exec_net.start_async(request_id=cur_request_id, inputs={self.input_blob: in_frame})
        
        if self.exec_net.requests[cur_request_id].wait(-1) == 0:
            #Parse detection results of the current request
            res = self.exec_net.requests[cur_request_id].outputs[self.out_blob]
            return res

    
    def placeBoxes(self,res, labels_map, prob_threshold, frame, initial_w, initial_h, is_async_mode, cur_request_id):
        for obj in res[0][0]:
            # Draw only objects when probability more than specified threshold
            if obj[2] > prob_threshold:
                xmin = int(obj[3] * initial_w)
                ymin = int(obj[4] * initial_h)
                xmax = int(obj[5] * initial_w)
                ymax = int(obj[6] * initial_h)
                class_id = int(obj[1])
                # Draw box and label\class_id
                '''
                inf_time_message = "Inference time: N\A for async mode" if is_async_mode else \
                        "Inference time: {:.3f} ms".format(det_time * 1000)
                async_mode_message = "Async mode is on. Processing request {}".format(cur_request_id) if is_async_mode else \
                        "Async mode is off. Processing request {}".format(cur_request_id)
                '''
                color = (min(class_id * 12.5, 255), min(class_id * 7, 255), min(class_id * 5, 255))
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
                det_label = labels_map[class_id] if labels_map else str(class_id)
                cv2.putText(frame, det_label + ' ' + str(round(obj[2] * 100, 1)) + ' %', (xmin, ymin - 7), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0,0,255), 1)
                #cv2.putText(frame, inf_time_message, (15, 15), cv2.FONT_HERSHEY_COMPLEX, 0.5, (200, 10, 10), 1)
                #cv2.putText(frame, async_mode_message, (10, int(initial_h - 20)), cv2.FONT_HERSHEY_COMPLEX, 0.5,(10, 10, 200), 1)

        return frame





