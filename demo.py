from toolkit.got10k.experiments import *
from tracker.trackerRTM import TrackerSiamRTM

import argparse
import os
import json
import numpy as np

import cv2

from utils.torchvotrt import noise_handler 

parser = argparse.ArgumentParser(description='SiamRTM demo')

parser.add_argument('video', help='path to demo video')
parser.add_argument('--config', default='./configs/parameters.json')
parser.add_argument('--checkpoint', default='./checkpoints/model_RTM.pth', metavar='DIR',help='path to weight')
parser.add_argument('--show', default=False, help='visualize tracking results')
parser.add_argument('--distortion', default='original', choices=['original', 'WGN', 'SnP', 'GB'], 
                    help='choose image distortion (original -> no distortion)')



selectingObject = True
ix, iy, cx, cy = -1, -1, -1, -1
w, h = 0, 0

def draw_boundingbox(event, x, y, flags, param):
	global selectingObject, init_tracker, onTracking, ix, iy, cx,cy, w, h
	
	if event == cv2.EVENT_LBUTTONDOWN:
		selectingObject = True
		onTracking = False
		ix, iy = x, y
		cx, cy = x, y
	
	elif event == cv2.EVENT_MOUSEMOVE:
		cx, cy = x, y
	
	elif event == cv2.EVENT_LBUTTONUP:
		selectingObject = False
		if(abs(x-ix)>10 and abs(y-iy)>10):
			w, h = abs(x - ix), abs(y - iy)
			ix, iy = min(x, ix), min(y, iy)
			init_tracker = True
		else:
			onTracking = False
	
	elif event == cv2.EVENT_RBUTTONDOWN:
		onTracking = False
		if(w>0):
			ix, iy = x-w/2, y-h/2
			init_tracker = True

if __name__ == '__main__':
    args = parser.parse_args()
    
    # setup tracker
    with open(args.config) as data_file:
        params = json.load(data_file)
    tracker = TrackerSiamRTM(params, args.checkpoint)
    
    if args.video.isdigit():
        print("Webcam")
        webcam = True
        vid = cv2.VideoCapture(int(args.video))
        num_frames = float('inf')
    else:
        print("Demo video")
        webcam = False
        vid = cv2.VideoCapture(args.video)
        num_frames = round(vid.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = round(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = round(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    frame_counter = 0
    init_tracker = True
    cv2.namedWindow('demo')
    cv2.setMouseCallback('demo',draw_boundingbox)
    
    if not webcam:
        ret, frame = vid.read()
        while selectingObject:
            frame_temp = np.copy(frame)
            cv2.rectangle(frame_temp,(ix,iy), (cx,cy), (0,255,255), 1)
            cv2.imshow('demo', frame_temp)
            cv2.waitKey(1)
    
    while vid.isOpened():
        ret, frame = vid.read()
        if not ret:
            break
        frame = noise_handler(frame, args.distortion)
        if selectingObject:
            cv2.rectangle(frame,(ix,iy), (cx,cy), (0,255,255), 1)
        elif init_tracker:
            bbox = np.asarray([int(ix), int(iy), int(w), int(h)])
            tracker.init(frame, bbox)
            init_tracker = False
        else:
            bbox = tracker.update(frame).astype(int)
            cv2.rectangle(frame, [bbox[0], bbox[1]], [bbox[0] + bbox[2], bbox[1] + bbox[3]], [0, 255, 0])      
        cv2.imshow('demo', frame)
        cv2.waitKey(1)
        
    cv2.destroyAllWindows()
    
