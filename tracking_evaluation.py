from __future__ import absolute_import
import argparse
import os
import json

from configs.datasets import set_dataset
from tracker.trackerRTM import TrackerSiamRTM

parser = argparse.ArgumentParser(description='SiamRTM tracking evaluation')
parser.add_argument('--dataset', default='otb100', metavar='DIR',
                    help='lason/otb/got10kval/got10ktest/uav123/uav20l/tc128/vot2017')
parser.add_argument('--checkpoint', default='./checkpoints/model_RTM.pth', metavar='DIR',help='path to weight')
parser.add_argument('--config', default='./configs/parameters.json')
parser.add_argument('--visualize', default=False, help='visualize')
parser.add_argument('--distortion', default='original', choices=['original', 'WGN', 'SnP', 'GB'], 
                    help='choose image distortion (original -> no distortion)')

if __name__ == '__main__':
    args = parser.parse_args()
    
    with open(args.config) as data_file:
        params = json.load(data_file)

    '''setup dataset'''
    dataset = set_dataset(args.dataset)
    
    '''setup trackers'''
    tracker = TrackerSiamRTM(params, args.checkpoint, 
                            distortion=args.distortion)    
    dataset.run(tracker, visualize=args.visualize)
    dataset.report([tracker.name])