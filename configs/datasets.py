from configs.local import dataset_path
from toolkit.got10k.experiments import *

def set_dataset(dataset_name):
    assert dataset_name in ['lasot', 'trackingnet', 'vot2017', 'got10ktest',
                            'got10kval', 'otb100', 'tc128', 'uav123', 'uav20l'], \
            'dataset name not known'

    if dataset_name == 'lasot':
        dataset = ExperimentLaSOT(dataset_path.lasot_path, 
            result_dir=('output/results'),
            report_dir=('output/reports'))
    if dataset_name == 'trackingnet':
        dataset = ExperimentTrackingNet('/media/data/iason/tracking_datasets/trackingnet/',
            result_dir=('output/results'))
    if dataset_name == 'vot2017':
        dataset = ExperimentVOT(dataset_path.vot_path, 
            download = False,
            experiments='supervised', version=2017,
            result_dir=('output/results'),
            report_dir=('output/reports'))
    if dataset_name == 'got10ktest': 
        dataset = ExperimentGOT10k(dataset_path.got10k_path, 
            subset='test',
            result_dir=('output/results'),
            report_dir=('output/reports'))
    if dataset_name == 'otb100':
        dataset = ExperimentOTB(dataset_path.otb_path, 
            version=2015,
            result_dir=('output/results'),
            report_dir=('output/reports'))
    if dataset_name == 'got10kval':
        dataset = ExperimentGOT10k(dataset_path.got10k_path, 
            subset='val',
            result_dir=('output/results'),
            report_dir=('output/reports'))
    if dataset_name == 'uav123':
        dataset = ExperimentUAV123(dataset_path.uav_path, 
            version='UAV123',
            result_dir=('output/results'),
            report_dir=('output/reports'))
    if dataset_name == 'uav20l':
        dataset = ExperimentUAV123(dataset_path.uav_path, 
            version='UAV20L',
            result_dir=('output/results'),
            report_dir=('output/reports'))
    if dataset_name == 'tc128':
        dataset = ExperimentTColor128(dataset_path.tc128_path,
            result_dir=('output/results'),
            report_dir=('output/reports'))

    return dataset

