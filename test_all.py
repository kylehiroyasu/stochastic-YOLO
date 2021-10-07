import argparse
import itertools
import json
import sys

import numpy as np
import pandas as pd
from torch.nn.functional import dropout
from torch.utils.data import DataLoader
from torchvision.ops import box_convert

from models import *
from utils.datasets import *
from utils.utils import *

sys.path.append('./cocoapi/PythonAPI/')

sys.path.append('./pdq_evaluation')
from read_files import convert_coco_det_to_rvc_det

def enable_dropout(m):
    for each_module in m.modules():
        if each_module.__class__.__name__.startswith('Dropout'):
            each_module.train()

def change_dropout_rate(m, perc):
    for each_module in m.modules():
        if each_module.__class__.__name__.startswith('Dropout'):
            each_module.p = perc

def get_single_darknet_model(cfg, imgsz, weights, device, dropout_ids, new_drop_rate):
    # Initialize model
    model = Darknet(cfg, imgsz)

    # Load weights
    attempt_download(weights)
    if weights.endswith('.pt'):  # pytorch format
        loaded_model = torch.load(weights, map_location=device)['model']
        if dropout_ids is not None:
            change_model_state_dict(loaded_model, dropout_ids=dropout_ids)
        model.load_state_dict(loaded_model)
    else:  # darknet format
        load_darknet_weights(model, weights)
    
    if new_drop_rate is not None:
        print('Changing default dropout rate...')
        change_dropout_rate(m=model, perc=new_drop_rate)

    # Fuse
    model.fuse()
    model.to(device)

    return model

def bounding_box_variance(samples):
    # Take mean of predictions for single image
    variance = []
    for i, m in enumerate(samples.mean(dim=1)):
        expectation = m.dot(m)
        sum = 0
        # loop over defined samples of a given prediction
        for sample in samples[i]:
            square_predictions = sample.dot(sample)
            sum += square_predictions - expectation
        average = sum / samples[i].shape[0]
        variance.append(average)
    variance = torch.stack(variance).tolist()
    return variance

def test(cfg,
         data,
         weights=None,
         batch_size=16,
         imgsz=416,
         conf_thres=0.001,
         iou_thres=0.6,  # for nms
         save_json=False,
         single_cls=False,
         augment=False,
         model=None,
         dataloader=None,
         multi_label=False,
         dropout_ids=None,
         name='',
         dropout_at_inference=False,
         num_samples=1,
         corruption_num=None,
         severity=None,
         get_unknowns=False,
         only_inference=False,
         new_drop_rate=None,
         with_cached_mcdrop=False,
         ensemble_main_name=None):

    # Change name automatically if there is corruption going on
    if corruption_num is not None:
        name = name + f'_c{corruption_num}s{severity}'
    # Initialize/load model and set device
    if model is None:
        is_training = False
        device = torch_utils.select_device(opt.device, batch_size=batch_size)
        verbose = opt.task == 'test'

        # Remove previous
        for f in glob.glob(f'output/test_batch_{name}_{conf_thres}_{iou_thres}_*.jpg'):
            os.remove(f)

        if ensemble_main_name is None:
            model = get_single_darknet_model(cfg, imgsz, weights, device, dropout_ids, new_drop_rate)   

            if with_cached_mcdrop:
                model_decorator = DecoratorDarknetMCDrop(darknet_model=model)
            if device.type != 'cpu' and torch.cuda.device_count() > 1:
                model = nn.DataParallel(model)
        else:
            ensemble_models = []
            for i in range(num_samples):
                weights_path = f'weights/{ensemble_main_name}_{i + 1}.pt'
                model = get_single_darknet_model(cfg, imgsz, weights_path, device, dropout_ids, new_drop_rate)
                if device.type != 'cpu' and torch.cuda.device_count() > 1:
                    model = nn.DataParallel(model)

                model.eval()
                if dropout_at_inference:
                    enable_dropout(model)
                _ = model(torch.zeros((1, 3, imgsz, imgsz), device=device)) if device.type != 'cpu' else None  # run once

                ensemble_models.append(model)
            print(f'Evaluating on an ensemble of {len(ensemble_models)} models')
            
    else:  # called by train.py
        is_training = True
        device = next(model.parameters()).device  # get model device
        verbose = False

    
    # Configure run
    data = parse_data_cfg(data)
    nc = 1 if single_cls else int(data['classes'])  # number of classes
    path = data['valid']  # path to test images
    names = load_classes(data['names'])  # class names
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    iouv = iouv[0].view(1)  # comment for mAP@0.5:0.95
    niou = iouv.numel()

    # Dataloader
    if dataloader is None:
        if corruption_num is not None:
            print(f'Dataloader will have corrupted images with number {corruption_num} and severity {severity}')
        dataset = LoadImagesAndLabels(path, imgsz, batch_size, rect=True, single_cls=opt.single_cls, pad=0.5,
                                      corruption_num=corruption_num, severity=severity)
        batch_size = min(batch_size, len(dataset))
        dataloader = DataLoader(dataset,
                                batch_size=batch_size,
                                num_workers=min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8]),
                                pin_memory=True,
                                collate_fn=dataset.collate_fn)

    seen = 0
    if ensemble_main_name is None:
        model.eval()
        if dropout_at_inference:
            enable_dropout(model)
            
        _ = model(torch.zeros((1, 3, imgsz, imgsz), device=device)) if device.type != 'cpu' else None  # run once

    coco91class = coco80_to_coco91_class()
    s = ('%20s' + '%10s' * 6) % ('Class', 'Images', 'Targets', 'P', 'R', 'mAP@0.5', 'F1')
    p, r, f1, mp, mr, map, mf1, t0, t1 = 0., 0., 0., 0., 0., 0., 0., 0., 0.
    loss = torch.zeros(3, device=device)
    jdict, stats, ap, ap_class = [], [], [], []
    predictions = []
    for batch_i, (imgs, targets, paths, shapes) in enumerate(tqdm(dataloader, desc=s)):
        imgs = imgs.to(device).float() / 255.0  # uint8 to float32, 0 - 255 to 0.0 - 1.0
        targets = targets.to(device)
        nb, _, height, width = imgs.shape  # batch size, channels, height, width
        whwh = torch.Tensor([width, height, width, height]).to(device)

        #targets[:,2] += targets[:,4]/2
        #targets[:,3] += targets[:,5]/2

        # Disable gradients
        with torch.no_grad():
            # Run model
            t = torch_utils.time_synchronized()
            if num_samples == 1:
                inf_out, train_out = model(imgs, augment=augment)  # inference and training outputs
            # MC-Dropout sampling
            elif num_samples > 1:
                if with_cached_mcdrop:
                    infs_all = model_decorator.sample_from_model(imgs, num_samples=num_samples, augment=augment)
                else:
                    infs_all = []
                    for i in range(num_samples):
                        # train_out is only for when training
                        if ensemble_main_name is None:
                            inf_out_i, _ = model(imgs, augment=augment) 
                        else:
                            inf_out_i, _ = ensemble_models[i](imgs, augment=augment) 
                        
                        # Appending batch_size X detections X 1 X 85
                        infs_all.append(inf_out_i.unsqueeze(2))
                    
                inf_mean = torch.mean(torch.stack(infs_all), dim=0)
                infs_all.insert(0, inf_mean)
                
                # Creating a single tensor with the averaged tensor for calculations, and all the sampled tensors for variability
                # batch_size X detections X (mean_tensor + sampled tensors) X 85
                inf_out = torch.cat(infs_all, dim=2)
                
            t0 += torch_utils.time_synchronized() - t

            # Compute loss
            if is_training:  # if model has loss hyperparameters
                loss += compute_loss(train_out, targets, model)[1][:3]  # GIoU, obj, cls

            # Run NMS
            t = torch_utils.time_synchronized()

            output, all_scores, sampled_coords, sampled_probs = non_max_suppression(inf_out, conf_thres=conf_thres, iou_thres=iou_thres, multi_label=multi_label,
                                                                     max_width=width, max_height=height, get_unknowns=get_unknowns)
            
            t1 += torch_utils.time_synchronized() - t

        # Statistics per image
        for si, pred in enumerate(output):
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  # target class
            seen += 1

            if pred is None:
                if nl:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                continue

            # Append to text file
            # with open('test.txt', 'a') as file:
            #    [file.write('%11.5g' * 7 % tuple(x) + '\n') for x in pred]

            # Clip boxes to image bounds
            clip_coords(pred, (height, width))
            
            # Append to pycocotools JSON dictionary
            if save_json:                    
                
                # [{"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}, ...
                #image_id = int(Path(paths[si]).stem.split('_')[-1])
                # For ccpd
                image_id = Path(paths[si]).stem
                box = pred[:, :4].clone()  # xyxy
                scale_coords(imgs[si].shape[1:], box, shapes[si][0], shapes[si][1])  # to original shape
                box = xyxy2xywh(box)  # xywh
                box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner

                # Getting covariances
                # The transformations to coordinates follow the ones that are done below here after the if clause
                if num_samples > 1:
                    # output: BS(list) x NUM_DETECTIONS x 6
                    # sampled_coords : BS(list) x NUM_DETECTIONS x NUM_SAMPLES x 4
                    # sampled_boxes : NUM_DETECTIONS x NUM_SAMPLES x 4
                    sampled_boxes = xywh2xyxy(sampled_coords[si].reshape(-1, 4)).reshape(sampled_coords[si].shape)
                    clip_coords(sampled_boxes.reshape(-1, 4), (height, width))
                    
                    scale_coords(imgs[si].shape[1:], sampled_boxes.reshape(-1, 4), shapes[si][0], shapes[si][1])
                        
                    # It will have 2 covariances matrices of 2X2 for each one of the two xy coordinates
                    covar_batch = torch.zeros(sampled_boxes.shape[0], 2, 2, 2)
                    for det_id in range(sampled_boxes.shape[0]):
                        covar_batch[det_id, 0, ...] = cov(sampled_boxes[det_id, :, :2])
                        covar_batch[det_id, 1, ...] = cov(sampled_boxes[det_id, :, 2:])
                        
                    # Rounding it for smaller size
                    covar_batch = np.around(covar_batch.numpy(), 5).tolist()
                else:
                    # Just dummy covars for the json zip() down below
                    covar_batch = [None] * pred.shape[0]

                for p, b, p_all, covar_xyxy in zip(pred.tolist(), box.tolist(), all_scores[si].tolist(), covar_batch):
                    if covar_xyxy is not None:
                        # Covariances need to be positive semi-definite, so just transform them here already
                        for i, covar_tmp in enumerate(covar_xyxy):
                            covar_tmp = np.array(covar_tmp)
                            if not is_pos_semidef(covar_tmp):
                                print('Warning: Converted covar to near PSD')
                                covar_xyxy[i] = get_near_psd(covar_tmp).tolist()

                    jdict.append({'image_id': image_id,
                                  'category_id': coco91class[int(p[5])],
                                  'bbox': [round(x, 3) for x in b],
                                  'score': round(p[4], 5),
                                  'all_scores': [round(x, 5) for x in p_all],
                                  'covars' : covar_xyxy})
                    

            # Assign all predictions as incorrect
            correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device=device)
            if nl and not only_inference:
                detected = []  # target indices
                tcls_tensor = labels[:, 0]

                # target boxes
                tbox = xywh2xyxy(labels[:, 1:5]) * whwh

                # Per target class
                for cls in torch.unique(tcls_tensor):
                    ti = (cls == tcls_tensor).nonzero().view(-1)  # target indices
                    pi = (cls == pred[:, 5]).nonzero().view(-1)  # prediction indices

                    # Search for detections
                    if pi.shape[0]:
                        # Prediction to target ious
                        ious, i = box_iou(pred[pi, :4], tbox[ti]).max(1)  # best ious, indices

                        # Append detections
                        for j in (ious > iouv[0]).nonzero():
                            d = ti[i[j]]  # detected target
                            if d not in detected:
                                detected.append(d)
                                correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                                if len(detected) == nl:  # all targets already located in image
                                    break

            # Append statistics (correct, conf, pcls, tcls)
            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))
            cpu_pred = pred.cpu()
            # Loop over batch of images
            if sampled_coords is not None:
                variance = bounding_box_variance(sampled_coords[si])
            else:
                variance = [0] * correct.shape[0]

            #TODO: Figure out if this is xyxy or cxcywh
            # TODO: find better way to calulate this
            predictions.append({
                'correct': correct.cpu().tolist(),
                'bounding_boxes': cpu_pred[:,:4].tolist(),
                'bounding_box_variance': variance,
                'confidence': cpu_pred[:, 4].tolist(),
                'entropy': (-cpu_pred[:, 4] * cpu_pred[:, 4].log() - (1-cpu_pred[:, 4]) * (1-cpu_pred[:, 4]).log()).tolist(),
                'predicted_class': cpu_pred[:, 5].tolist(),
                'image_name': Path(paths[si]).stem
            })

        # Plot images for batch
        if batch_i < 1:
            if not only_inference:
                f = f'output/batch_figures/test_batch_{name}_{conf_thres}_{iou_thres}_%g_gt.jpg' % batch_i  # filename
                plot_images(imgs, targets, paths=paths, names=names, fname=f, max_subplots=batch_size)  # ground truth
            f = f'output/batch_figures/test_batch_{name}_{conf_thres}_{iou_thres}_%g_pred.jpg' % batch_i
            plot_images(imgs, output_to_target(output, width, height), paths=paths, names=names, fname=f, max_subplots=batch_size)  # predictions
        
    if not only_inference:
        # Compute statistics
        stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
        if len(stats):
            p, r, ap, f1, ap_class = ap_per_class(*stats)
            if niou > 1:
                p, r, ap, f1 = p[:, 0], r[:, 0], ap.mean(1), ap[:, 0]  # [P, R, AP@0.5:0.95, AP@0.5]
            mp, mr, map, mf1 = p.mean(), r.mean(), ap.mean(), f1.mean()
            nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
        else:
            nt = torch.zeros(1)

        # Print results
        pf = '%20s' + '%10.3g' * 6  # print format
        print(pf % ('all', seen, nt.sum(), mp, mr, map, mf1))

        # Print results per class
        if verbose and nc > 1 and len(stats):
            for i, c in enumerate(ap_class):
                print(pf % (names[c], seen, nt[c], p[i], r[i], ap[i], f1[i]))

    # Print speeds
    if verbose or save_json:
        t = tuple(x / seen * 1E3 for x in (t0, t1, t0 + t1)) + (imgsz, imgsz, batch_size)  # tuple
        print('Speed: %.1f/%.1f/%.1f ms inference/NMS/total per %gx%g image at batch-size %g' % t)

    # Save JSON
    if save_json and len(jdict):
        print(f'\nSaving {len(jdict)} detections...')
        print('\nCOCO mAP with pycocotools...')
        # imgIds = [int(Path(x).stem.split('_')[-1]) for x in dataloader.dataset.img_files]
        with open(f'output/dets_{name}_{conf_thres}_{iou_thres}.json', 'w') as file:
            json.dump(jdict, file)

        '''
        No need for this part as it will be evaluated later
        try:
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval

            # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
            cocoGt = COCO(glob.glob(data['instances_path'])[0])  # initialize COCO ground truth api
            cocoDt = cocoGt.loadRes(f'output/dets_{name}_{conf_thres}_{iou_thres}.json')  # initialize COCO pred api

            cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
            cocoEval.params.imgIds = imgIds  # [:32]  # only evaluate these images
            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()
            # mf1, map = cocoEval.stats[:2]  # update to pycocotools results (mAP@0.5:0.95, mAP@0.5)
        except Exception as e:
            print(e)
            print('WARNING: pycocotools must be installed with numpy==1.17 to run correctly. '
                  'See https://github.com/cocodataset/cocoapi/issues/356')
        '''
        del jdict
        print('Converting to RVC1 format...')
        convert_coco_det_to_rvc_det(det_filename=f'output/dets_{name}_{conf_thres}_{iou_thres}.json', 
                                    gt_filename=glob.glob(data['instances_path'])[0], 
                                    save_filename=f'output/dets_converted_{name}_{conf_thres}_{iou_thres}.json')

    # Return results
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    return (mp, mr, map, mf1, *(loss.cpu() / len(dataloader)).tolist()), maps, predictions


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('--batch-size', type=int, default=16, help='size of each image batch')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='IOU threshold for NMS')
    parser.add_argument('--save-json', action='store_true', default=False, help='save a cocoapi-compatible JSON results file')
    parser.add_argument('--task', default='test', help="'test', 'study', 'benchmark'")
    parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1) or cpu')
    parser.add_argument('--single-cls', action='store_true', help='train as single-class dataset')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--dropout_ids', nargs='*', type=int, help='when weights file is from a non-dropout model, and cfg is a dropout model, this indicates what are the dropout layers IDs starting from 0')
    parser.add_argument('--name', default='', help='renames resulting files in this script using this name')
    parser.add_argument('--num_samples', type=int, default=10, help='How many times to sample if doing MC-Dropout')
    parser.add_argument('--corruption_num', type=int, help='which corruption number to use from imagecorruptions')
    parser.add_argument('--severity', type=int, help='which severity to use for the corruption in --corruption_num')
    parser.add_argument('--get_unknowns', action='store_true', help='get bboxes of unknowns conf_labels < 0.5 and threshold > 0.1')
    parser.add_argument('--only_inference', action='store_true', help='to indicate that the info in --data does not have valid ground truths or to not calculate some evaluations')
    parser.add_argument('--new_drop_rate', type=float, help='change the dropout rate of Dropout layers to this, regardless of the values in .cfg file')
    parser.add_argument('--ensemble_main_name', type=str, help='the main name of the ensemble model in which --num_samples will be sampled from. This takes precendence on other options. This considers pretrained models will be in weights/ folder.')
    opt = parser.parse_args()
    #opt.save_json = opt.save_json or any([x in opt.data for x in ['coco.data', 'coco2014.data', 'coco2017.data', 'coco2017_sampled.data']])
    #opt.cfg = check_file(opt.cfg)  # check file
    #opt.data = check_file(opt.data)  # check file
    print(opt)

    if opt.new_drop_rate is not None and (opt.new_drop_rate < 0 or opt.new_drop_rate > 1):
        sys.exit('Wrong format for --new_drop_rate. It needs to be between [0, 1]')
    #if opt.with_cached_mcdrop and (opt.ensemble_main_name is not None):
    #     sys.exit('--with_cached_mcdrop cannot be used together with --ensemble_main_name')

    datasets = [
        #'ccpd.data',
        'ccpd_blur.data',
        'ccpd_challenge.data',
        'ccpd_db.data',
        'ccpd_fn.data',
        'ccpd_rotate.data',
        'ccpd_tilt.data',
    ]
    datasets = ['ccpd_weather.data']
    inference_type = ['ensemble', 'normal','dropout', ]
    #inference_type = ['dropout']
    all_results = []
    for data, inf_type in itertools.product(datasets, inference_type):
        if inf_type == 'ensemble':
            name = f'{data.split(".")[0]}_ensemble'
            cfg = 'cfg/yolov3-custom-ccpd.cfg'
            weights = None
            ensemble_main_name = 'best_ccpd_ensemble'
            dropout = False
            num_samples = 3
            new_drop_rate = 0.0
        elif inf_type == 'dropout':
            name = f'{data.split(".")[0]}_dropout'
            cfg = 'cfg/yolov3-mcdrop25-ccpd.cfg'
            weights = 'weights/best_ccpd_mcdrop0.pt'
            ensemble_main_name = None
            dropout = True
            num_samples = 10
            new_drop_rate = 0.25
        else:
            name = data.split('.')[0]
            cfg = 'cfg/yolov3-mcdrop25-ccpd.cfg'
            weights = 'weights/best_ccpd_mcdrop0.pt'
            ensemble_main_name = None
            dropout = False
            num_samples = 1
            new_drop_rate = 0.0
        result = test(cfg=cfg,
                        data=data,
                        weights=weights,
                        batch_size=opt.batch_size,
                        imgsz=opt.img_size,
                        conf_thres=opt.conf_thres,
                        iou_thres=opt.iou_thres,
                        save_json=opt.save_json,
                        single_cls=opt.single_cls,
                        augment=opt.augment,
                        dropout_ids=opt.dropout_ids,
                        name=name,
                        dropout_at_inference=dropout,
                        num_samples=num_samples,
                        corruption_num=opt.corruption_num,
                        severity=opt.severity,
                        get_unknowns=opt.get_unknowns,
                        only_inference=opt.only_inference,
                        new_drop_rate=new_drop_rate,
                        with_cached_mcdrop=dropout,
                        ensemble_main_name=ensemble_main_name)
        # (mp, mr, map, mf1, *(loss.cpu() / len(dataloader)).tolist()), maps
        all_results.append({
            'data': data,
            'dropout': dropout,
            'mode': inf_type,
            'mp': result[0][0],
            'mr': result[0][1],
            'map': result[0][2],
            'mf1': result[0][3],
        })
        predictions = result[2]
        with open(f'{name}.csv', mode='w') as f:
            for p in predictions:
                f.write(f'{json.dumps(p)}\n')
        df = pd.DataFrame(all_results)
        df.to_csv('all_test_results.csv')