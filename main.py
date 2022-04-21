from __future__ import print_function
import argparse
from math import log10

import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from dbpn import Net as DBPN
from dbpn import Net as DBPNLL
from dbpn import Net as DBPNS
from dbpn import Net as DBPNITER
from data import get_training_set
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import argparse
import logging
import matplotlib
matplotlib.use("TKAgg")
import matplotlib.pyplot as plt
import math
import cv2
import PSNR
import random
import time
from copy import deepcopy
from pathlib import Path
from threading import Thread
import socket
import numpy as np
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
import yaml
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DFDP
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils.datasets import letterbox
import test  # import test.py to get mAP after each epoch
from models.experimental import attempt_load
from models.yolo import Model
from utils.autoanchor import check_anchors
from utils.datasets import create_dataloader
from utils.general import labels_to_class_weights, increment_path, labels_to_image_weights, init_seeds, \
    fitness, strip_optimizer, get_latest_run, check_dataset, check_file, check_git_status, check_img_size, \
    check_requirements, print_mutation, set_logging, one_cycle, colorstr
from utils.google_utils import attempt_download
from utils.loss import ComputeLoss
from utils.plots import plot_images, plot_labels, plot_results, plot_evolution
from utils.torch_utils import ModelEMA, select_device, intersect_dicts, torch_distributed_zero_first, is_parallel
from utils.wandb_logging.wandb_utils import WandbLogger, check_wandb_resume
from torchvision import transforms as trans
import copy
from utils.datasets import LoadImagesAndLabels
logger = logging.getLogger(__name__)
# import train as YOLO
import evalx3
from torchvision.utils import save_image as svim
def makedirs(path): 
   try: 
        os.makedirs(path) 
   except OSError: 
       if not os.path.isdir(path): 
        #    raise
           pass
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
    parser.add_argument('--upscale_factor', type=int, default=3, help="super resolution upscale factor")
    parser.add_argument('--batchSize', type=int, default=1, help='training batch size')
    parser.add_argument('--nEpochs', type=int, default=3000, help='number of epochs to train for')
    parser.add_argument('--snapshots', type=int, default=1, help='Snapshots')
    parser.add_argument('--start_iter', type=int, default=1, help='Starting Epoch')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning Rate. Default=0.01')
    parser.add_argument('--gpu_mode', type=bool, default=True)
    parser.add_argument('--threads', type=int, default=0, help='number of threads for data loader to use')
    parser.add_argument('--seed', type=int, default=13, help='random seed to use. Default=123')
    parser.add_argument('--gpus', default=1, type=int, help='number of gpu')
    parser.add_argument('--data_dir', type=str, default='./Dataset')
    parser.add_argument('--data_augmentation', type=bool, default=False)
    parser.add_argument('--hr_train_dataset', type=str, default='222x3')
    #parser.add_argument('--hr_train_dataset', type=str, default='testing')
    parser.add_argument('--HR',default ='./Dataset/222x3'+'/')
    parser.add_argument('--test_dataset', type=str, default='./Dataset/x3')
    parser.add_argument('--model_type', type=str, default='DBPN')
    parser.add_argument('--residual', type=bool, default=False)
    parser.add_argument('--patch_size', type=int, default=4, help='Size of cropped HR image')
    # parser.add_argument('--pretrained_sr', default='/pre_trained_DBPN_image.pth', help='sr pretrained base model')
    parser.add_argument('--pretrained_sr', default='/DBPN/x3.pth', help='sr pretrained base model')
    parser.add_argument('--pretrained', type=bool, default=True)
    parser.add_argument('--save_folder', default='weights/end-to-endx3', help='Location to save checkpoint models')
    parser.add_argument('--prefix', default='tpami_residual_filter8', help='Location to save checkpoint models')
    # parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--output', default='./Results/end-to-endx3/', help='Location to save checkpoint models')

    # parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='HR_best.pt', help='initial weights path')
    parser.add_argument('--cfg', type=str, default='models/yolov5x.yaml', help='model.yaml path')
    parser.add_argument('--data', type=str, default='data/CarNumberTestx3.yaml', help='data.yaml path')
    parser.add_argument('--hyp', type=str, default='data/hyp.scratch.yaml', help='hyperparameters path')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=1, help='total batch size for all GPUs')
    parser.add_argument('--img-size', nargs='+', type=int, default=[256,256], help='[train, test] image sizes')# 128*384(256-256/2)(256+256/2)
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--notest', action='store_true', help='only test final epoch')
    parser.add_argument('--noautoanchor', action='store_true', help='disable autoanchor check')
    parser.add_argument('--evolve', action='store_true', help='evolve hyperparameters')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
    parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
    parser.add_argument('--device', default='1', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    parser.add_argument('--adam', action='store_true', help='use torch.optim.Adam() optimizer')
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')
    parser.add_argument('--workers', type=int, default=8, help='maximum number of dataloader workers')
    parser.add_argument('--project', default='runs/train', help='save to project/name')
    parser.add_argument('--entity', default=None, help='W&B entity')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--quad', action='store_true', help='quad dataloader')
    parser.add_argument('--linear-lr', action='store_true', help='linear LR')
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
    parser.add_argument('--upload_dataset', action='store_true', help='Upload dataset as W&B artifact table')
    parser.add_argument('--bbox_interval', type=int, default=-1, help='Set bounding-box image logging interval for W&B')
    parser.add_argument('--save_period', type=int, default=-1, help='Log model after every "save_period" epoch')
    parser.add_argument('--artifact_alias', type=str, default="latest", help='version of dataset artifact to be used')
    opt = parser.parse_args()
   
    # Set DDP variables
    opt.world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    opt.global_rank = int(os.environ['RANK']) if 'RANK' in os.environ else -1
    set_logging(opt.global_rank)
    if opt.global_rank in [-1, 0]:
        check_git_status()
        check_requirements()

    wandb_run = check_wandb_resume(opt)
    if opt.resume and not wandb_run:  # resume an interrupted run
        ckpt = opt.resume if isinstance(opt.resume, str) else get_latest_run()  # specified or most recent path
        assert os.path.isfile(ckpt), 'ERROR: --resume checkpoint does not exist'
        apriori = opt.global_rank, opt.local_rank
        with open(Path(ckpt).parent.parent / 'opt.yaml') as f:
            opt = argparse.Namespace(**yaml.safe_load(f))  # replace
        opt.cfg, opt.weights, opt.resume, opt.batch_size, opt.global_rank, opt.local_rank = \
            '', ckpt, True, opt.total_batch_size, *apriori  # reinstate
        logger.info('Resuming training from %s' % ckpt)
    else:
        opt.hyp = opt.hyp or ('hyp.finetune.yaml' if opt.weights else 'hyp.scratch.yaml')
        opt.data, opt.cfg, opt.hyp = check_file(opt.data), check_file(opt.cfg), check_file(opt.hyp)  # check files
        assert len(opt.cfg) or len(opt.weights), 'either --cfg or --weights must be specified'
        opt.img_size.extend([opt.img_size[-1]] * (2 - len(opt.img_size)))  # extend to 2 sizes (train, test)
        opt.name = 'evolve' if opt.evolve else opt.name
        opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok | opt.evolve))

    # DDP mode
    opt.total_batch_size = opt.batch_size
    device = select_device(opt.device, batch_size=opt.batch_size)
    if opt.local_rank != -1:
        assert torch.cuda.device_count() > opt.local_rank
        torch.cuda.set_device(opt.local_rank)
        device = torch.device('cuda', opt.local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')  # distributed backend
        assert opt.batch_size % opt.world_size == 0, '--batch-size must be multiple of CUDA device count'
        opt.batch_size = opt.total_batch_size // opt.world_size

    # Hyperparameters
    with open(opt.hyp) as f:
        hyp = yaml.safe_load(f)  # load hyps


    logger.info(opt)
    if not opt.evolve:
        tb_writer = None  # init loggers
        if opt.global_rank in [-1, 0]:
            prefix = colorstr('tensorboard: ')
            logger.info(f"{prefix}Start with 'tensorboard --logdir {opt.project}', view at http://localhost:6006/")
            tb_writer = SummaryWriter(opt.save_dir)  # Tensorboard
    # opt = parser.parse_args()
    opt.save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok | opt.evolve)  # increment run
    save_dir, epochs, batch_size, total_batch_size, weights, rank = \
        Path(opt.save_dir), opt.epochs, opt.batch_size, opt.total_batch_size, opt.weights, opt.global_rank    
    gpus_list = range(opt.gpus)
    hostname = str(socket.gethostname())
    cudnn.benchmark = True
    # print(opt)
    # def save_img(img, img_name, epoch=''):
    
    #     save_dir = opt.output+'epoch'+str(epoch)

    #     save_img = img.squeeze().clamp(0, 1).numpy().transpose(1,2,0)
    #     # save img
    #     yolo_data="../DBPNdatasetWithYolov5/images/train"+'/'+img_name
    
    #     if not os.path.exists(save_dir):
    #         os.makedirs(save_dir)
        
    #     save_fn = save_dir +'/'+ img_name
    #     print(save_fn)

    #     cv2.imwrite(save_fn, cv2.cvtColor(save_img*255, cv2.COLOR_BGR2RGB),  [cv2.IMWRITE_PNG_COMPRESSION, 0])
    #     # cv2.imwrite(yolo_data, cv2.cvtColor(save_img*255, cv2.COLOR_BGR2RGB),  [cv2.IMWRITE_PNG_COMPRESSION, 0])
    
    #     print(save_fn)
    #     print(opt.HR+img_name)

    
    #     Lis = PSNR.cal_PSNRandSSIM(opt.HR+img_name ,save_fn)
    
    #     f = open(save_dir+"/PSNR.txt", 'a')
    #     f.write(img_name+f" PSNR is {Lis[0]}\n")
    #     f.close()
    
    #     f = open(save_dir+"/SSIM.txt",'a')
    #     f.write(img_name+f"SSIM is {Lis[1]}\n")
    #     f.close()
    
    #     return Lis
    def img2label_paths(img_paths):
        # Define label paths as a function of image paths
        sa, sb = os.sep + 'images' + os.sep, os.sep + 'labels' + os.sep  # /images/, /labels/ substrings
        print(sa)
        return ['txt'.join(x.replace(sa, sb, 1).rsplit(x.split('.')[-1], 1)) for x in img_paths]
    def load_image(img):
                h0, w0 = img.shape[2:4]  # orig hw
                print(h0,w0)
                r = 256 / max(h0, w0)  # ratio
                if r != 1:  # if sizes are not equal
                    img = cv2.resize(img, (int(w0 * r), int(h0 * r)),
                             interpolation=cv2.INTER_AREA if r < 1  else cv2.INTER_LINEAR)
                return img, (h0, w0), img.shape[:2]  # img, hw_original, hw_resized
    
    rank=0
    # if True:
        # # Generate indices
        # if rank in [-1, 0]:
        #     cw = yolo_model.class_weights.cpu().numpy() * (1 - maps) ** 2 / nc  # class weights
        #     iw = labels_to_image_weights(yolo_dataset.labels, nc=nc, class_weights=cw)  # image weights
        #     yolo_dataset.indices = random.choices(range(yolo_dataset.n), weights=iw, k=yolo_dataset.n)  # rand weighted idx
        # # Broadcast if DDP
        # if rank != -1:
        #     indices = (torch.tensor(yolo_dataset.indices) if rank == 0 else torch.zeros(yolo_dataset.n)).int()
        #     dist.broadcast(indices, 0)
        #     if rank != 0:
        #         yolo_dataset.indices = indices.cpu().numpy()

     
    
    def train(epoch):
    
        skipcount = 0
        epoch_loss = 0
        sum_yolo_loss=0
        sum_yolo_torch=torch.zeros(4,device="cuda:0")
        final_loss=0
        model.train()
        
        start_time = time.time()
        print(start_time)
        print("====\n\n")
        # new=0
        # sum_yolo_loss=0
        total_batch_size=11428
        accumulate = max(round(64 / total_batch_size), 1)  # accumulate loss before optimizing
        i=0
        
        
        for iteration, batch in enumerate(training_data_loader, 1):# dbpn train loader
            ni = iteration + 8 * epoch
            xi = [0, nw]  # x interp
            # model.gr = np.interp(ni, xi, [0.0, 1.0])  # iou loss ratio (obj_loss = 1.0 or iou)
            accumulate = max(1, np.interp(ni, xi, [1, nbs / total_batch_size]).round())
            accumulate=8
            for j, x in enumerate(yolo_optimizer.param_groups):
                # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                x['lr'] = np.interp(ni, xi, [hyp['warmup_bias_lr'] if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                if 'momentum' in x:
                    x['momentum'] = np.interp(ni, xi, [hyp['warmup_momentum'], hyp['momentum']])
            
            input, target, bicubic,name = Variable(batch[0]), Variable(batch[1]), Variable(batch[2]), batch[3]
            nb=8
            
            
            aver = [0,0] 
            number = 0


            if cuda:
                # start_time = time.time()
                input = input.cuda(gpus_list[0])
                target = target.cuda(gpus_list[0])
                bicubic = bicubic.cuda(gpus_list[0])



            optimizer.zero_grad()
            t0 = time.time()
            # input2=copy.deepcopy(input)
            prediction = model(input)
            # prediction2=prediction
            num=epoch
            # print(prediction.shape)
            # print(target.shape)
            # criterion = nn.L1Loss()
            loss =criterion(prediction,target)

            # yolo_prediction=prediction
            yolo_prediction=torch.nn.functional.interpolate(prediction, (int(256), int(256)),mode='bicubic',align_corners=False )   
            # print(yolo_prediction.shape)
            print(i,"/11428")
            save_imgs=imgs_list[i].to(device, non_blocking=True).float()/255.0
            svim(save_imgs, './image_check1.jpeg')
     
            svim(yolo_prediction.float() ,'./image_check2.jpeg')





            # svim(imgs_list[i].to(device, non_blocking=True).float()/255.0 , './image_check1.jpeg')

            # svim(yolo_prediction.float(), './image_check2.jpeg')





            # logger.info(('\n' + '%10s' * 8) % ('Epoch', 'gpu_mem', 'box', 'obj', 'cls', 'total', 'labels', 'img_size'))
            # if rank in [-1, 0]:
            #     pbar = tqdm(pbar, total=nb)  #11428 progress bar #image

         
  
            imgs =yolo_prediction.float()#/255.0
            # imgs_list[i]=imgs_list[i].to(device, non_blocking=True).float()# / 255.0 

            with amp.autocast(enabled=cuda):


                pred = yolo_model(imgs)
   
                yolo_loss, loss_items = compute_loss(pred, target_list[i].to(device))  # loss scaled by batch_size
            i+=1

            sum_yolo_torch+=loss_items

            # loss/=2.0
        
            epoch_loss += loss.data
            sum_yolo_loss+=yolo_loss.data
  

            new_loss=loss+yolo_loss*0.02
            final_loss+=new_loss


            scaler.scale(new_loss).backward()#retain_graph=True)


            accumulate=16
            if ni % accumulate == 0:
                print("step")
                scaler.step(optimizer)  # optimizer.step
                scaler.update()
                optimizer.zero_grad()
                yolo_optimizer.zero_grad()

            optimizer.zero_grad()
                


        plot_epoch_loss.append(epoch_loss.item()/len(training_data_loader))
        plot_sum_yolo_loss.append((sum_yolo_torch[0]+sum_yolo_torch[1]+sum_yolo_torch[2]).item()/len(training_data_loader))
        plot_box_loss.append(sum_yolo_torch[0].item()/len(training_data_loader))
        plot_obj_loss.append(sum_yolo_torch[1].item()/len(training_data_loader))
        plot_cls_loss.append(sum_yolo_torch[2].item()/len(training_data_loader))
        plot_final_loss.append(final_loss[0].item()/len(training_data_loader))
        print(time.gmtime(start_time))
        print("걸린 시간 : "+str(time.time()-start_time))
        printer = ("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss / len(training_data_loader)))
        print(printer)
        # print(epoch)
        axis = np.linspace(1, epoch, epoch)     
        label = 'Backward_loss on {}'.format("Train")
        fig = plt.figure()
        plt.title(label)
        # for idx_scale, scale in enumerate([self.opt.scale[0]]):
        plt.plot(
                axis,
                plot_final_loss,
                # self.log[:, idx_scale].numpy(),
                label='Scale {}'.format(3),
                linestyle='dashed', marker='o', 
                markersize =1, 
                markerfacecolor='black'
            )
        # plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('Backward_loss')
        plt.grid(True)
        plt.savefig('{}/test_{}.pdf'.format(opt.save_folder, "Backward_loss"))
        plt.close(fig)
        axis = np.linspace(1, epoch, epoch)
        # print(axis)
        label = 'SR loss on {}'.format("Train")
        fig = plt.figure()
        plt.title(label)
        # for idx_scale, scale in enumerate([self.opt.scale[0]]):
        plt.plot(
                axis,
                plot_epoch_loss,
                # self.log[:, idx_scale].numpy(),
                label='Scale {}'.format(3),
                linestyle='dashed', marker='o', 
                markersize =1, 
                markerfacecolor='black'
            )
        # plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('SR_loss')
        plt.grid(True)
        plt.savefig('{}/test_{}.pdf'.format(opt.save_folder, "SR_loss"))
        plt.close(fig)
        axis = np.linspace(1, epoch, epoch)
        label = 'Character recognition on {}'.format("Train")
        fig = plt.figure()
        plt.title(label)
        # for idx_scale, scale in enumerate([self.opt.scale[0]]):
        plt.plot(
                axis,
                plot_sum_yolo_loss,
                # self.log[:, idx_scale].numpy(),
                label='Scale {}'.format(3),
                linestyle='dashed', marker='o', 
                markersize =1, 
                markerfacecolor='blue'
            )
        # plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('Character_recognition_loss')
        plt.grid(True)
        plt.savefig('{}/test_{}.pdf'.format(opt.save_folder, "character_recognition"))
        plt.close(fig)
        axis = np.linspace(1, epoch, epoch)     
        label = 'box loss on {}'.format("Train")
        fig = plt.figure()
        plt.title(label)
        # for idx_scale, scale in enumerate([self.opt.scale[0]]):
        plt.plot(
                axis,
                plot_box_loss,
                # self.log[:, idx_scale].numpy(),
                label='Scale {}'.format(3),
                linestyle='dashed', marker='o', 
                markersize =2, 
                markerfacecolor='blue'
            )
        # plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('Box_loss')
        plt.grid(True)
        plt.savefig('{}/test_{}.pdf'.format(opt.save_folder, "box_loss"))
        plt.close(fig)

        axis = np.linspace(1, epoch, epoch)     
        label = 'object_loss on {}'.format("Train")
        fig = plt.figure()
        plt.title(label)
        # for idx_scale, scale in enumerate([self.opt.scale[0]]):
        plt.plot(
                axis,
                plot_obj_loss,
                # self.log[:, idx_scale].numpy(),
                label='Scale {}'.format(3),
                linestyle='dashed', marker='o', 
                markersize =1, 
                markerfacecolor='blue'
            )
        # plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('Object_loss')
        plt.grid(True)
        plt.savefig('{}/test_{}.pdf'.format(opt.save_folder, "obj_loss"))
        plt.close(fig)

        axis = np.linspace(1, epoch, epoch)     
        label = 'cls_loss on {}'.format("Train")
        fig = plt.figure()
        plt.title(label)
        # for idx_scale, scale in enumerate([self.opt.scale[0]]):
        plt.plot(
                axis,
                plot_cls_loss,
                # self.log[:, idx_scale].numpy(),
                label='Scale {}'.format(3),
                linestyle='dashed', marker='o', 
                markersize =1, 
                markerfacecolor='blue'
            )
        # plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('Cls_loss')
        plt.grid(True)
        plt.savefig('{}/test_{}.pdf'.format(opt.save_folder, "cls_loss"))
        plt.close(fig)

        # print(epoch_loss)
        ##############################################
        f = open("./DBPN_avg_lossx3.txt", 'a')
        f.write(printer+'\n')
        # f.write(epoch_loss)
        f.close()
            ##############################################
        f = open("./yolo_lossx3.txt", 'a')
        f.write(str(epoch)+f" Avg yolo loss is {sum_yolo_loss}\n")
        f.close()

        f = open("./yolo_torch_lossx3.txt", 'a')
        f.write(str(epoch)+f" Avg yolo torch loss is {sum_yolo_torch/len(training_data_loader)}\n")
        # f.write(str(plot_sum_yolo_loss)+"\n")
        f.close()
        f=open("./DBPN_YOLO비교x3.txt",'a')
        # f.write(printer+'\n')
        f.write(str(epoch)+f" Avg yolo torch loss is {sum_yolo_torch/len(training_data_loader)}\n")
        f.close()
        f=open("./DBPN+YOLOx3",'a')
        # f.write(printer+'\n')fptr
        f.write(str(epoch)+f" Avg final(yolo+DBPN) loss is {final_loss/len(training_data_loader)}\n")
        f.close()

        f=open("./plot용리스트저장x3.txt",'a')
        f.write(str(plot_epoch_loss)+"\n")
        f.write(str(plot_sum_yolo_loss)+"\n")
        f.write(str(plot_box_loss)+"\n")
        f.write(str(plot_obj_loss)+"\n")
        f.write(str(plot_cls_loss)+"\n")
        f.write(str(plot_final_loss)+"\n")
        f.close()
 
        final_epoch=1
        ema.update_attr(model, include=['yaml', 'nc', 'hyp', 'gr', 'names', 'stride', 'class_weights'])
        # os.makedirs(save_dir)
#         if True:
#             results, maps, times,map50 = test.test(opt.data,
#                                             batch_size=256,
#                                             imgsz=imgsz_test,
#                                             model=ema.ema,
#                                             single_cls=opt.single_cls,
#                                             dataloader=testloader,
#                                             save_dir=save_dir,
#                                             verbose=nc < 50 and final_epoch,
#                                             # plots=plots and final_epoch,
#                                             # log_imgs=opt.log_imgs if wandb else 0,
#                                             compute_loss=compute_loss)
#  # print(results)
#         plot_val_box.append(results[4])
#         plot_val_obj.append(results[5])
#         plot_val_cls.append(results[6])
#         plot_val_yolo_loss.append(results[4]+results[5]+results[6])
#         plot_map.append(map50)
        save_dir=opt.save_folder
        f = open(str(save_dir)+"/train_SR_loss.txt",'a')
        f.write(str(plot_epoch_loss))
        f.close()
        f = open(str(save_dir)+"/train_character_loss.txt",'a')
        f.write(str(plot_sum_yolo_loss))
        f.close()
        f = open(str(save_dir)+"/train_box_loss.txt",'a')
        f.write(str(plot_box_loss))
        f.close()
        f = open(str(save_dir)+"/train_obj_loss.txt",'a')
        f.write(str(plot_obj_loss))
        f.close()
        f = open(str(save_dir)+"/train_cls_loss.txt",'a')
        f.write(str(plot_cls_loss))
        f.close()
        f = open(str(save_dir)+"/train_final_loss.txt",'a')
        f.write(str(plot_final_loss))
        f.close()
        # f = open(str(save_dir)+"/box_loss.txt",'a')
        # f.write(str(results[4]))
        # f.close()
        # f = open(str(save_dir)+"/obj_loss.txt",'a')
        # f.write(str(results[5]))
        # f.close()
        # f = open(str(save_dir)+"/cls_loss.txt",'a')
        # f.write(str(results[6]))
        # f.close()
        # f = open(str(save_dir)+"/character_loss.txt",'a')
        # f.write(str(results[4]+results[5]+results[6]))
        # f.close()
        # f = open(str(save_dir)+"/map.txt",'a')
        # f.write(str(map50))
        # f.close()       
        
        # axis = np.linspace(1, epoch, epoch)     
        # label = 'mAP on {}'.format("val")
        # fig = plt.figure()
        # plt.title(label)
        # # for idx_scale, scale in enumerate([self.opt.scale[0]]):
        # plt.plot(
        #         axis,
        #         plot_map,
        #         # self.log[:, idx_scale].numpy(),
        #         label='Scale {}'.format(4)
        #     )
        # # plt.legend()
        # plt.xlabel('Epochs')
        # plt.ylabel('mAP')
        # plt.grid(True)
        # plt.savefig('{}/test_{}.pdf'.format(opt.save_folder, "val_map"))
        # plt.close(fig)
        
        
        # axis = np.linspace(1, epoch, epoch)     
        # label = 'box loss on {}'.format("val")
        # fig = plt.figure()
        # plt.title(label)
        # # for idx_scale, scale in enumerate([self.opt.scale[0]]):
        # plt.plot(
        #         axis,
        #         plot_val_box,
        #         # self.log[:, idx_scale].numpy(),
        #         label='Scale {}'.format(4)
        #     )
        # # plt.legend()
        # plt.xlabel('Epochs')
        # plt.ylabel('box loss')
        # plt.grid(True)
        # plt.savefig('{}/test_{}.pdf'.format(opt.save_folder, "val_box_loss"))
        # plt.close(fig)
        
        # axis = np.linspace(1, epoch, epoch)     
        # label = 'obj loss on {}'.format("character recognition loss")
        # fig = plt.figure()
        # plt.title(label)
        # # for idx_scale, scale in enumerate([self.opt.scale[0]]):
        # plt.plot(
        #         axis,
        #         plot_val_obj,
        #         # self.log[:, idx_scale].numpy(),
        #         label='Scale {}'.format(4)
        #     )
        # # plt.legend()
        # plt.xlabel('Epochs')
        # plt.ylabel('obj loss')
        # plt.grid(True)
        # plt.savefig('{}/test_{}.pdf'.format(opt.save_folder, "val_obj_loss"))
        # plt.close(fig)
        
        # axis = np.linspace(1, epoch, epoch)     
        # label = 'cls loss on {}'.format("character recognition loss")
        # fig = plt.figure()
        # plt.title(label)
        # # for idx_scale, scale in enumerate([self.opt.scale[0]]):
        # plt.plot(
        #         axis,
        #         plot_val_cls,
        #         # self.log[:, idx_scale].numpy(),
        #         label='Scale {}'.format(4)
        #     )
        # # plt.legend()
        # plt.xlabel('Epochs')
        # plt.ylabel('cls loss')
        # plt.grid(True)
        # plt.savefig('{}/test_{}.pdf'.format(opt.save_folder, "val_character recognition loss"))
        # plt.close(fig)
        
        # axis = np.linspace(1, epoch, epoch)     
        # label = 'Character recognition_loss on {}'.format("val")
        # fig = plt.figure()
        # plt.title(label)
        # # for idx_scale, scale in enumerate([self.opt.scale[0]]):
        # plt.plot(
        #         axis,
        #         plot_val_yolo_loss,
        #         # self.log[:, idx_scale].numpy(),
        #         label='Scale {}'.format(4)
        #     )
        # # plt.legend()
        # plt.xlabel('Epochs')
        # plt.ylabel('Character recognition_loss')
        # plt.grid(True)
        # plt.savefig('{}/test_{}.pdf'.format(opt.save_folder, 'Character recognition_loss'))
        # plt.close(fig)
        # except:
        #     print("Pass")
        # torch.cuda.empty_cache()


    def print_network(net):
        num_params = 0
        for param in net.parameters():
            num_params += param.numel()
        print(net)
        print('Total number of parameters: %d' % num_params)

    def checkpoint(epoch):
        model_out_path = opt.save_folder+"/DBPN/"+"_epoch_{}.pth".format(epoch)
        yolo_model_out_path = opt.save_folder+"/YOLO/"+"_epoch_{}.pt".format(epoch)
        save_dir=opt.save_folder
        torch.save(model.state_dict(), model_out_path)
        print("Checkpoint saved to {}".format(model_out_path))
        torch.save(yolo_model.state_dict(), yolo_model_out_path)
        print("Checkpoint saved to {}".format(yolo_model_out_path))
        
       
    
        val_p,val_s,val_l=evalx3.eval(epoch,model_out_path)
        plot_val_sr_loss.append(round(val_l,6))
        plot_val_psnr.append(round(val_p,3))
        plot_val_ssim.append(round(val_s,3))
        axis = np.linspace(1, epoch, epoch)     
        label = 'SR_loss on {}'.format("val")
        fig = plt.figure()
        plt.title(label)
        # for idx_scale, scale in enumerate([self.opt.scale[0]]):
        plt.plot(
                axis,
                plot_val_sr_loss,
                # self.log[:, idx_scale].numpy(),
                label='Scale {}'.format(3),color="red",
                linestyle='dashed', marker='o', 
                markersize =2, 
                markerfacecolor='red'
            )
        # plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('SR_val_loss')
        plt.grid(True)
        plt.savefig('./'+opt.save_folder+'test_{}.pdf'.format("val_SR_loss"))
        plt.close(fig)

        axis = np.linspace(1, epoch, epoch)     
        label = 'PSNR on {}'.format("val")
        fig = plt.figure()
        plt.title(label)
        # for idx_scale, scale in enumerate([self.opt.scale[0]]):
        plt.plot(
                axis,
                plot_val_psnr,
                # self.log[:, idx_scale].numpy(),
                label='Scale {}'.format(3),
                linestyle='dashed', marker='o', 
                markersize =2, 
                markerfacecolor='blue'
            )
        # plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('PSNR')
        plt.grid(True)
        plt.savefig('./'+opt.save_folder+'test_{}.pdf'.format("val_PSNR"))
        plt.close(fig)

        axis = np.linspace(1, epoch, epoch)     
        label = 'SSIM on {}'.format("val")
        fig = plt.figure()
        plt.title(label)
        # for idx_scale, scale in enumerate([self.opt.scale[0]]):
        plt.plot(
                axis,
                plot_val_ssim,
                # self.log[:, idx_scale].numpy(),
                label='Scale {}'.format(3),
                color="red",
                linestyle='dashed', marker='o', 
                markersize =2, 
                markerfacecolor='red'
            )
        # plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('SSIM')
        plt.grid(True)
        plt.savefig('./'+opt.save_folder+'test_{}.pdf'.format("val_SSIM"))
        plt.close(fig)

    
   

    cuda = opt.gpu_mode
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

    torch.manual_seed(opt.seed)
    if cuda:
        torch.cuda.manual_seed(opt.seed)


    with open(opt.data) as f:
        data_dict = yaml.safe_load(f)  # data dict
        train_path = data_dict['train']
        val_path = data_dict['val']
     # Save run settings


    with open(opt.data) as f:
        data_dict = yaml.safe_load(f)  # data dict
        train_path = data_dict['train']
        val_path = data_dict['val']
        print(train_path)
        nc = 1 if opt.single_cls else int(data_dict['nc'])  # number of classes
        names = ['item'] if opt.single_cls and len(data_dict['names']) != 1 else data_dict['names']  # class names
        yolo_model = Model(opt.cfg, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # create ##448 채널 input  
        gs = max(int(yolo_model.stride.max()), 32)  # grid size (max stride)
        nl = yolo_model.model[-1].nl  # number of detection layers (used for scaling hyp['obj'])
        imgsz, imgsz_test = [check_img_size(x, gs) for x in opt.img_size]  # verify imgsz are gs-multiples
            # for i in range(1000):
            #     print(imgsz)
        save_dir, epochs, batch_size, total_batch_size, weights, rank = \
        Path(opt.save_dir), opt.epochs, opt.batch_size, opt.total_batch_size, opt.weights, opt.global_rank
        results_file = save_dir / 'results.txt'
        start_epoch, best_fitness = 0, 0.0
        pretrained = weights.endswith('.pt')
        # Model parameters
        hyp['box'] *= 3. / nl  # scale to layers
        hyp['cls'] *= nc / 80. * 3. / nl  # scale to classes and layers
        hyp['obj'] *= (imgsz / 640) ** 2 * 3. / nl  # scale to image size and layers
        hyp['label_smoothing'] = opt.label_smoothing
        yolo_model.nc = nc  # attach number of classes to model
        yolo_model.hyp = hyp  # attach hyperparameters to model
        yolo_model.gr = 1.0  # iou loss ratio (obj_loss = 1.0 or iou)
        yolo_dataset = LoadImagesAndLabels(train_path, 256, batch_size,
                                      augment=False,  # augment images
                                      hyp=hyp,  # augmentation hyperparameters
                                      rect=opt.rect,  # rectangular training
                                      cache_images=opt.cache_images,
                                      single_cls=False,
                                      stride=int(0),
                                      pad=0.0,
                                      image_weights=opt.image_weights,
                                      prefix=colorstr('train: '))
        yolo_model.class_weights = labels_to_class_weights(yolo_dataset.labels, nc).to(device) * nc  # attach class weights
        yolo_model.names = names
        cuda = opt.gpu_mode
        # Start training
        t0 = time.time()
        nb=128
        nw = max(round(hyp['warmup_epochs'] * nb), 1000)  # number of warmup iterations, max(3 epochs, 1k iterations)
        # nw = min(nw, (epochs - start_epoch) / 2 * nb)  # limit warmup to < 1/2 of training
        last_opt_step = -1
        maps = np.zeros(nc)  # mAP per class
        results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)

        scaler = amp.GradScaler(enabled=cuda)
        device = next(yolo_model.parameters()).device  # get model device
        compute_loss = ComputeLoss(yolo_model)  # init loss class
        
        # # compute_loss=compute_loss.cuda(gpus_list[0])
        # logger.info(f'Image sizes {imgsz} train, {imgsz_test} test\n'
        #             f'Using {yolo_dataloader.num_workers} dataloader workers\n'
        #             f'Logging results to {save_dir}\n'
        #             f'Starting training for {epochs} epochs...')
        nbs = 64  # nominal batch size 
        # yolo_model=yolo_model.cuda(gpus_list[0])
        freeze = []  # parameter names to freeze (full or partial)
        for k, v in yolo_model.named_parameters():
            v.requires_grad =True  # train all layers
        if any(x in k for x in freeze):
            print('freezing %s' % k)
            v.requires_grad = False
        # Optimizer
        ema = ModelEMA(yolo_model) if rank in [-1, 0] else None
        yolo_model.train()
        nbs = 64  # nominal batch size
        accumulate = max(round(nbs / total_batch_size), 1)  # accumulate loss before optimizing
        hyp['weight_decay'] *= total_batch_size * accumulate / nbs  # scale weight_decay
        logger.info(f"Scaled weight_decay = {hyp['weight_decay']}")

        pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
        for k, v in yolo_model.named_modules():
            if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
                pg2.append(v.bias)  # biases
            if isinstance(v, nn.BatchNorm2d):
                pg0.append(v.weight)  # no decay
            elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
                pg1.append(v.weight)  # apply decay
        yolo_model=yolo_model.cuda(gpus_list[0])
        if opt.adam:

            yolo_optimizer = optim.Adam(pg0, lr=1e-3, betas=(hyp['momentum'], 0.999))  # adjust beta1 to momentum
        else:

            yolo_optimizer = optim.SGD(pg0, lr=1e-3, momentum=hyp['momentum'], nesterov=True)

        yolo_optimizer.add_param_group({'params': pg1, 'weight_decay': hyp['weight_decay']})  # add pg1 with weight_decay
        yolo_optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
        logger.info('Optimizer groups: %g .bias, %g conv.weight, %g other' % (len(pg2), len(pg1), len(pg0)))
        del pg0, pg1, pg2
    if opt.linear_lr:
        lf = lambda x: (1 - x / (epochs - 1)) * (1.0 - hyp['lrf']) + hyp['lrf']  # linear
    else:
        lf = one_cycle(1, hyp['lrf'], epochs)  # cosine 1->hyp['lrf']
            
   
    scheduler = lr_scheduler.LambdaLR(yolo_optimizer, lr_lambda=lf)   # 이 줄 때문에 잠을 못잣네 ㄷㄷ

    scheduler.last_epoch = start_epoch - 1  # do not move
    ema = ModelEMA(yolo_model) if rank in [-1, 0] else None
    if pretrained:
            # Optimizer
            ckpt = torch.load(weights, map_location=device)  # load checkpoint
            exclude = ['anchor'] if (opt.cfg or hyp.get('anchors')) and not opt.resume else []  # exclude keys
            state_dict = ckpt['model'].float().state_dict()  # to FP32
            state_dict = intersect_dicts(state_dict, yolo_model.state_dict(), exclude=exclude)  # intersect
            if ckpt['optimizer'] is not None:
                yolo_optimizer.load_state_dict(ckpt['optimizer'])
                best_fitness = ckpt['best_fitness']

            # EMA
            if ema and ckpt.get('ema'):
                ema.ema.load_state_dict(ckpt['ema'].float().state_dict())
                ema.updates = ckpt['updates']

            # # Results
            # if ckpt.get('training_results') is not None:
            #     results_file.write_text(ckpt['training_results'])  # write results.txt

            # Epochs
            start_epoch = ckpt['epoch'] + 1
            if opt.resume:
                assert start_epoch > 0, '%s training to %g epochs is finished, nothing to resume.' % (weights, epochs)
            if epochs < start_epoch:
                logger.info('%s has been trained for %g epochs. Fine-tuning for %g additional epochs.' %
                            (weights, ckpt['epoch'], epochs))
            epochs += ckpt['epoch']  # finetune additional epochs

            del ckpt, state_dict
    

    # yolo_dataloader, yolo_dataset = create_dataloader(train_path, imgsz, batch_size, gs, opt,    # yolo dataloader
    #                                         hyp=hyp, augment=True, cache=opt.cache_images, rect=opt.rect, rank=rank,
    #                                         world_size=opt.world_size, workers=opt.workers,
    #                                         image_weights=opt.image_weights, quad=opt.quad, prefix=colorstr('train: '))
    

    
    batch_size = min(batch_size, len(yolo_dataset))
    world_size=2
    workers=2
    quad=False
    image_weights=True
    nw = min([os.cpu_count() // world_size, batch_size if batch_size > 1 else 0, workers])  # number of workers
    sampler = torch.utils.data.distributed.DistributedSampler(yolo_dataset) if rank != -1 else None
    loader = torch.utils.data.DataLoader if image_weights else InfiniteDataLoader
    # Use torch.utils.data.DataLoader() if dataset.properties will update during training else InfiniteDataLoader()
    yolo_dataloader = loader(yolo_dataset,
                        batch_size=batch_size,
                        num_workers=nw,
                        sampler=sampler,
                        pin_memory=True,
                        collate_fn=LoadImagesAndLabels.collate_fn4 if quad else LoadImagesAndLabels.collate_fn)

    # mlc = np.concatenate(yolo_dataset.labels, 0)[:, 0].max()  # max label class
        # print(yolo_dataset[1])
    test_path = data_dict['val']
    # testloader = create_dataloader(val_path, imgsz_test, batch_size * 2, gs, opt,  # testloader
    #                                    hyp=hyp, cache=opt.cache_images and not opt.notest, rect=True, rank=-1,
    #                                    world_size=opt.world_size, workers=opt.workers,
                                    #    pad=0.5, prefix=colorstr('val: '))[0]        
    nb = len(yolo_dataloader)  # number of batches
    # assert mlc < nc, 'Label class %g exceeds nc=%g in %s. Possible class labels are 0-%g' % (mlc, nc, opt.data, nc - 1)
    if rank in [-1, 0]:
        ema.updates = start_epoch * nb // accumulate  # set EMA updates
        testloader = create_dataloader(test_path, imgsz_test, total_batch_size, gs, opt,  # testloader
                                       hyp=hyp, cache=opt.cache_images and not opt.notest, rect=True, rank=-1,
                                       world_size=opt.world_size, workers=opt.workers,
                                       pad=0.5, prefix=colorstr('val: '))[0]
    epochs=2

    

    plot_epoch_loss=[]
    plot_sum_yolo_loss= []
    plot_box_loss=[]
    plot_obj_loss=[]
    plot_cls_loss=[]
    plot_final_loss=[]
    print(len(plot_box_loss))
    print(len(plot_sum_yolo_loss))
    print(len(plot_cls_loss))
    print(len(plot_obj_loss))
    print(len(plot_final_loss))




    plot_val_sr_loss=[]
    plot_val_psnr=[]
    plot_val_ssim=[]
    plot_val_box=[]
    plot_val_obj=[]
    plot_val_cls=[]
    plot_val_yolo_loss=[]
    plot_map=[]
    # f=open("DBPN_avg_lossx2.txt", 'r')
    # p=open("./Results/end-to-endx2/PSNR.txt",'r')
    # s=open("./Results/end-to-endx2/SSIM.txt",'r')
    # l=open("./Results/end-to-endx2/val_loss.txt",'r')
    # while True:
    #     line = f.readline()
    #     if not line: break
    #     plot_epoch_loss.append(float(line.split()[6]))
    # print(len(plot_epoch_loss))
    # while True:
    #     line = p.readline()
    #     if not line: break
    #     plot_val_psnr.append(float(line.split()[4]))
    # print(len(plot_val_psnr))
    # while True:
    #     line = s.readline()
    #     if not line: break
    #     plot_val_ssim.append(float(line.split()[4]))
    # print(len(plot_val_ssim))
    # while True:
    #     line = l.readline()
    #     if not line: break
    #     plot_val_sr_loss.append(float(line.split()[4]))
    # print(len(plot_epoch_loss))
    # print(len(plot_sum_yolo_loss))
    # print(len(plot_box_loss))
    # print(len(plot_obj_loss))
    # print(len(plot_cls_loss))
    # print(len(plot_final_loss))
    mloss = torch.zeros(4, device=device)  # mean losses
    # if rank != -1:
    #     yolo_dataloader.sampler.set_epoch(epoch)
    pbar = enumerate(yolo_dataloader)
    

    logger.info(('\n' + '%10s' * 8) % ('Epoch', 'gpu_mem', 'box', 'obj', 'cls', 'total', 'labels', 'img_size'))
    if rank in [-1, 0]:
        pbar = tqdm(pbar, total=nb)  # progress bar
    target_list=[]
    paths_list=[]
    imgs_list=[]
    for i, (imgs, targets, paths, _) in pbar:
        imgs_list.append(imgs)
        # print(imgs.to(device).shape)
        # print(targets.to(device).shape)
        target_list.append(targets)
        paths_list.append(paths)
    # for i in range(11428):
        # print(str(imgs_list[i])+"\n")
        # print(str(target_list[i])+"\n")
        # print(str(paths_list[i])+"\n")
    # print("시작")
    # print('===> Loading datasets')
    train_set = get_training_set(opt.data_dir, opt.hr_train_dataset, opt.upscale_factor, opt.patch_size, opt.data_augmentation)
    training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=False)   #DBPN 데이터로더
    # testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.testBatchSize, shuffle=False)

    # print('===> Building model ', opt.model_type)
    if opt.model_type == 'DBPNLL':
        model = DBPNLL(num_channels=3, base_filter=64,  feat = 256, num_stages=10, scale_factor=opt.upscale_factor) 
    elif opt.model_type == 'DBPN-RES-MR64-3':
        model = DBPNITER(num_channels=3, base_filter=64,  feat = 256, num_stages=3, scale_factor=opt.upscale_factor)
    else:
        model = DBPN(num_channels=3, base_filter=64,  feat = 256, num_stages=7, scale_factor=opt.upscale_factor) 
    
    model = torch.nn.DataParallel(model, device_ids=gpus_list)
    for k, v in model.named_parameters():
        v.requires_grad =True  # train all layers
    criterion = nn.L1Loss()
    # criterion2= nn.L1Loss()

    print('---------- Networks architecture -------------')
    print_network(model)
    # print('----------------------------------------------')

    if opt.pretrained:
        model_name = os.path.join(opt.save_folder + opt.pretrained_sr)
        print("name : "+model_name)
        if os.path.exists(model_name):
            #model= torch.load(model_name, map_location=lambda storage, loc: storage)
            print("name : "+model_name)
            print("name : "+model_name)
            model.load_state_dict(torch.load(model_name, map_location=lambda storage, loc: storage))
            print('Pre-trained SR model is loaded.')
        else:
            print('Pre-trained SR model is not loaded.')

    if cuda:
        model = model.cuda(gpus_list[0])
        criterion = criterion.cuda(gpus_list[0])
        yolo_model=yolo_model.cuda(gpus_list[0])

        
        
        # criterion2 = criterion.cuda(gpus_list[0])

    optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.999), eps=1e-8)
    for epoch in range(opt.start_iter, opt.nEpochs + 1):
        train(epoch)
        

    # learning rate is decayed by a factor of 10 every half of total epochs
        if (epoch+1) % (opt.nEpochs/2) == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] /= 10.0
            print('Learning rate decay: lr={}'.format(optimizer.param_groups[0]['lr']))
            
        if (epoch+1) % (opt.snapshots) == 0:
            checkpoint(epoch)
    # Scheduler




# def x8_forward(img, model, precision='single'):
#     def _transform(v, op):
#         if precision != 'single': v = v.float()

#         v2np = v.data.cpu().numpy()
#         if op == 'vflip':
#             tfnp = v2np[:, :, :, ::-1].copy()
#         elif op == 'hflip':
#             tfnp = v2np[:, :, ::-1, :].copy()
#         elif op == 'transpose':
#             tfnp = v2np.transpose((0, 1, 3, 2)).copy()
        
#         ret = torch.Tensor(tfnp).cuda()

#         if precision == 'half':
#             ret = ret.half()
#         elif precision == 'double':
#             ret = ret.double()

#         with torch.no_grad():
#             ret = Variable(ret)

#         return ret

#     inputlist = [img]
#     for tf in 'vflip', 'hflip', 'transpose':
#         inputlist.extend([_transform(t, tf) for t in inputlist])

#     outputlist = [model(aug) for aug in inputlist]
#     for i in range(len(outputlist)):
#         if i > 3:
#             outputlist[i] = _transform(outputlist[i], 'transpose')
#         if i % 4 > 1:
#             outputlist[i] = _transform(outputlist[i], 'hflip')
#         if (i % 4) % 2 == 1:
#             outputlist[i] = _transform(outputlist[i], 'vflip')
    
#     output = reduce((lambda x, y: x + y), outputlist) / len(outputlist)

#     return output
    
# def chop_forward(x, model, scale, shave=8, min_size=80000, nGPUs=opt.gpus):
#     b, c, h, w = x.size()
#     h_half, w_half = h // 2, w // 2
#     h_size, w_size = h_half + shave, w_half + shave
#     inputlist = [
#         x[:, :, 0:h_size, 0:w_size],
#         x[:, :, 0:h_size, (w - w_size):w],
#         x[:, :, (h - h_size):h, 0:w_size],
#         x[:, :, (h - h_size):h, (w - w_size):w]]

#     if w_size * h_size < min_size:
#         outputlist = []
#         for i in range(0, 4, nGPUs):
#             with torch.no_grad():
#                 input_batch = torch.cat(inputlist[i:(i + nGPUs)], dim=0)
#             if opt.self_ensemble:
#                 with torch.no_grad():
#                     output_batch = x8_forward(input_batch, model)
#             else:
#                 with torch.no_grad():
#                     output_batch = model(input_batch)
#             outputlist.extend(output_batch.chunk(nGPUs, dim=0))
#     else:
#         outputlist = [
#             chop_forward(patch, model, scale, shave, min_size, nGPUs) \
#             for patch in inputlist]

#     h, w = scale * h, scale * w
#     h_half, w_half = scale * h_half, scale * w_half
#     h_size, w_size = scale * h_size, scale * w_size
#     shave *= scale

#     with torch.no_grad():
#         output = Variable(x.data.new(b, c, h, w))

#     output[:, :, 0:h_half, 0:w_half] \
#         = outputlist[0][:, :, 0:h_half, 0:w_half]
#     output[:, :, 0:h_half, w_half:w] \
#         = outputlist[1][:, :, 0:h_half, (w_size - w + w_half):w_size]
#     output[:, :, h_half:h, 0:w_half] \
#         = outputlist[2][:, :, (h_size - h + h_half):h_size, 0:w_half]
#     output[:, :, h_half:h, w_half:w] \
#         = outputlist[3][:, :, (h_size - h + h_half):h_size, (w_size - w + w_half):w_size]

#     return output