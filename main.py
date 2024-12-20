
# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import os
import time
import json
import random
import argparse
import datetime
import numpy as np
import torch.optim as optim
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import accuracy, AverageMeter

from config import get_config
from models import build_model
from data import build_loader
from lr_scheduler import build_scheduler
from optimizer import build_optimizer
from logger import create_logger
from utils import load_checkpoint, load_pretrained, save_checkpoint, NativeScalerWithGradNormCount, auto_resume_helper, \
    reduce_tensor
from sklearn.metrics import precision_score, recall_score, f1_score
import torch
# pytorch major version (1.x or 2.x)
PYTORCH_MAJOR_VERSION = int(torch.__version__.split('.')[0])


def parse_option():
    parser = argparse.ArgumentParser('Swin Transformer training and evaluation script', add_help=False)
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
    parser.add_argument('--data-path', type=str, help='path to dataset')
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                             'full: cache all data, '
                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--pretrained',
                        help='pretrained weight from checkpoint, could be imagenet22k pretrained weight')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--disable_amp', action='store_true', help='Disable pytorch amp')
    parser.add_argument('--amp-opt-level', type=str, choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used (deprecated!)')
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')

    # distributed training
    # for pytorch >= 2.0, use `os.environ['LOCAL_RANK']` instead
    # (see https://pytorch.org/docs/stable/distributed.html#launch-utility)
    if PYTORCH_MAJOR_VERSION == 1:
        parser.add_argument("--local_rank", type=int, required=True, help='local rank for DistributedDataParallel')

    # for acceleration
    parser.add_argument('--fused_window_process', action='store_true',
                        help='Fused window shift & window partition, similar for reversed part.')
    parser.add_argument('--fused_layernorm', action='store_true', help='Use fused layernorm.')
    ## overwrite optimizer in config (*.yaml) if specified, e.g., fused_adam/fused_lamb
    parser.add_argument('--optim', type=str,
                        help='overwrite optimizer if provided, can be adamw/sgd/fused_adam/fused_lamb.')

    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    return args, config


def main(config):
    dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn = build_loader(config)

    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_model(config)
    #logger.info(str(dataset_train))

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")
    if hasattr(model, 'flops'):
        flops = model.flops()
        logger.info(f"number of GFLOPs: {flops / 1e9}")

    #model.cuda()
    device = torch.device("cpu")
    model.to(device)
    model_without_ddp = model

    #optimizer = build_optimizer(config, model)
    lr=0.00013914064388085564
    optimizer= optim.AdamW(model.parameters(), lr=lr,
    weight_decay=1e-4)
    #model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.LOCAL_RANK], broadcast_buffers=False)
    #model_without_ddp = model  # No need for DistributedDataParallel
    #loss_scaler = NativeScalerWithGradNormCount()
    loss_scaler = None  # Placeholder since gradient scaling is not needed
    #if config.TRAIN.ACCUMULATION_STEPS > 1:
        #lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train) // config.TRAIN.ACCUMULATION_STEPS)
    #else:
    #lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train))
    #print(f'schedlur is {lr_scheduler}')
    #if config.AUG.MIXUP > 0.:
        #print(f'loss invalid')
        # smoothing is handled with mixup label transform
        #criterion = SoftTargetCrossEntropy()
    #elif config.MODEL.LABEL_SMOOTHING > 0.:
        #criterion = LabelSmoothingCrossEntropy(smoothing=config.MODEL.LABEL_SMOOTHING)
    #else:
    print(f'loss cross entropy')
    criterion = torch.nn.CrossEntropyLoss()
    print(f'loss method is:{criterion}')

    max_accuracy = 0.0
    '''
    if config.TRAIN.AUTO_RESUME:
        resume_file = auto_resume_helper(config.OUTPUT)
        if resume_file:
            if config.MODEL.RESUME:
                logger.warning(f"auto-resume changing resume file from {config.MODEL.RESUME} to {resume_file}")
            config.defrost()
            config.MODEL.RESUME = resume_file
            config.freeze()
            logger.info(f'auto resuming from {resume_file}')
        else:
            logger.info(f'no checkpoint found in {config.OUTPUT}, ignoring auto resume')
  
    if config.MODEL.RESUME:
        max_accuracy = load_checkpoint(config, model_without_ddp, optimizer, lr_scheduler, loss_scaler, logger)
        acc1, acc5, loss = validate(config, data_loader_val, model)
        logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")
        if config.EVAL_MODE:
            return
    
    if config.MODEL.PRETRAINED and (not config.MODEL.RESUME):
        load_pretrained(config, model_without_ddp, logger)
        acc1, acc5, loss = validate(config, data_loader_val, model)
        logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")

    if config.THROUGHPUT_MODE:
        throughput(data_loader_val, model, logger)
        return
    '''
    logger.info("Start training")
    start_time = time.time()
    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        #data_loader_train.sampler.set_epoch(epoch)
        #logger.info(f'config : {config}, ignoring auto resume')

        train_one_epoch(device,config, model, criterion, data_loader_train, optimizer, epoch, mixup_fn,
                        loss_scaler)
        #if dist.get_rank() == 0 and (epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1)):
        #save_checkpoint(config, epoch, model_without_ddp, max_accuracy, optimizer, lr_scheduler, loss_meter,
                            #logger)

        acc, loss, val_metrics = validate(device,config, data_loader_val, model)
        logger.info(f"Accuracy of the network on the {len(dataset_val)}  test images: {acc:.3f}% and epoch is : {epoch}")
        max_accuracy = max(max_accuracy, acc)
        logger.info(f'Max accuracy: {max_accuracy:.2f}%')
        logger.info(f"Precision: {val_metrics['precision']:.4f}, Recall: {val_metrics['recall']:.4f}, F1: {val_metrics['f1_score']:.4f}")
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))


def train_one_epoch(device,config, model, criterion, data_loader, optimizer, epoch, mixup_fn, loss_scaler):
    model.train()
    optimizer.zero_grad()
    
    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()
    scaler_meter = AverageMeter()
    
    start = time.time()
    end = time.time()
    for idx, (samples, targets) in enumerate(data_loader):
        #samples = samples.cuda(non_blocking=True)
        #targets = targets.cuda(non_blocking=True)
        samples = samples.to(device)
        targets = targets.to(device)
        #targets=targets.view(-1, 1)
        #targets = targets.unsqueeze(dim=-1)
        targets = targets.squeeze()
        #targets = targets.view(-1)  # This will reshape the target tensor to shape (16,)
        #targets = torch.argmax(targets, dim=1)  # Convert to class indices (shape: (16,))
        #print(f'target shape {targets.shape}')
        #print(f'target  {targets}')

        '''
        if mixup_fn is not None:
                # Ensure that samples and targets are on CPU
                samples = samples.to('cpu')
                targets = targets.to('cpu')
                samples, targets = mixup_fn(samples.to(device), targets.to(device))

        '''
        #with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
        #print(f'samples shape : {samples.shape}')  # Should output: [batch_size, 1, 28, 28]
        outputs = model(samples)
        #print(f'output shape {outputs}')

        loss = criterion(outputs, targets)
        #loss = loss / config.TRAIN.ACCUMULATION_STEPS
        '''
        for name, param in model.named_parameters():
          if param.requires_grad:
             print (name, param.data)
        print('dooooooooooooooooooooone')
        '''
        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        '''
        grad_norm = loss_scaler(loss, optimizer, clip_grad=config.TRAIN.CLIP_GRAD,
                                parameters=model.parameters(), create_graph=is_second_order,
                                update_grad=(idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0)
        
        if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
            optimizer.zero_grad()
            lr_scheduler.step_update((epoch * num_steps + idx) // config.TRAIN.ACCUMULATION_STEPS)
        loss_scale_value = loss_scaler.state_dict()["scale"]
        '''
       
        loss_meter.update(loss.item(), targets.size(0))
        # Backward pass
        optimizer.zero_grad()  # Clear gradients from previous step
        loss.backward()  # Compute gradients
        
        # Update weights
        optimizer.step()  # Apply gradients to update weights
        '''
        if grad_norm is not None:  # loss_scaler return None if not update
            norm_meter.update(grad_norm)
        scaler_meter.update(loss_scale_value)
        '''
        batch_time.update(time.time() - end)
        acc = (outputs.argmax(dim=1) == targets).float().mean()
        end = time.time()
        lr = optimizer.param_groups[0]['lr']
        wd = optimizer.param_groups[0]['weight_decay']
        '''
        logger.info(
                f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'accuracy training {acc:.3f}')
        '''
    #epoch_time = time.time() - start
    #logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")

def compute_metrics(y_true, y_pred, average='macro'):
    """
    Compute precision, recall, and F1-score.

    Args:
        y_true: Ground truth labels (list or tensor).
        y_pred: Predicted labels (list or tensor).
        average: Averaging method for multi-class ('macro', 'micro', or 'weighted').

    Returns:
        A dictionary with precision, recall, and F1-score.
    """
    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()

    precision = precision_score(y_true, y_pred, average=average)
    recall = recall_score(y_true, y_pred, average=average)
    f1 = f1_score(y_true, y_pred, average=average)

    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }
@torch.no_grad()
def validate(device,config, data_loader, model):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()
    all_preds = []
    all_labels = []

    end = time.time()
    for idx, (images, target) in enumerate(data_loader):
        #images = images.cuda(non_blocking=True)
        #target = target.cuda(non_blocking=True)
        images = images.to(device)
        target = target.to(device)
        # compute output
        #with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
        output = model(images)
        target=np.squeeze(target)
        # measure accuracy and record loss
        loss = criterion(output, target)
        #acc1, acc5 = accuracy(output, target, topk=(1,1))
        acc = (output.argmax(dim=1) == target).float().mean()

        #acc1 = reduce_tensor(acc1)
        #acc5 = reduce_tensor(acc5)
        #loss = reduce_tensor(loss)

        loss_meter.update(loss.item(), target.size(0))
        acc1_meter.update(acc.item(), target.size(0))
        #acc5_meter.update(acc5.item(), target.size(0))
        #print(f'target : {target}')
        #print(f'output : {output}')
        # Store predictions and true labels
        # Get predictions
        _, preds = torch.max(output.data, 1)
        #print(f'outputdata : {output.data}')

        #print(f'predition are : {preds} and labels are {target}')
        all_preds.append(preds)
        all_labels.append(target)
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
    # Concatenate all predictions and labels
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    #compute metrics 
    metrics = compute_metrics(all_labels, all_preds)

    '''
        if idx % config.PRINT_FREQ == 0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            logger.info(
                f'Test: [{idx}/{len(data_loader)}]\t'
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t'
                f'Acc@5 {acc5_meter.val:.3f} ({acc5_meter.avg:.3f})\t'
                f'Mem {memory_used:.0f}MB')
    '''
    logger.info(f' * Accuracy validation@ {acc1_meter.avg:.3f} ')
    return acc1_meter.avg, loss_meter.avg,metrics


@torch.no_grad()
def throughput(data_loader, model, logger):
    model.eval()

    for idx, (images, _) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        batch_size = images.shape[0]
        for i in range(50):
            model(images)
        torch.cuda.synchronize()
        logger.info(f"throughput averaged with 30 times")
        tic1 = time.time()
        for i in range(30):
            model(images)
        torch.cuda.synchronize()
        tic2 = time.time()
        logger.info(f"batch_size {batch_size} throughput {30 * batch_size / (tic2 - tic1)}")
        return


if __name__ == '__main__':
    args, config = parse_option()

    if config.AMP_OPT_LEVEL:
        print("[warning] Apex amp has been deprecated, please use pytorch amp instead!")
    '''
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
    '''
    rank = -1
    world_size = -1
    #torch.cuda.set_device(config.LOCAL_RANK)
    #torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    #torch.distributed.barrier()

    seed = config.SEED
    torch.manual_seed(seed)
    #torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    #cudnn.benchmark = True
    
    # linear scale the learning rate according to total batch size, may not be optimal
    # For single-device training, set world_size to 1
    world_size = 1
    linear_scaled_lr = config.TRAIN.BASE_LR * config.DATA.BATCH_SIZE * world_size / 512.0
    linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR * config.DATA.BATCH_SIZE * world_size / 512.0
    linear_scaled_min_lr = config.TRAIN.MIN_LR * config.DATA.BATCH_SIZE * world_size / 512.0
    # gradient accumulation also need to scale the learning rate
    if config.TRAIN.ACCUMULATION_STEPS > 1:
        linear_scaled_lr = linear_scaled_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_warmup_lr = linear_scaled_warmup_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_min_lr = linear_scaled_min_lr * config.TRAIN.ACCUMULATION_STEPS
    config.defrost()
    config.TRAIN.BASE_LR = linear_scaled_lr
    config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
    config.TRAIN.MIN_LR = linear_scaled_min_lr
    #config.freeze()
    config.AMP_ENABLE = False
    config.freeze()
    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT, name=f"{config.MODEL.NAME}")
    '''
    if dist.get_rank() == 0:
        path = os.path.join(config.OUTPUT, "config.json")
        with open(path, "w") as f:
            f.write(config.dump())
        logger.info(f"Full config saved to {path}")
    '''
    # print config
    logger.info(config.dump())
    logger.info(json.dumps(vars(args)))

    main(config)