import os
import sys
import time
import argparse
import datetime
import numpy as np
import tqdm

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.cuda.amp import autocast, GradScaler

from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import accuracy, AverageMeter

from config import get_config
from models import build_model
from data import build_loader
from lr_scheduler import build_scheduler
from optimizer import build_optimizer
from logger import create_logger
from utils import load_checkpoint, save_checkpoint, save_checkpoint_new, get_grad_norm, auto_resume_helper, \
    reduce_tensor, load_pretrained

import warnings

warnings.filterwarnings('ignore')

from fvcore.nn import FlopCountAnalysis


def parse_option():
    parser = argparse.ArgumentParser('FLatten Transformer training and evaluation script', add_help=False)
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
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--amp', action='store_true', default=False)
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')
    parser.add_argument('--pretrained', type=str, default='')
    parser.add_argument('--find-unused-params', action='store_true', default=False)
    # parser.add_argument('--find-unused-params', default=True)

    parser.add_argument("--init_method", type=str, default='env://', help='init_method')

    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    return args, config


def main():
    os.environ["NCCL_BLOCKING_WAIT"] = "1"
    args, config = parse_option()

    rank = int(os.environ["RANK"])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    # dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    dist.init_process_group(backend='nccl', init_method=args.init_method, world_size=world_size, rank=rank)

    seed = config.SEED + dist.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.enabled = True
    cudnn.benchmark = True

    config.defrost()
    config.LOCAL_RANK = local_rank
    config.freeze()

    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=dist.get_rank(), name=f"{config.MODEL.NAME}")

    if dist.get_rank() == 0:
        path = os.path.join(config.OUTPUT, "config.json")
        with open(path, "w") as f:
            f.write(config.dump())
        logger.info(f"Full config saved to {path}")

    # print config
    logger.info(config.dump())

    _, dataset_val, data_loader_train, data_loader_val, mixup_fn = build_loader(config)

    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_model(config)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")

    model.cuda()
    logger.info(str(model))

    model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=True,
                                                find_unused_parameters=True)
    model_without_ddp = model.module

    # compute_flops(model)
    # inference_time(model)
    # throughput(data_loader_val, model, logger)

    if args.pretrained != '':
        checkpoint = torch.load(args.pretrained, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'], strict=False)

    for module in model.modules():
        if module.__class__.__name__ == 'SwinTransformerBlock':
            module.merge_bn()
        elif module.__class__.__name__ == 'PatchMerging':
            module.merge_bn()
        elif module.__class__.__name__ == 'PatchEmbed':
            module.merge_bn()
    for module in model.modules():
        if module.__class__.__name__ == 'SwinTransformer':
            module.merge_bn()
    for module in model.modules():
        print(module)
    throughput(data_loader_val, model, logger)
    acc1, acc5, loss = validate(config, data_loader_val, model, logger)
    logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")
    return


@torch.no_grad()
def validate(config, data_loader, model, logger):
    criterion = nn.CrossEntropyLoss()
    model.eval()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()

    end = time.time()
    for idx, (images, target) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        # im_show = images[20, :, :, :].permute(1, 2, 0).cpu().numpy()
        # plt.imshow(im_show)
        # plt.show()
        target = target.cuda(non_blocking=True)

        # compute output
        output = model(images)

        # measure accuracy and record loss
        loss = criterion(output, target)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        acc1 = reduce_tensor(acc1)
        acc5 = reduce_tensor(acc5)
        loss = reduce_tensor(loss)

        loss_meter.update(loss.item(), target.size(0))
        acc1_meter.update(acc1.item(), target.size(0))
        acc5_meter.update(acc5.item(), target.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (idx + 1) % config.PRINT_FREQ == 0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            logger.info(
                f'Test: [{(idx + 1)}/{len(data_loader)}]\t'
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t'
                f'Acc@5 {acc5_meter.val:.3f} ({acc5_meter.avg:.3f})\t'
                f'Mem {memory_used:.0f}MB')
    logger.info(f' * Acc@1 {acc1_meter.avg:.3f} Acc@5 {acc5_meter.avg:.3f}')
    return acc1_meter.avg, acc5_meter.avg, loss_meter.avg


@torch.no_grad()
def compute_flops(model):
    model.eval()

    dummy_input = torch.rand(1, 3, 224, 224).cuda()
    flops = FlopCountAnalysis(model, dummy_input).total()
    print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    return


@torch.no_grad()
def throughput(data_loader, model, logger):
    model.eval()

    for _, (images, _) in enumerate(data_loader):
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


@torch.no_grad()
def inference_time(model, repetitions=300):
    model.eval()

    dummy_input = torch.rand(1, 3, 224, 224).cuda()

    print('warm up ...\n')
    for _ in range(100):
        _ = model(dummy_input)

    torch.cuda.synchronize()
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    timings = np.zeros((repetitions, 1))

    print('testing ...\n')
    for rep in tqdm.tqdm(range(repetitions)):
        starter.record()
        _ = model(dummy_input)
        ender.record()
        torch.cuda.synchronize()
        curr_time = starter.elapsed_time(ender)
        timings[rep] = curr_time
    avg = timings.sum() / repetitions
    print(timings.max())
    print(timings.min())
    std = np.std(timings)
    print('\navg={}\n'.format(avg))
    print('\nstd={}\n'.format(std))
    return


if __name__ == '__main__':
    main()
