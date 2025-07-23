import argparse
import logging
import os
import random
import shutil
import sys

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from skimage.measure import label

from dataloaders.dataset import (BaseDataSets, RandomGenerator, TwoStreamBatchSampler)
from networks.net_factory import net_factory
from utils import val_2d
from utils.CRL_utils import generate_mask_2D, supervison_loss, count_params
from utils.CRL_utils import inconsistent_region, pseudo_align_loss, mix_kl_loss, features_diversity_loss

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='C:/PycharmProjects/dataset/ACDC', help='Name of Experiment')
parser.add_argument('--exp', type=str, default='ddpcl', help='experiment_name')
parser.add_argument('--model', type=str, default='unet', help='model_name')
parser.add_argument('--pre_iterations', type=int, default=10000, help='maximum epoch number to train')
parser.add_argument('--train_iterations', type=int, default=30000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=24, help='batch_size per iteration')
parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float, default=0.01, help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list, default=[256, 256], help='patch size of network input')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--num_classes', type=int, default=4, help='output channel of network')
# label and unlabel
parser.add_argument('--labeled_bs', type=int, default=12, help='labeled_batch_size per gpu')
parser.add_argument('--labelnum', type=int, default=7, help='labeled data')
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
# -- settings
parser.add_argument('--l_weight', type=float, default=1.0, help='weight of labeled pixels')
parser.add_argument('--u_weight', type=float, default=0.5, help='weight of unlabeled pixels')
parser.add_argument('--align_weight', type=float, default=0.4, help='weight of correction loss')
parser.add_argument('--div_weight', type=float, default=0.5, help='weight of feature diversity loss')
parser.add_argument('--kl_weight', type=float, default=0.02, help='weight of kl_weight loss')
parser.add_argument('--mask_ratio', type=float, default=1/2, help='ratio of mask/image')
args = parser.parse_args()

def load_net(net, path):
    state = torch.load(str(path))
    net.load_state_dict(state['net'])

def load_net_opt(net, optimizer, path):
    state = torch.load(str(path))
    net.load_state_dict(state['net'])
    optimizer.load_state_dict(state['opt'])

def save_net_opt(net, optimizer, path):
    state = {
        'net':net.state_dict(),
        'opt':optimizer.state_dict(),
    }
    torch.save(state, str(path))

def get_ACDC_LargestCC(segmentation):
    class_list = []
    for i in range(1, 4):
        temp_prob = segmentation == i * torch.ones_like(segmentation)
        temp_prob = temp_prob.detach().cpu().numpy()
        labels = label(temp_prob)
        assert(labels.max() != 0)  # assume at least 1 CC
        largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
        class_list.append(largestCC * i)
    acdc_largestCC = class_list[0] + class_list[1] + class_list[2]
    return torch.from_numpy(acdc_largestCC).cuda()

def get_ACDC_2DLargestCC(segmentation):
    batch_list = []
    N = segmentation.shape[0]
    for i in range(0, N):
        class_list = []
        for c in range(1, 4):
            temp_seg = segmentation[i]
            temp_prob = torch.zeros_like(temp_seg)
            temp_prob[temp_seg == c] = 1
            temp_prob = temp_prob.detach().cpu().numpy()
            labels = label(temp_prob)          
            if labels.max() != 0:
                largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
                class_list.append(largestCC * c)
            else:
                class_list.append(temp_prob)
        
        n_batch = class_list[0] + class_list[1] + class_list[2]
        batch_list.append(n_batch)

    return torch.Tensor(batch_list).cuda()

def get_ACDC_masks(output, nms=0):
    probs = F.softmax(output, dim=1)
    _, probs = torch.max(probs, dim=1)
    if nms == 1:
        probs = get_ACDC_2DLargestCC(probs)      
    return probs

def patients_to_slices(dataset, patiens_num):
    ref_dict = None
    if "ACDC" in dataset:
        ref_dict = {"1": 32, "3": 68, "7": 136,
                    "14": 256, "21": 396, "28": 512, "35": 664, "70": 1312}
    elif "Prostate":
        ref_dict = {"2": 47, "4": 111, "7": 191,
                    "11": 306, "14": 391, "18": 478, "35": 940}
    else:
        print("Error")
    return ref_dict[str(patiens_num)]

def pre_train(args, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    train_iterations = args.pre_iterations
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    labeled_sub_bs, _ = int(args.labeled_bs/2), int((args.batch_size-args.labeled_bs) / 2)
    model = net_factory(in_chns=1, class_num=num_classes)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    db_train = BaseDataSets(base_dir=args.root_path,
                            split="train",
                            num=None,
                            transform=transforms.Compose([RandomGenerator(args.patch_size)]))
							
    db_val = BaseDataSets(base_dir=args.root_path, split="val")
    #db_val = BaseDataSets(base_dir=args.root_path, split="test")

    total_slices = len(db_train)
    labeled_slice = patients_to_slices(args.root_path,args.labelnum)
    print("Total slices is: {}, labeled slices is:{}".format(total_slices, labeled_slice))
    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batch_size, args.batch_size-args.labeled_bs)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=0, pin_memory=True, worker_init_fn=worker_init_fn)
    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=0)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("Start pre_training")
    logging.info("{} iterations per epoch".format(len(trainloader)))

    model.train()
    iter_num = 0
    max_epoch = train_iterations // len(trainloader) + 1
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for _ in iterator:
        for _, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

            img_a, img_b = volume_batch[:labeled_sub_bs], volume_batch[labeled_sub_bs:args.labeled_bs]
            lab_a, lab_b = label_batch[:labeled_sub_bs], label_batch[labeled_sub_bs:args.labeled_bs]
            with torch.no_grad():
                img_mask, _ = generate_mask_2D(img_a, args.mask_ratio)

            # CutMix input and label
            mix_input = img_a * img_mask + img_b * (1 - img_mask)
            mix_lab = lab_a * img_mask + lab_b * (1 - img_mask)

            # compute two model outputs
            outputs1, outputs2, feature1, feature2 = model(mix_input)

            # supervison_loss
            loss_sup1 = supervison_loss(outputs1, mix_lab, class_num=args.num_classes)
            loss_sup2 = supervison_loss(outputs2, mix_lab, class_num=args.num_classes)
            loss_sup = (loss_sup1 + loss_sup2) * args.l_weight
            
            # features_diversity_loss
            loss_div1 = features_diversity_loss(feature1, feature2)
            loss_div2 = features_diversity_loss(feature2, feature1)
            loss_div = (loss_div1 + loss_div2) * args.div_weight

            # compute the overall loss
            loss = loss_sup + loss_div

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter_num += 1
            writer.add_scalar('pre/loss_sup', loss_sup, iter_num)
            writer.add_scalar('pre/loss_div', loss_div, iter_num)
            writer.add_scalar('pre/loss', loss, iter_num)
            logging.info(
                'iteration %d : loss: %03f, loss_sup: %03f, loss_div: %03f' % (iter_num, loss, loss_sup, loss_div))

            if iter_num % 200 == 1:
                image = mix_input[1, 0:1, :, :]
                writer.add_image('pre_train/Mixed_Image', image, iter_num)
                mix_outputs = (outputs1 + outputs2) / 2
                outputs = torch.argmax(torch.softmax(mix_outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('pre_train/Mixed_Prediction', outputs[1, ...] * 50, iter_num)
                labs = mix_lab[1, ...].unsqueeze(0) * 50
                writer.add_image('pre_train/Mixed_GroundTruth', labs, iter_num)

            if iter_num > 0 and iter_num % 200 == 0:
                model.eval()
                metric_list = 0.0
                for _, sampled_batch in enumerate(valloader):
                    metric_i = val_2d.test_single_volume(sampled_batch["image"], sampled_batch["label"], model, classes=num_classes)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes-1):
                    writer.add_scalar('info/val_{}_dice'.format(class_i+1), metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/val_{}_hd95'.format(class_i+1), metric_list[class_i, 1], iter_num)
                performance = np.mean(metric_list, axis=0)[0]
                writer.add_scalar('info/val_mean_dice', performance, iter_num)

                if performance > best_performance:
                    best_performance = performance
                    save_mode_path = os.path.join(snapshot_path, 'iter_{}_dice_{}.pth'.format(iter_num, round(best_performance, 4)))
                    save_best_path = os.path.join(snapshot_path,'{}_best_model.pth'.format(args.model))
                    save_net_opt(model, optimizer, save_mode_path)
                    save_net_opt(model, optimizer, save_best_path)
                logging.info('iteration %d : mean_dice : %f' % (iter_num, performance))
                model.train()

            if iter_num >= train_iterations:
                break
        if iter_num >= train_iterations:
            iterator.close()
            break
    writer.close()

def train(args ,pre_snapshot_path, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    train_iterations = args.train_iterations
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    # pre_trained_model = os.path.join(pre_snapshot_path,'{}_best_model.pth'.format(args.model))
    pre_trained_model = os.path.join('./model/ACDC/{}_best_model.pth'.format(args.model))
    labeled_sub_bs, unlabeled_sub_bs = int(args.labeled_bs/2), int((args.batch_size-args.labeled_bs) / 2)
     
    model = net_factory(in_chns=1, class_num=num_classes)
    logging.info('===Total params: {:.3f}M'.format(count_params(model)))
    
    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    db_train = BaseDataSets(base_dir=args.root_path,split="train", num=None,
                            transform=transforms.Compose([RandomGenerator(args.patch_size)]))
							
    db_val = BaseDataSets(base_dir=args.root_path, split="val")
    #db_val = BaseDataSets(base_dir=args.root_path, split="test")
	
    total_slices = len(db_train)
    labeled_slice = patients_to_slices(args.root_path,args.labelnum)
    print("Total slices is: {}, labeled slices is:{}".format(total_slices, labeled_slice))
    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batch_size, args.batch_size-args.labeled_bs)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=0, pin_memory=True, worker_init_fn=worker_init_fn)
    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=0)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    load_net_opt(model, optimizer, pre_trained_model)
    logging.info("Loaded from {}".format(pre_trained_model))

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("Start self_training")
    logging.info("{} iterations per epoch".format(len(trainloader)))

    model.train()
    iter_num = 0
    max_epoch = train_iterations // len(trainloader) + 1
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for _ in iterator:
        for _, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            img_a, img_b = volume_batch[:labeled_sub_bs], volume_batch[labeled_sub_bs:args.labeled_bs]
            unimg_a, unimg_b = volume_batch[args.labeled_bs:args.labeled_bs + unlabeled_sub_bs], volume_batch[args.labeled_bs + unlabeled_sub_bs:]
            lab_a, lab_b = label_batch[:labeled_sub_bs], label_batch[labeled_sub_bs:args.labeled_bs]
            ulab_a, ulab_b = label_batch[args.labeled_bs:args.labeled_bs + unlabeled_sub_bs], label_batch[args.labeled_bs + unlabeled_sub_bs:]

            """ for labeled data """
            with torch.no_grad():
                img_mask, _ = generate_mask_2D(img_a, args.mask_ratio)

            # CutMix labeled input
            mixl_img = img_a * img_mask + img_b * (1 - img_mask)
            mixl_lab = lab_a * img_mask + lab_b * (1 - img_mask)
            outputs1_l, outputs2_l, features1_l, features2_l = model(mixl_img)

            # supervison_loss
            loss_sup1_l = supervison_loss(outputs1_l, mixl_lab, class_num=args.num_classes)
            loss_sup2_l = supervison_loss(outputs2_l, mixl_lab, class_num=args.num_classes)
            loss_l = (loss_sup1_l + loss_sup2_l) * args.l_weight
            
            # features_diversity_loss
            loss_div1_l = features_diversity_loss(features1_l, features2_l)
            loss_div2_l = features_diversity_loss(features2_l, features1_l)
            loss_div_l = loss_div1_l + loss_div2_l

            # pseudo_align_loss  
            with torch.no_grad():
                diff_mask_l = inconsistent_region(outputs1_l, outputs2_l)
            loss_mse1_l = pseudo_align_loss(outputs1_l, mixl_lab, diff_mask=diff_mask_l, num_classes=args.num_classes)
            loss_mse2_l = pseudo_align_loss(outputs2_l, mixl_lab, diff_mask=diff_mask_l, num_classes=args.num_classes)
            loss_mse_l = loss_mse1_l + loss_mse2_l

            """ for unlabeled data """
            with torch.no_grad():
                unoutputs1, unoutputs2, _, _ = model(volume_batch[args.labeled_bs:])
                # get pseudo label
                plab1 = get_ACDC_masks(unoutputs1, nms=1)
                plab2 = get_ACDC_masks(unoutputs2, nms=1)
                unimg_mask, _ = generate_mask_2D(unimg_a, args.mask_ratio)

            mixu_img = unimg_a * unimg_mask + unimg_b * (1 - unimg_mask)
            mixu_lab = ulab_a * unimg_mask + ulab_b * (1 - unimg_mask)
            # Supervise the cutmix portion with the sub model's pseudo label, and the rest is self-supervised.
            mixu_plab1 = plab2[:unlabeled_sub_bs] * unimg_mask + plab1[unlabeled_sub_bs:] * (1 - unimg_mask)
            mixu_plab2 = plab1[:unlabeled_sub_bs] * unimg_mask + plab2[unlabeled_sub_bs:] * (1 - unimg_mask)

            # two model outputs
            outputs1_u, outputs2_u, features1_u, features2_u = model(mixu_img)
            
            # supervison_loss, pseudo label supervised
            loss_sup_u = supervison_loss(outputs1_u, mixu_plab1, class_num=args.num_classes)
            sub_loss_sup_u = supervison_loss(outputs2_u, mixu_plab2, class_num=args.num_classes)
            loss_u = (loss_sup_u + sub_loss_sup_u) * args.u_weight
            
            # features_diversity_loss
            loss_div1_u = features_diversity_loss(features1_u, features2_u)
            loss_div2_u = features_diversity_loss(features2_u, features1_u)
            loss_div_u = (loss_div1_u + loss_div2_u) 

            loss_div = loss_div_l + loss_div_u * args.div_weight

            outputs1_u_soft = torch.softmax(outputs1_u, dim=1)
            outputs2_u_soft = torch.softmax(outputs1_u, dim=1)
            loss_cons = torch.mean((outputs1_u_soft - outputs2_u_soft) ** 2)

            # pseudo_align_loss 
            with torch.no_grad():
                diff_mask_u = inconsistent_region(outputs1_u, outputs2_u)
            loss_mse1_u = pseudo_align_loss(outputs1_u, mixu_plab1, diff_mask=diff_mask_u, num_classes=args.num_classes)
            loss_mse2_u = pseudo_align_loss(outputs2_u, mixu_plab2, diff_mask=diff_mask_u, num_classes=args.num_classes)
            loss_mse_u = loss_mse1_u + loss_mse2_u

            loss_mse = loss_mse_l + loss_mse_u * args.align_weight

            loss_kl1_u = mix_kl_loss(outputs1_u, mixu_plab1, diff_mask=diff_mask_u)
            loss_kl2_u = mix_kl_loss(outputs2_u, mixu_plab2, diff_mask=diff_mask_u)
            loss_kl = (loss_kl1_u + loss_kl2_u) * args.kl_weight

            # sys.exit()

            # The total loss
            loss = loss_l + loss_u + loss_div + loss_mse + loss_kl + loss_cons

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter_num += 1
            writer.add_scalar('Self/loss_l', loss_l, iter_num)
            writer.add_scalar('Self/loss_u', loss_u, iter_num)
            writer.add_scalar('Self/loss_div', loss_div, iter_num)
            writer.add_scalar('Self/loss_mse', loss_mse, iter_num)
            writer.add_scalar('Self/loss_kl', loss_kl, iter_num)
            writer.add_scalar('Self/loss', loss, iter_num)
            logging.info(
                'iteration %d : loss: %03f, loss_l: %03f, loss_u: %03f, loss_div: %03f, '
                'loss_mse: %03f, loss_kl: %03f, loss_cons: %03f'
                % (iter_num, loss, loss_l, loss_u, loss_div, loss_mse, loss_kl, loss_cons))

            if iter_num % 200 == 1:
                image_l = mixl_img[1, 0:1, :, :]
                writer.add_image('train/in_Image', image_l, iter_num)
                outputs_l = (outputs1_l + outputs2_l) / 2
                outputs = torch.argmax(torch.softmax(outputs_l, dim=1), dim=1, keepdim=True)
                writer.add_image('train/in_Prediction', outputs[1, ...] * 50, iter_num)
                labs = mixl_lab[1, ...].unsqueeze(0) * 50
                writer.add_image('train/in_GroundTruth', labs, iter_num)

                image_u = mixu_img[1, 0:1, :, :]
                writer.add_image('train/in_Image', image_u, iter_num)
                outputs_u = (outputs1_u + outputs2_u) / 2
                outputs = torch.argmax(torch.softmax(outputs_u, dim=1), dim=1, keepdim=True)
                writer.add_image('train/in_Prediction', outputs[1, ...] * 50, iter_num)
                labs = mixu_lab[1, ...].unsqueeze(0) * 50
                writer.add_image('train/in_GroundTruth', labs, iter_num)

            if iter_num > 0 and iter_num % 200 == 0:
                model.eval()
                metric_list = 0.0
                for _, sampled_batch in enumerate(valloader):
                    metric_i = val_2d.test_single_volume(sampled_batch["image"], sampled_batch["label"], model, classes=num_classes)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes-1):
                    writer.add_scalar('info/val_{}_dice'.format(class_i+1), metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/val_{}_hd95'.format(class_i+1), metric_list[class_i, 1], iter_num)

                performance = np.mean(metric_list, axis=0)[0]
                writer.add_scalar('info/val_mean_dice', performance, iter_num)

                if performance > best_performance:
                    best_performance = performance
                    save_mode_path = os.path.join(snapshot_path, 'iter_{}_dice_{}.pth'.format(iter_num, round(best_performance, 4)))
                    save_best_path = os.path.join(snapshot_path,'{}_best_model.pth'.format(args.model))
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best_path)

                logging.info('iteration %d : mean_dice : %f' % (iter_num, performance))
                model.train()

            if iter_num >= train_iterations:
                break
        if iter_num >= train_iterations:
            iterator.close()
            break
    writer.close()


if __name__ == "__main__":
    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    exp_name = "{}_lu{}{}_a{}_d{}_k{}_m{}".format(args.exp,
                            int(args.l_weight*10), int(args.u_weight*10),
                            int(args.align_weight*10), int(args.div_weight*10),
                            int(args.kl_weight*100),int(args.mask_ratio*10))

    pre_snapshot_path = "./model/ACDC/{}_{}_labeled/pre_train".format(exp_name, args.labelnum)
    train_snapshot_path = "./model/ACDC/{}_{}_labeled/train".format(exp_name, args.labelnum)

    for snapshot_path in [pre_snapshot_path, train_snapshot_path]:
        if not os.path.exists(snapshot_path):
            os.makedirs(snapshot_path)

    shutil.copy(sys.argv[0], train_snapshot_path + '/')

    logging.basicConfig(filename=train_snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    # -- Pre_train --
    print("Starting pre-training.")
    # pre_train(args, pre_snapshot_path)
    # -- train --
    print("Starting training.")
    train(args, pre_snapshot_path, train_snapshot_path)

    


