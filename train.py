# -*- coding:UTF-8 -*-

import os
import sys
import torch
import datetime
import torch.utils.data
import numpy as np
import time

from tqdm import tqdm

from configs import regformer_args
from tools.excel_tools import SaveExcel
from tools.euler_tools import quat2mat
from tools.logger_tools import log_print, creat_logger
from kitti_pytorch import points_dataset
from regformer_model import regformer_model, get_loss
from tools.collate_functions import collate_pair



args = regformer_args()

'''CREATE DIR'''
base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(base_dir)
# experiment dir
experiment_dir = os.path.join(base_dir, 'experiment')
if not os.path.exists(experiment_dir): os.makedirs(experiment_dir)

### task dir for one experiment ###
if not args.task_name:
    file_dir = os.path.join(experiment_dir, '{}_KITTI_{}'.format(args.model_name, str(
        datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))))
else:
    file_dir = os.path.join(experiment_dir, args.task_name)
if not os.path.exists(file_dir): os.makedirs(file_dir)
# eval dir
eval_dir = os.path.join(file_dir, 'eval')
if not os.path.exists(eval_dir): os.makedirs(eval_dir)
# log dir
log_dir = os.path.join(file_dir, 'logs')
if not os.path.exists(log_dir): os.makedirs(log_dir)
# checkpoint dir
checkpoints_dir = os.path.join(file_dir, 'checkpoints/regformer')
if not os.path.exists(checkpoints_dir): os.makedirs(checkpoints_dir)

os.system('cp %s %s' % ('train.py', log_dir))
os.system('cp %s %s' % ('configs.py', log_dir))
os.system('cp %s %s' % ('regformer_model.py', log_dir))
os.system('cp %s %s' % ('conv_util.py', log_dir))
os.system('cp %s %s' % ('kitti_pytorch.py', log_dir))

'''LOG'''
def calc_error_np(pred_R, pred_t, gt_R, gt_t):
    tmp = (np.trace(pred_R.transpose().dot(gt_R))-1)/2
    tmp = np.clip(tmp, -1.0, 1.0)
    L_rot = np.arccos(tmp)
    L_rot = 180 * L_rot / np.pi
    L_trans = np.linalg.norm(pred_t - gt_t)
    return L_rot, L_trans

def main():

    global args

    train_dir_list = [0, 1, 2, 3, 4, 5]
    val_dir_list = [6, 7]
    test_dir_list = [8, 9, 10]

    logger = creat_logger(log_dir, args.model_name)
    logger.info('----------------------------------------TRAINING----------------------------------')
    logger.info('PARAMETER ...')
    logger.info(args)

    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

    # Establish an excel to retain results
    excel_eval = SaveExcel(test_dir_list, log_dir)
    model = regformer_model(args, args.batch_size, args.H_input, args.W_input, args.is_training)

    # train set
    train_dataset = points_dataset(
        is_training = 1,
        num_point=args.num_points,
        data_dir_list=train_dir_list,
        config=args,
        data_keep=args.data_keep
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate_pair,
        pin_memory=True,
        worker_init_fn=lambda x: np.random.seed((torch.initial_seed()) % (2 ** 32))
    )

    if args.multi_gpu is not None:
        device_ids = [int(x) for x in args.multi_gpu.split(',')]
        torch.backends.cudnn.benchmark = True
        model = torch.nn.DataParallel(model, device_ids=device_ids)
        model.cuda(device_ids[0])
        log_print(logger, 'multi gpu are:' + str(args.multi_gpu))
    else:

        torch.backends.cudnn.benchmark = True
        torch.cuda.set_device(args.gpu)
        model.cuda()
        log_print(logger, 'just one gpu is:' + str(args.gpu))

    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate,
                                    momentum=args.momentum)
    elif args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999),
                                     eps=1e-08, weight_decay=args.weight_decay)
    optimizer.param_groups[0]['initial_lr'] = args.learning_rate
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_stepsize,
                                                gamma=args.lr_gamma, last_epoch=-1)

    if args.ckpt is not None:
        checkpoint = torch.load(args.ckpt)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['opt_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        init_epoch = checkpoint['epoch']
        log_print(logger, 'load model {}'.format(args.ckpt))

    else:
        init_epoch = 0
        log_print(logger, 'Training from scratch')

    if args.eval_before == 1:
        eval_pose(model, test_dir_list, init_epoch)
        excel_eval.update(eval_dir)

    best_train_loss = float('inf')
    best_val_loss = float('inf')

    for epoch in range(init_epoch + 1, args.max_epoch):
        total_loss = 0
        total_seen = 0
        optimizer.zero_grad()

        for i, data in tqdm(enumerate(train_loader, 0), total=len(train_loader), smoothing=0.9):
            
            torch.cuda.synchronize()
            start_train_one_batch = time.time()

            pos2, pos1, T_gt, T_trans, T_trans_inv, Tr = data


            torch.cuda.synchronize()
            # print('load_data_time: ', time.time() - start_train_one_batch)
            pos2 = [b.cuda() for b in pos2]
            pos1 = [b.cuda() for b in pos1]
            T_trans = T_trans.cuda().to(torch.float32)
            T_trans_inv = T_trans_inv.cuda().to(torch.float32)
            T_gt = T_gt.cuda().to(torch.float32)
            model = model.train()

            torch.cuda.synchronize()
            # print('load_data_time + model_trans_time: ', time.time() - start_train_one_batch)
            l0_q, l0_t, l1_q, l1_t, l2_q, l2_t, l3_q, l3_t, pc1_ouput, q_gt, t_gt, w_x, w_q = model(pos2, pos1, T_gt, T_trans, T_trans_inv)
            loss = get_loss(l0_q, l0_t, l1_q, l1_t, l2_q, l2_t, l3_q, l3_t, q_gt, t_gt, w_x, w_q)

            torch.cuda.synchronize()
            # print('load_data_time + model_trans_time + forward ', time.time() - start_train_one_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            torch.cuda.synchronize()
            # print('load_data_time + model_trans_time + forward + back_ward ', time.time() - start_train_one_batch)

            if args.multi_gpu is not None:
                total_loss += loss.mean().cpu().data * args.batch_size
            else:
                total_loss += loss.cpu().data * args.batch_size
            total_seen += args.batch_size


        # Adjusting lr
        train_loss = total_loss / total_seen
        log_print(logger, 'EPOCH {} train mean loss: {:04f}'.format(epoch, float(train_loss)))
        # val_loss = val_pose(model, val_dir_list, epoch)

        if train_loss < best_train_loss:
            torch.save(model.state_dict(), os.path.join(checkpoints_dir, 'best_train.pth'))
            best_train_loss = train_loss
            best_train_epoch = epoch #+ 1
        # if val_loss < best_val_loss:
        #     torch.save(model.state_dict(), os.path.join(checkpoints_dir, 'best_val.pth'))
        #     best_val_loss = val_loss
        #     best_val_epoch = epoch #+ 1

        # print('Best train epoch: {} Best train loss: {:.4f} Best val epoch: {} Best val loss: {:.4f}'.format(
        #     best_train_epoch, best_train_loss, best_val_epoch, best_val_loss
        # ))

        scheduler.step()
        lr = max(optimizer.param_groups[0]['lr'], args.learning_rate_clip)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        if epoch % 2 == 0:
            save_path = os.path.join(checkpoints_dir,
                                     '{}_{:03d}_{:04f}.pth.tar'.format(model.__class__.__name__, epoch, float(train_loss)))
            torch.save({
                'model_state_dict': model.module.state_dict() if args.multi_gpu else model.state_dict(),
                'opt_state_dict': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch

            }, save_path)
            log_print(logger, 'Save {}...'.format(model.__class__.__name__))

            eval_pose(model, test_dir_list, epoch)
            excel_eval.update(eval_dir)



def val_pose(model, val_list, epoch):
    total_loss = 0
    count = 0
    for item in val_list:
        val_dataset = points_dataset(
            is_training=0,
            num_point=args.num_points,
            data_dir_list=[item],
            config=args,
            data_keep=args.data_keep
        )

        val_loader = torch.utils.data.DataLoader(val_dataset,
                                batch_size=args.eval_batch_size,
                                num_workers=args.workers,
                                shuffle=False,
                                collate_fn=collate_pair,
                                pin_memory=True,
                                worker_init_fn=lambda x: np.random.seed((torch.initial_seed()) % (2 ** 32))) #drop_last


        with torch.no_grad():
            for i, data in tqdm(enumerate(val_loader), total=len(val_loader), smoothing=0.9):
                pos2, pos1, T_gt, T_trans, T_trans_inv, Tr = data
                pos2 = [b.cuda() for b in pos2]
                pos1 = [b.cuda() for b in pos1]
                T_trans = T_trans.cuda().to(torch.float32)
                T_trans_inv = T_trans_inv.cuda().to(torch.float32)
                T_gt = T_gt.cuda().to(torch.float32)
                model = model.eval()

                l0_q, l0_t, l1_q, l1_t, l2_q, l2_t, l3_q, l3_t, pc1_ouput, q_gt, t_gt, w_x, w_q = model(pos2, pos1, T_gt, T_trans, T_trans_inv)
                loss = get_loss(l0_q, l0_t, l1_q, l1_t, l2_q, l2_t, l3_q, l3_t, q_gt, t_gt, w_x, w_q)
                total_loss += loss.item()
                count += args.eval_batch_size
    total_loss = total_loss / count
    return total_loss


def eval_pose(model, test_list, epoch):
    for item in test_list:
        test_dataset = points_dataset(
            is_training=0,
            num_point=args.num_points,
            data_dir_list=[item],
            config=args,
            data_keep = args.data_keep
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_workers=args.workers,
            collate_fn=collate_pair,
            pin_memory=True,
            worker_init_fn=lambda x: np.random.seed((torch.initial_seed()) % (2 ** 32))
        )

        line = 0
        total_time = 0
        trans_error_list = []
        rot_error_list = []
        trans_thresh = 2.0
        rot_thresh = 5.0
        success_idx = []

        for batch_id, data in tqdm(enumerate(test_loader), total=len(test_loader), smoothing=0.9):
            idx = 0
            torch.cuda.synchronize()
            start_prepare = time.time()
            pos2, pos1, T_gt, T_trans, T_trans_inv, Tr = data

            torch.cuda.synchronize()
            # print('data_prepare_time: ', time.time() - start_prepare)

            pos2 = [b.cuda() for b in pos2]
            pos1 = [b.cuda() for b in pos1]
            T_trans = T_trans.cuda().to(torch.float32)
            T_trans_inv = T_trans_inv.cuda().to(torch.float32)
            T_gt = T_gt.cuda().to(torch.float32)
            model = model.eval()

            with torch.no_grad():

                torch.cuda.synchronize()
                start_time = time.time()

                l0_q, l0_t, l1_q, l1_t, l2_q, l2_t, l3_q, l3_t, pc1_ouput, q_gt, t_gt, w_x, w_q = model(pos2, pos1, T_gt, T_trans,T_trans_inv)

                torch.cuda.synchronize()
                total_time += (time.time() - start_time)

                pc1 = pc1_ouput.cpu().numpy()
                pred_q = l0_q.cpu().numpy()
                pred_t = l0_t.cpu().numpy()

                q_gt = q_gt.cpu().numpy()
                t_gt = t_gt.cpu().numpy()

                # deal with a batch_size
                for n0 in range(pc1.shape[0]):

                    total_idx = batch_id * args.eval_batch_size + idx

                    cur_Tr = Tr[n0, :, :]

                    qq = pred_q[n0:n0 + 1, :]
                    qq = qq.reshape(4)
                    RR = quat2mat(qq)
                    tt = pred_t[n0:n0 + 1, :]
                    tt = tt.reshape(3, 1)

                    gt_q = q_gt[n0:n0 + 1, :]
                    gt_q = gt_q.reshape(4)
                    gt_R = quat2mat(gt_q)
                    gt_t = t_gt[n0:n0 + 1, :]
                    gt_t = gt_t.reshape(3, 1)
                    filler = np.array([0.0, 0.0, 0.0, 1.0])
                    filler = np.expand_dims(filler, axis=0)  ##1*4
                    pred_T = np.concatenate([np.concatenate([gt_R, gt_t], axis=-1), filler], axis=0)


                    rot_error, trans_error = calc_error_np(RR, tt, gt_R, gt_t)
                    # print(rot_error)
                    # print(trans_error)
                    trans_error_list.append(trans_error)
                    rot_error_list.append(rot_error)
                    if trans_error < trans_thresh and rot_error < rot_thresh:
                        success_idx.append(total_idx)

                    idx += 1

                    filler = np.array([0.0, 0.0, 0.0, 1.0])
                    filler = np.expand_dims(filler, axis=0)  ##1*4

                    TT = np.concatenate([np.concatenate([RR, tt], axis=-1), filler], axis=0)

                    TT = np.matmul(cur_Tr, TT)
                    TT = np.matmul(TT, np.linalg.inv(cur_Tr))

                    if line == 0:
                        T_final = TT
                        T = T_final[:3, :]
                        T = T.reshape(1, 1, 12)
                        line += 1
                    else:
                        T_final = np.matmul(T_final, TT)
                        T_current = T_final[:3, :]
                        T_current = T_current.reshape(1, 1, 12)
                        T = np.append(T, T_current, axis=0)

        success_rate = len(success_idx) / total_idx
        trans_error_array = np.array(trans_error_list)
        rot_error_array = np.array(rot_error_list)
        trans_mean = np.mean(trans_error_array[success_idx])
        trans_std = np.std(trans_error_array[success_idx])
        rot_mean = np.mean(rot_error_array[success_idx])
        rot_std = np.std(rot_error_array[success_idx])
        print('Registration Recall {}:'.format(item) + str(success_rate * 100) + '%')
        print('trans_mean {}:'.format(item) + str(trans_mean))
        print('trans_std {}:'.format(item) + str(trans_std))
        print('rot_mean {}:'.format(item) + str(rot_mean))
        print('rot_std {}:'.format(item) + str(rot_std))


        avg_time = total_time / 4541
        # print('avg_time: ', avg_time)

        data_dir = os.path.join(eval_dir, 'regformer_' + str(item).zfill(2))
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        save_txt = os.path.join(data_dir, 'reg_output.txt')
        with open(save_txt, 'a+') as tt:
            tt.write('epoch is: {:d} \n'.format(epoch))
            tt.write('Registration Recall(%): {0:.4f}\n'.format(success_rate * 100))
            tt.write('trans_mean: {0:.4f} \n'.format(trans_mean))
            tt.write('trans_std: {0:.4f} \n'.format(trans_std))
            tt.write('rot_mean: {0:.4f} \n'.format(rot_mean))
            tt.write('rot_std: {0:.4f} \n'.format(rot_std))
    return 0


if __name__ == '__main__':
    main()
