import os
import sys
import argparse
import dublur_data
from metrics import *
from torch.utils.tensorboard import SummaryWriter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))
import time
import cv2
import random

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument("--log_path", type=str, default='./deblur_log/',
                        help="path to eyetracking_log")
    parser.add_argument('--train_h5_path', type=str,
                        default='./data/Ev-REDS/train/', help='train_data')
    parser.add_argument('--test_h5_path', type=str,
                        default='./data/Ev-REDS/test/', help='test_data')
    parser.add_argument('--save_path', type=str,
                        default='./checkpoint/',
                        help='model_path')
    parser.add_argument('--pretrain_model_path', type=str, default='./pretrain/MIMO-UNetPlus.pkl', help='model_path')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size in training')
    parser.add_argument('--sensor_width', type=int, default=1280, help='sensor width')
    parser.add_argument('--sensor_height', type=int, default=640, help='sensor height')
    parser.add_argument('--epoch', default=120, type=int, help='number of epoch in training')
    parser.add_argument('--loss', default='MS_ssim_mae_fft', type=str, help='loss')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    return parser.parse_args()


def validate(epoch, net, val_loader, criterion, output_path):
    net.eval()
    total_loss = 0.0
    total_ssim = 0.0
    total_psnr = 0.0
    cnt = 0
    with torch.no_grad():
        for batch_idx, (blur, sharp, event_frame, event_point) in enumerate(val_loader):
            outputs, _ = net(blur.cuda(), event_frame.cuda(), event_point.cuda())

            if epoch % 10 == 0 or epoch == (args.epoch - 1):
                output1_ = outputs[2].permute(0, 2, 3, 1)
                output1_ = torch.clamp(output1_, min=0, max=1)
                output1_np = output1_.squeeze(0).cpu().detach().numpy()
                output1_np = (output1_np * 255).astype(np.uint8)
                cv2.imwrite(os.path.join(output_path, f"output_{batch_idx}.png"), output1_np)

            sharp = sharp.cuda()
            loss = criterion(outputs, sharp)

            total_loss += loss.item()

            output_np = outputs[2].squeeze(0).permute(1, 2, 0)
            output_np = output_np.cpu().detach().numpy()
            output_np = (output_np * 255).astype(np.uint8)
            sharp_np = sharp.squeeze(0).permute(1, 2, 0)
            sharp_np = sharp_np.cpu().detach().numpy()
            sharp_np = (sharp_np * 255).astype(np.uint8)
            current_psnr, current_ssim = compare_psnr(sharp_np, output_np), compare_ssim(sharp_np, output_np)
            total_ssim = total_ssim + current_ssim
            total_psnr = total_psnr + current_psnr
            cnt += 1

        avg_ssim = total_ssim / cnt
        avg_psnr = total_psnr / cnt

    return total_loss / len(val_loader), avg_ssim, avg_psnr


def train(epoch, net, trainloader, optimizer, criterion, output_path):
    net.train()
    ###############
    total_loss = 0.0
    ###############

    for batch_idx, (blur, sharp, event_frame, event_point) in enumerate(trainloader):
        blur, sharp, event_frame, event_point = blur.cuda(), sharp.cuda(), event_frame.cuda(), event_point.cuda()
        optimizer.zero_grad()
        logits, point_vis = net(blur, event_frame, event_point)
        loss = criterion(logits, sharp)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if epoch % 10 == 0 or epoch == (args.epoch - 1):
            output_ = logits[2][0].unsqueeze(0).permute(0, 2, 3, 1)
            output_ = torch.clamp(output_, min=0, max=1)
            output_np = output_.squeeze(0).cpu().detach().numpy()
            output_np = (output_np * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(output_path, f"output_{batch_idx}.png"), output_np)

    return net, total_loss / len(trainloader)


def main(args):
    set_seed(513)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    dataset = dublur_data.H5Dataset_eframe_point_rand_crop(directory_path=args.train_h5_path, mode='train')
    trainDataLoader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4,
                                                  drop_last=True)

    dataset_test = dublur_data.H5Dataset_eframe_point_rand_crop(directory_path=args.test_h5_path,
                                                                          mode='inference')
    testDataLoader = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=4,
                                                 drop_last=True)

    '''MODEL LOADING'''

    if 1:
        from models.MTGNet import MIMOUNetPlus_MS_point_eframe_best_dialate_mlp_right
        classifier = MIMOUNetPlus_MS_point_eframe_best_dialate_mlp_right(num_res=20,
                                                                      pretrained_path=args.pretrain_model_path)

        if args.loss == "MS_ssim_mae_fft":
            criterion = L1_mae_fft_HybridLoss(alpha=0.5)
        elif args.loss == "PSNR":
            criterion = PSNRLoss(loss_weight=0.5)
        elif args.loss == "CharbonnierLoss":
            criterion = CharbonnierLoss()
        else:
            raise ValueError("Invalid loss name")

        if not args.use_cpu:
            classifier = classifier.cuda()


        start_epoch = 0

        optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-08,
                                     weight_decay=0.0005, amsgrad=False)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80, 100])

        best_psnr = float("inf")
        writer = SummaryWriter(args.eyetracking_log_path + '/ssim_mae_MTG_right_ablation_range_7')
        output_path = './training_process/train/ssim_mae_MTG_right_ablation_range_7/'
        point_path = './training_process/train/ssim_mae_MTG_right_ablation_range_7/'
        output_path_test = './training_process/test/ssim_mae_MTG_right_ablation_range_7/'
        point_path_test = './training_process/test/ssim_mae_MTG_right_ablation_range_7/'
        os.makedirs(output_path, exist_ok=True)
        os.makedirs(output_path_test, exist_ok=True)
        os.makedirs(point_path, exist_ok=True)
        os.makedirs(point_path_test, exist_ok=True)

        for epoch in range(start_epoch, args.epoch):

            classifier = classifier.train()

            scheduler.step(epoch)

            net, train_loss = train(epoch, classifier, trainDataLoader, optimizer, criterion, output_path=output_path)

            writer.add_scalar('Train/train_loss', train_loss, epoch)

            val_loss, avg_ssim, avg_psnr = validate(epoch, net, testDataLoader, criterion, output_path=output_path_test)
            if avg_psnr > best_psnr:
                best_psnr = avg_psnr
                if not os.path.exists(args.save_path):
                    os.makedirs(args.save_path)
                torch.save(net.state_dict(), os.path.join(args.save_path, "best_checkpoint.pth"))

            os.makedirs(args.save_path, exist_ok=True)
            torch.save(net.state_dict(), os.path.join(args.save_path, "checkpoint.pth"))


            print(f"Epoch {epoch + 1}/{args.epoch}: Train Loss: {train_loss:.4f}")

        writer.close()


if __name__ == '__main__':
    args = parse_args()
    main(args)
