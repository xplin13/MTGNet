import os
import torch
import numpy as np
import cv2
import SSIM
import dublur_data
# from models.MIMOUNet_w_eframe import MIMOUNetPlus, MIMOUNetPlus_eframe_downsample
# from models.MIMOUNet_w_point import MIMOUNetPlus_MS_point_eframe_best, MIMOUNetPlus_MS_point_eframe_best_dialate, MIMOUNetPlus_MS_point_MS_eframe_best_dialate_croatt_two_weight, MIMOUNetPlus_MS_point_eframe_best_dialate_croatt_two_weight, MIMOUNetPlus_MS_point_eframe_best_dialate_croatt, MIMOUNetPlus_MS_point_eframe_best_test, MIMOUNetPlus_MS_point_wo_eframe, MIMOUNetPlus_MS_point_eframe_infer, MIMOUNetPlus_eframe_downsample_auto_mask_first_stage, MIMOUNetPlus_MS_point_eframe_auto_weight, MIMOUNetPlus_MS_point_eframe_auto_weight_cross_att, MIMOUNetPlus_MS_point_eframe_auto_weight_cross_att_dialate
from models.MTGNet import *
from metrics import *
import lpips


# loss_fn_vgg = lpips.LPIPS(net='alex', pretrained=False)
# loss_fn_vgg.load_state_dict(torch.load('./alex.pth', map_location='cpu'))
# loss_fn_vgg.eval().cuda()

loss_fn_vgg = lpips.LPIPS(net='alex').eval().to()

def lpips_cal(img1,img2):
    img1 = (torch.tensor(img1.transpose(2,0,1))).to(torch.float32).unsqueeze(0)
    img2 = (torch.tensor(img2.transpose(2,0,1))).to(torch.float32).unsqueeze(0)
    re = loss_fn_vgg(img1, img2).detach().numpy()
    return re

def main():
    model = MIMOUNetPlus_MS_point_eframe_best_dialate_mlp_right(num_res=20)
    # model = MIMOUNetPlus_MS_point_eframe_best_test_att(num_res=20)
    # model = MIMOUNetPlus_eframe_downsample_auto_mask_first_stage(num_res=20)
    model = model.cuda()


    h5_filename = "D:\PhDcareer\Fain_gain_temp_deblur\data/Ev-REDS/test_f32_all_30_downsample_frame_all_point_1024_high_time4_confirm/"
    # h5_filename = "./data/Ev-REDS/test_f32_all_30_downsample_frame_all_point_1024_high_time4_confirm/"
    output_path = "./output/test/EV/ablation_TCSVT/R3/"
    # h5_filename = "./data/HS-ERGB/test_f32_all_30_downsample_frame_all_point_1024/"
    # # output_path = "./output/test/HS-ERGB/best/"
    # output_path = "./output/test/HS-ERGB/test_mlp_train_hs_w_ev_pre_last/"
    # output_path = "./output/test/HS-ERGB/MIMOUNetPlus_MS_point_eframe_best_dialate_posatt_intensity/"
    # output_path = "./output/test/HS-ERGB/MIMOUNetPlus_MS_point_eframe_best_dialate_test_att/"
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    dataset_test = dataloader_per_sample.H5Dataset_eframe_point_rand_crop(directory_path=h5_filename, mode='inference', first_inference=True)
    testDataLoader = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle= False, num_workers=0, drop_last=True)

    # model_path = "./checkpoint/ssim_mes_fft_adam_frame_w_MS_point_eframe_1024_downsample_MIMO_unet_30_pretrain_resblock2_HSERGB_best_dialate_res/last_checkpoint.pth"
    # model_path = "./checkpoint/good_performance_two_weight/last_checkpoint.pth"
    # model_path = "./checkpoint/0.7498/last_checkpoint.pth"

    # model_path = "./checkpoint/ssim_mae_adam_frame_w_MS_point_eframe_1024_downsample_MIMO_unet_30_pretrain_resblock2_EvRED_best_dialate_res/last_checkpoint.pth"
    # model_path = "./checkpoint/hpc/HSERGB_NO_PRE/ssim_mae_fft_adam_frame_w_MS_point_eframe_1024_downsample_MIMO_unet_30_pretrain_resblock2_HSERGB_best_dialate_res_no_pretrain/last_checkpoint.pth"
    # model_path = "./checkpoint/HS-ERGB_best/last_checkpoint.pth"
    # model_path = "./checkpoint/hpc/EV/ssim_mae_fft_adam_frame_w_MS_point_eframe_1024_downsample_MIMO_unet_30_pretrain_resblock2_EvRED_best_dialate_res_last_mlp/last_epoch_checkpoint.pth"
    # model_path = "./checkpoint/rebuttal/hs_mlp_right/last_epoch_checkpoint.pth"
    # model_path = "D:/PhDcareer/Fain_gain_temp_deblur/checkpoint/HS-ERGB_best/last_checkpoint.pth"
    model_path = "./checkpoint/ssim_mae_MTG_right_ablation_range_7/checkpoint.pth"
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # total_ssim = 0
    # total_psnr = 0
    count = 0
    psnr_val_rgb = []
    ssim_val_rgb = []
    lpips_val_rgb = []

    with torch.no_grad():
        for batch_idx, (blur, sharp, event, point) in enumerate(testDataLoader):
            blur, sharp, event_frame, event_point = blur.cuda(), sharp, event.cuda(), point.cuda()
            output, _ = model(blur, event_frame, event_point)
            # output = model(blur, event_frame)
            # output = torch.clamp(output, 0, 1)
            output = torch.clamp(output[2], 0, 1)

            # 保存输出图像
            output = output.permute(0, 2, 3, 1)
            output = output.squeeze(0).cpu().detach().numpy()
            output = (output * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(output_path, f"output_{batch_idx}.png"), output)

            # 计算SSIM和PSNR
            sharp_ = sharp.permute(0, 2, 3, 1)
            sharp_np = sharp_.squeeze(0).numpy()
            sharp_np = (sharp_np * 255).astype(np.uint8)

            # 计算SSIM和PSNR
            current_psnr, current_ssim = compare_psnr(sharp_np, output), compare_ssim(sharp_np, output)

            lpips_value = lpips_cal(sharp_np, output)
            # total_ssim = total_ssim + current_ssim
            # total_psnr = total_psnr + current_psnr

            count += 1
            print('cnt=', count)

            ssim_val_rgb.append(current_ssim)
            psnr_val_rgb.append(current_psnr)
            lpips_val_rgb.append(lpips_value)

        # 计算平均SSIM和PSNR
        # avg_ssim = total_ssim / count
        # avg_psnr = total_psnr / count

        avg_ssim = np.mean(ssim_val_rgb)
        avg_psnr = np.mean(psnr_val_rgb)
        avg_lpips = np.mean(lpips_val_rgb)

        print(f"Average SSIM: {avg_ssim:.4f}")
        print(f"Average PSNR: {avg_psnr:.4f}")
        print(f"Average PSNR: {avg_lpips:.4f}")

if __name__ == "__main__":
    main()