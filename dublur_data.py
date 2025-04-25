import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
import os

class H5Dataset_eframe_point_rand_crop(Dataset):
    def __init__(self, directory_path, mode='train', first_inference=True):
        self.directory_path = directory_path
        self.file_list = [os.path.join(directory_path, f) for f in os.listdir(directory_path) if f.endswith('.h5')]
        self.mode = mode
        self.first_inference = first_inference
        self.indices_cache = {}

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        with h5py.File(file_path, 'r') as f:
            data = {}
            for group_name in f:
                group = f[group_name]
                x = group['x'][:]
                y = group['y'][:]
                t = group['t'][:]
                blur = group['blur'][:]
                sharp = group['sharp'][:]
                e_frame = group['frames'][:]
                sharp_time_range = group['sharp_time_range'][:]

            if self.mode == 'train':
                return self.process_train(x, y, t, blur, sharp, e_frame, sharp_time_range)
            else:
                return self.process_inference(x, y, t, blur, sharp, e_frame, sharp_time_range, idx)

    def process_train(self, x, y, t, blur, sharp, e_frame, sharp_time_range, k=512):
        h, w = e_frame.shape[1:]
        crop_size = k

        summed_frame = np.sum(e_frame, axis=0)

        valid_summed_frame = summed_frame[crop_size // 2:h - crop_size // 2, crop_size // 2:w - crop_size // 2]

        threshold = np.percentile(valid_summed_frame, 80)
        high_value_pixels = np.where(valid_summed_frame > threshold)

        high_value_pixels = (high_value_pixels[0] + crop_size // 2, high_value_pixels[1] + crop_size // 2)

        if len(high_value_pixels[0]) > 0:
            selected_index = np.random.choice(len(high_value_pixels[0]))
            center_y, center_x = high_value_pixels[0][selected_index], high_value_pixels[1][selected_index]
        else:
            raise ValueError("No high-value pixels found in the top 10% brightness range.")

        start_y = center_y - crop_size // 2
        start_x = center_x - crop_size // 2

        crop_blur = blur[:, start_y:start_y + k, start_x:start_x + k]
        crop_sharp = sharp[:, start_y:start_y + k, start_x:start_x + k]
        crop_e_frame = e_frame[:, start_y:start_y + k, start_x:start_x + k]

        crop_e_frame_norm = crop_e_frame / (crop_e_frame.max(axis=(1, 2), keepdims=True).astype(np.float32) + 1e-15)

        min_time = sharp_time_range[0]
        max_time = sharp_time_range[-1]
        time_segments = np.linspace(min_time, max_time, 31)
        all_sampled_events = []
        # cnt = 0

        for i in range(0, len(time_segments) - 1):
            start, end = time_segments[i], time_segments[i + 1]
            mask = (t >= start) & (t < end) & (x >= start_x) & (x < start_x + k) & (y >= start_y) & (y < start_y + k)
            x_in_range = x[mask]
            y_in_range = y[mask]
            t_in_range = t[mask]

            if len(x_in_range) == 0:
                step = 1
                while len(x_in_range) < 1024:
                    expanded_start = max(min_time, start - step * (end - start))
                    expanded_end = min(max_time, end + step * (end - start))
                    mask = (t >= expanded_start) & (t < expanded_end) & (x >= start_x) & (x < start_x + k) & (
                            y >= start_y) & (y < start_y + k)
                    x_in_range = x[mask]
                    y_in_range = y[mask]
                    t_in_range = t[mask]
                    step += 1
                    if expanded_start == min_time and expanded_end == max_time:
                        break

            num_events = len(x_in_range)

            if num_events > 1024:
                indices = np.random.choice(num_events, 1024, replace=False)
            else:
                indices = np.random.choice(num_events, 1024, replace=True)

            sample_x = x_in_range[indices]
            sample_y = y_in_range[indices]
            sample_t = t_in_range[indices]

            sorted_indices = np.argsort(sample_t)
            sorted_t = sample_t[sorted_indices]
            norm_t = (sorted_t - np.min(sorted_t)) / ((np.max(sorted_t) - np.min(sorted_t)) + 1e-15)
            sorted_x = sample_x[sorted_indices]
            norm_x = (sorted_x - start_x) / k
            sorted_y = sample_y[sorted_indices]
            norm_y = (sorted_y - start_y) / k

            sampled_events = np.stack((norm_x, norm_y, norm_t), axis=-1)
            all_sampled_events.append(sampled_events)

        if all_sampled_events:
            combined_events = np.array(all_sampled_events)
        else:
            combined_events = np.array([])

        return torch.from_numpy(crop_blur).float(), torch.from_numpy(crop_sharp).float(), \
            torch.from_numpy(crop_e_frame_norm).float(), torch.from_numpy(combined_events).float()

    def process_inference(self, x, y, t, blur, sharp, e_frame, sharp_time_range):
        h, w = blur.shape[1:]

        min_time = sharp_time_range[0]
        max_time = sharp_time_range[-1]
        time_segments = np.linspace(min_time, max_time, 31)  # Divide into 10 segments
        all_sampled_events = []

        for i in range(0, len(time_segments) - 1):
            start, end = time_segments[i], time_segments[i + 1]
            mask = (t >= start) & (t < end)
            x_in_range = x[mask]
            y_in_range = y[mask]
            t_in_range = t[mask]

            if len(x_in_range) == 0:
                step = 1
                while len(x_in_range) < 1024:
                    expanded_start = max(min_time, start - step * (end - start))
                    expanded_end = min(max_time, end + step * (end - start))
                    mask = (t >= expanded_start) & (t < expanded_end)
                    x_in_range = x[mask]
                    y_in_range = y[mask]
                    t_in_range = t[mask]
                    step += 1
                    if expanded_start == min_time and expanded_end == max_time:
                        break

            num_events = len(x_in_range)
            if num_events > 1024:
                indices = np.random.choice(num_events, 1024, replace=False)
            else:
                indices = np.random.choice(num_events, 1024, replace=True)

            sample_x = x_in_range[indices]
            sample_y = y_in_range[indices]
            sample_t = t_in_range[indices]

            sorted_indices = np.argsort(sample_t)
            sorted_t = sample_t[sorted_indices]
            norm_t = (sorted_t - np.min(sorted_t)) / ((np.max(sorted_t) - np.min(sorted_t)) + 1e-15)
            sorted_x = sample_x[sorted_indices]
            norm_x = sorted_x / w
            sorted_y = sample_y[sorted_indices]
            norm_y = sorted_y / h

            sampled_events = np.stack((norm_x, norm_y, norm_t), axis=-1)
            all_sampled_events.append(sampled_events)

        if all_sampled_events:
            combined_events = np.array(all_sampled_events)
        else:
            combined_events = np.array([])

        max_values = e_frame.max(axis=(1, 2), keepdims=True)

        e_frame_norm = e_frame / (max_values.astype(np.float32) + + 1e-15)

        return torch.from_numpy(blur).float(), torch.from_numpy(sharp).float(), \
            torch.from_numpy(e_frame_norm).float(), torch.from_numpy(combined_events).float()

    def close(self):
        pass
