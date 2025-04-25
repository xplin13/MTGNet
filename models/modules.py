import torch
import torch.nn as nn
import torch.nn.functional as F

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def furthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        distance = torch.min(distance, dist)
        farthest = torch.max(distance, -1)[1]
    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim=-1, largest=False, sorted=False)
    return group_idx


class LocalGrouper(nn.Module):
    def __init__(self, channel, groups, kneighbors, use_xyz=True, normalize="center", **kwargs):
        """
        Give xyz[b,p,3] and fea[b,p,d], return new_xyz[b,g,3] and new_fea[b,g,k,d]
        :param groups: groups number
        :param kneighbors: k-nerighbors
        :param kwargs: others
        """
        super(LocalGrouper, self).__init__()
        self.groups = groups
        self.kneighbors = kneighbors
        self.use_xyz = use_xyz
        if normalize is not None:
            self.normalize = normalize.lower()
        else:
            self.normalize = None
        if self.normalize not in ["center", "anchor"]:
            print(f"Unrecognized normalize parameter (self.normalize), set to None. Should be one of [center, anchor].")
            self.normalize = None
        if self.normalize is not None:
            add_channel=3 if self.use_xyz else 0
            self.affine_alpha = nn.Parameter(torch.ones([1,1,1,channel + add_channel]))
            self.affine_beta = nn.Parameter(torch.zeros([1, 1, 1, channel + add_channel]))

    def forward(self, xyz, points):
        B, N, C = xyz.shape
        S = self.groups
        xyz = xyz.contiguous()  # xyz [btach, points, xyz]
        fps_idx = furthest_point_sample(xyz, self.groups).long()  # [B, npoint]
        # fps_idx = farthest_point_sample(xyz, self.groups).long()
        fps_idx,indices = torch.sort(fps_idx, dim=1)
        new_xyz = index_points(xyz, fps_idx)  # [B, npoint, 3]
        new_points = index_points(points, fps_idx)  # [B, npoint, d]

        idx = knn_point(self.kneighbors, xyz, new_xyz)
        grouped_xyz = index_points(xyz, idx)  # [B, npoint, k, 3]
        grouped_points = index_points(points, idx)  # [B, npoint, k, d]
        min_val_group = torch.min(grouped_points,dim=-2,keepdim=True)[0]
        if self.use_xyz:
            grouped_points = torch.cat([grouped_points, grouped_xyz],dim=-1)  # [B, npoint, k, d+3]
        if self.normalize is not None:
            if self.normalize =="center":
                mean = torch.mean(grouped_points, dim=2, keepdim=True)
            if self.normalize =="anchor":
                mean = torch.cat([new_points, new_xyz],dim=-1) if self.use_xyz else new_points
                mean = mean.unsqueeze(dim=-2)  # [B, npoint, 1, d+3]
            std = torch.std((grouped_points-mean).reshape(B,-1),dim=-1,keepdim=True).unsqueeze(dim=-1).unsqueeze(dim=-1)
            grouped_points = (grouped_points-mean)/(std + 1e-5)
            grouped_points = self.affine_alpha*grouped_points + self.affine_beta
        new_points = torch.cat([grouped_points, new_points.view(B, S, 1, -1).repeat(1, 1, self.kneighbors, 1)], dim=-1)
        return new_xyz, new_points



class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, lstm_output):
        attn_weights = self.linear(lstm_output).squeeze(-1)
        attn_probs = torch.softmax(attn_weights, dim=1)
        return attn_probs


class AMM_w_AFDM(nn.Module):
    def __init__(self, in_channels, max_dilation=5):
        super(AMM_w_AFDM, self).__init__()
        self.max_dilation = max_dilation

        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=1, stride=1, padding=0),
            nn.GroupNorm(num_groups=4, num_channels=16),
            nn.ReLU(inplace=True)
        )

        self.gap = nn.AdaptiveAvgPool2d((16, 16))

        self.dx = torch.arange(-self.max_dilation, self.max_dilation + 1)
        self.dy = torch.arange(-self.max_dilation, self.max_dilation + 1)
        self.dx, self.dy = torch.meshgrid(self.dx, self.dy, indexing='ij')
        self.distances = torch.sqrt(self.dx ** 2 + self.dy ** 2)

        self.mlp = nn.Sequential(
            nn.Linear(16*16*16, 128, bias=False),
            nn.ReLU(),
            nn.Linear(128, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x, xyz, feature):
        x_ori = x
        feature_ori = feature.clone()

        height = feature.shape[-2]
        width = feature.shape[-1]

        sum_feature = torch.sum(feature, dim=1).unsqueeze(1)
        sum_featuremin_val = sum_feature.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]
        sum_featuremax_val = sum_feature.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]

        sum_feature = (sum_feature - sum_featuremin_val) / (sum_featuremax_val - sum_featuremin_val + 1e-5)

        reduced_feature = self.conv1x1(feature_ori)

        reduced_feature = sum_feature + reduced_feature
        reduced_feature = self.gap(reduced_feature)
        reduced_feature_flattened = reduced_feature.view(reduced_feature.size(0), -1)

        dilation_log = self.mlp(reduced_feature_flattened).squeeze()

        x_coords_ori = (xyz[:, :, 0] * (width - 1)).long()
        y_coords_ori = (xyz[:, :, 1] * (height - 1)).long()

        linear_indices_ori = x_coords_ori + y_coords_ori * width

        feature = feature.view(feature.size(0), feature.size(1), -1)

        batch_size, num_points, num_features = x.shape
        x = x.permute(0, 2, 1)
        linear_indices_ori = linear_indices_ori.expand(num_features, -1, -1).permute(1, 0, 2)

        feature.scatter_add_(2, linear_indices_ori, x)

        feature = feature.view(batch_size, num_features, height, width)
        ori_feature = torch.zeros_like(feature)

        x_coords = (xyz[:, :, 0] * (width - 1)).long()
        y_coords = (xyz[:, :, 1] * (height - 1)).long()

        dilation = self.max_dilation * dilation_log.unsqueeze(-1).unsqueeze(-1)
        distances = self.distances.cuda().unsqueeze(0)
        weights = torch.exp(-distances / dilation)
        weights = weights / weights.sum(dim=[-1, -2], keepdim=True)

        x_coords = x_coords.unsqueeze(-1) + self.dx.flatten().cuda()
        y_coords = y_coords.unsqueeze(-1) + self.dy.flatten().cuda()
        x_coords = x_coords.clamp(0, width - 1)
        y_coords = y_coords.clamp(0, height - 1)

        linear_indices = x_coords + y_coords * width

        ori_feature = ori_feature.view(ori_feature.size(0), ori_feature.size(1), -1)

        batch_size, num_points, num_features = x_ori.shape

        batch_results = []

        for b in range(x_ori.shape[0]):
            current_x_ori = x_ori[b]
            current_x_ori = current_x_ori.permute(1, 0).unsqueeze(-1)
            expanded_x_ori = current_x_ori.expand(-1, -1, weights[b].numel())
            current_weights = weights[b].flatten()
            weighted_x_ori = expanded_x_ori * current_weights
            batch_results.append(weighted_x_ori)
        x_ori = torch.stack(batch_results, dim=0)

        linear_indices = linear_indices.expand(num_features, -1, -1, -1).permute(1, 0, 2, 3)

        ori_feature.scatter_add_(2, linear_indices.reshape(batch_size, num_features, -1),
                                 x_ori.reshape(batch_size, num_features, -1))

        ori_feature = ori_feature.view(batch_size, num_features, height, width)

        output = feature + ori_feature

        return output