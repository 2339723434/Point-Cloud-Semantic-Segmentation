import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.pointops.functions import pointops
from modules.pointops2.functions import pointops as pointops2


from modules.polar_utils import xyz2sphere
from modules.recons_utils import cal_const, cal_normal, cal_center, check_nan_umb
from modules.pointops.functions import pointops

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from torch_scatter import scatter, scatter_softmax, scatter_sum, scatter_std, scatter_max


def sample_and_group(stride, nsample, center, normal, feature, offset, return_polar=False, num_sector=1, training=True):
    # sample
    if stride > 1:
        new_offset, sample_idx = [offset[0].item() // stride], offset[0].item() // stride
        for i in range(1, offset.shape[0]):
            sample_idx += (offset[i].item() - offset[i - 1].item()) // stride
            new_offset.append(sample_idx)
        new_offset = torch.cuda.IntTensor(new_offset)
        if num_sector > 1 and training:
            fps_idx = pointops.sectorized_fps(center, offset, new_offset, num_sector)  # [M]
        else:
            fps_idx = pointops.furthestsampling(center, offset, new_offset)  # [M]
        new_center = center[fps_idx.long(), :]  # [M, 3]
        new_normal = normal[fps_idx.long(), :]  # [M, 3]
    else:
        new_center = center
        new_normal = normal
        new_offset = offset

    # group
    
    N, M, D = center.shape[0], new_center.shape[0], normal.shape[1]
    group_idx, _ = pointops.knnquery(nsample, center, new_center, offset, new_offset)  # [M, nsample]
    group_center = center[group_idx.view(-1).long(), :].view(M, nsample, 3)  # [M, nsample, 3]
    group_normal = normal[group_idx.view(-1).long(), :].view(M, nsample, D)  # [M, nsample, 10]
    group_center_norm = group_center - new_center.unsqueeze(1)
    
    if return_polar:
        group_polar = xyz2sphere(group_center_norm)
        group_center_norm = torch.cat([group_center_norm, group_polar], dim=-1)

    if feature is not None:
        C = feature.shape[1]
        group_feature = feature[group_idx.view(-1).long(), :].view(M, nsample, C)
        new_feature = torch.cat([group_center_norm, group_normal, group_feature], dim=-1)   # [npoint, nsample, C+D]
    else:
        new_feature = torch.cat([group_center_norm, group_normal], dim=-1)
    #print(nsample, center.shape, new_center.shape) #32 torch.Size([213099, 3]) torch.Size([53274, 3])
    

    return new_center, new_normal, new_feature, new_offset    


def resort_points(points, idx):
    """
    Resort Set of points along G dim

    :param points: [N, G, 3]
    :param idx: [N, G]
    :return: [N, G, 3]
    """
    device = points.device
    N, G, _ = points.shape

    n_indices = torch.arange(N, dtype=torch.long).to(device).view([N, 1]).repeat([1, G])
    new_points = points[n_indices, idx, :]

    return new_points


def _fixed_rotate(xyz):
    # y-axis:45deg -> z-axis:45deg
    rot_mat = torch.FloatTensor([[0.5, -0.5, 0.7071], [0.7071, 0.7071, 0.], [-0.5, 0.5, 0.7071]]).to(xyz.device)
    return xyz @ rot_mat


def group_by_umbrella_v2(xyz, new_xyz, offset, new_offset, k=9):
    """
    Group a set of points into umbrella surfaces

    :param xyz: [N, 3]
    :param new_xyz: [N', 3]
    :param k: number of homogenous neighbors
    :return: [N', K-1, 3, 3]
    """
    group_idx, _ = pointops.knnquery(k, xyz, new_xyz, offset, new_offset)  # [M, K]
    group_xyz = xyz[group_idx.view(-1).long(), :].view(new_xyz.shape[0], k, 3)  # [M, K, 3]
    group_xyz_norm = group_xyz - new_xyz.unsqueeze(-2)
    group_phi = xyz2sphere(_fixed_rotate(group_xyz_norm))[..., 2]  # [M, K-1]
    sort_idx = group_phi.argsort(dim=-1)  # [M, K-1]

    # [M, K-1, 1, 3]
    sorted_group_xyz = resort_points(group_xyz_norm, sort_idx).unsqueeze(-2)
    sorted_group_xyz_roll = torch.roll(sorted_group_xyz, -1, dims=-3)
    group_centriod = torch.zeros_like(sorted_group_xyz)
    umbrella_group_xyz = torch.cat([group_centriod, sorted_group_xyz, sorted_group_xyz_roll], dim=-2)

    return umbrella_group_xyz


def group_by_umbrella(xyz, new_xyz, offset, new_offset, k=9):
    """
    Group a set of points into umbrella surfaces

    :param xyz: [N, 3]
    :param new_xyz: [N', 3]
    :param k: number of homogenous neighbors
    :return: [N', K-1, 3, 3]
    """
    group_idx, _ = pointops.knnquery(k, xyz, new_xyz, offset, new_offset)  # [M, K]
    group_xyz = xyz[group_idx.view(-1).long(), :].view(new_xyz.shape[0], k, 3)  # [M, K, 3]
    group_xyz_norm = group_xyz - new_xyz.unsqueeze(-2)
    group_phi = xyz2sphere(group_xyz_norm)[..., 2]  # [M, K-1]
    sort_idx = group_phi.argsort(dim=-1)  # [M, K-1]

    # [M, K-1, 1, 3]
    sorted_group_xyz = resort_points(group_xyz_norm, sort_idx).unsqueeze(-2)
    sorted_group_xyz_roll = torch.roll(sorted_group_xyz, -1, dims=-3)
    group_centriod = torch.zeros_like(sorted_group_xyz)
    umbrella_group_xyz = torch.cat([group_centriod, sorted_group_xyz, sorted_group_xyz_roll], dim=-2)

    return umbrella_group_xyz

    
def sort_factory(s_type):
    if s_type is None:
        return group_by_umbrella
    elif s_type == 'fix':
        return group_by_umbrella_v2
    else:
        raise Exception('No such sorting method')

    
# Feature Downsampling block
class SymmetricTransitionDownBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, nsample):
        super().__init__()
        self.stride, self.nsample = stride, nsample
        
        if stride != 1:
            self.channel_shrinker = nn.Sequential( # input.shape = [N*K, L]
                nn.Linear(3+in_planes, in_planes, bias=False),
                nn.BatchNorm1d(in_planes),
                nn.ReLU(inplace=True),
                nn.Linear(in_planes, 1))

        else:
            self.linear2 = nn.Sequential(
                nn.Linear(in_planes, out_planes, bias=False),
                nn.BatchNorm1d(out_planes),
                nn.ReLU(inplace=True))
        
    def forward(self, fea_knn):
        m, k, c = fea_knn.shape[0],fea_knn.shape[1],fea_knn.shape[2]
        print(m,k,c)
        fea_knn_flatten = rearrange(fea_knn, 'm k c -> (m k) c')
        fea_knn_flatten_shrink = self.channel_shrinker(fea_knn_flatten) # (m*nsample, 1)
        fea_knn_shrink = rearrange(fea_knn_flatten_shrink, '(m k) c -> m k c', m=m, k=k)
        fea_knn_prob_shrink = F.softmax(fea_knn_shrink, dim=1) #torch.Size([63127, 32, 1])
        
        fea_knn_weighted = fea_knn * fea_knn_prob_shrink
        fea = torch.sum(fea_knn_weighted, dim=1).contiguous()

        return fea


# Symmetric Upsampling Block
class SymmetricTransitionUpBlock(nn.Module):
    def __init__(self, in_planes, out_planes, nsample):
        super().__init__()
        self.nsample = nsample
        if out_planes is None:
            self.linear1 = nn.Sequential(
                nn.Linear(2*in_planes, in_planes, bias=False), 
                nn.BatchNorm1d(in_planes), 
                nn.ReLU(inplace=True))
            self.linear2 = nn.Sequential(
                nn.Linear(in_planes, in_planes), 
                nn.ReLU(inplace=True))            
        else:
            self.linear1 = nn.Sequential( # input.shape = [N, L]
                nn.Linear(out_planes, out_planes, bias=False), 
                nn.BatchNorm1d(out_planes),  
                nn.ReLU(inplace=True))
            self.linear2 = nn.Sequential( # input.shape = [N, L]
                nn.Linear(in_planes, out_planes, bias=False), 
                nn.BatchNorm1d(out_planes), 
                nn.ReLU(inplace=True))
            self.channel_shrinker = nn.Sequential( # input.shape = [N*K, L]
                nn.Linear(in_planes+3, in_planes, bias=False),
                nn.BatchNorm1d(in_planes),
                nn.ReLU(inplace=True),
                nn.Linear(in_planes, 1))
        
    def forward(self, pxo1, pxo2=None):
        if pxo2 is None:
            _, x, o = pxo1  # (n, 3), (n, c), (b)
            x_tmp = []
            for i in range(o.shape[0]):
                if i == 0:
                    s_i, e_i, cnt = 0, o[0], o[0]
                else:
                    s_i, e_i, cnt = o[i-1], o[i], o[i] - o[i-1]
                x_b = x[s_i:e_i, :]
                x_b = torch.cat((x_b, self.linear2(x_b.sum(0, True) / cnt).repeat(cnt, 1)), 1)
                x_tmp.append(x_b)
            x = torch.cat(x_tmp, 0)
            y = self.linear1(x) # this part is the same as TransitionUp module.
        else:
            # x1.shape: (n, c) encoder/fine-grain points
            # x2.shape: (m, c) decoder/coase points
            # p1.shape: (436, 3) # (n, 3) # p1 is the upsampled one.
            # p2.shape: (109, 3) # (m, 3)
            p1, x1, o1 = pxo1; p2, x2, o2 = pxo2 #torch.Size([1294, 3]) torch.Size([322, 3]) 
            knn_idx = pointops.knnquery(self.nsample, p1, p2, o1, o2)[0].long() #torch.Size([322, 16])
            # knn_idx.shape: (109, 16) # (m, nsample) 
            # knn_idx.max() == 435 == n # filled with x1's idx

            with torch.no_grad():
                knn_idx_flatten = rearrange(knn_idx, 'm k -> (m k)')
            #print(p1[knn_idx_flatten, :].shape) #torch.Size([5152, 3])
            p_r = p1[knn_idx_flatten, :].view(len(p2), self.nsample, 3) - p2.unsqueeze(1) #torch.Size([322, 16, 3])
            x2_knn = x2.view(len(p2), 1, -1).repeat(1, self.nsample, 1) #torch.Size([322, 16, 512])
            x2_knn = torch.cat([p_r, x2_knn], dim=-1) # (322, 16, 515) # (m, nsample, 3+c)

            with torch.no_grad():
                knn_idx_flatten = knn_idx_flatten.unsqueeze(-1) # (m*nsample, 1) torch.Size([5152, 1])
            x2_knn_flatten = rearrange(x2_knn, 'm k c -> (m k) c') # c = 3+out_planes torch.Size([5152, 515])
            x2_knn_flatten_shrink = self.channel_shrinker(x2_knn_flatten) # (m, nsample, 1) torch.Size([322, 16, 256])
            

            x2_knn_prob_flatten_shrink = scatter_softmax(
                x2_knn_flatten_shrink, knn_idx_flatten, dim=0) #torch.Size([5152, 1])

            x2_knn_prob_shrink = rearrange(
                x2_knn_prob_flatten_shrink, '(m k) 1 -> m k 1', k=self.nsample)
            up_x2_weighted = self.linear2(x2).unsqueeze(1) * x2_knn_prob_shrink #torch.Size([322, 16, 256]) = torch.Size([322, 1, 256]) * torch.Size([322, 16, 1])
            
            up_x2_weighted_flatten = rearrange(up_x2_weighted, 'm k c -> (m k) c') #torch.Size([[5152, 256]])
            
            up_x2 = scatter_sum(
                up_x2_weighted_flatten, knn_idx_flatten, dim=0, dim_size=len(p1))

            #print(x1.shape,up_x2.shape)
            y = self.linear1(x1) + up_x2
        return y

    

class SurfaceAbstractionCD(nn.Module):
    """
    Surface Abstraction Module w/ Channel De-differentiation

    """

    def __init__(self, stride, nsample, feat_channel, pos_channel, mlp, return_normal=True, return_polar=False,
                 num_sector=1):
        super(SurfaceAbstractionCD, self).__init__()
        self.stride = stride
        self.nsample = nsample
        self.return_normal = return_normal
        self.return_polar = return_polar
        self.num_sector = num_sector
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        self.pos_channel = pos_channel
        self.feat_channel = feat_channel
        
        #self.transdown = SymmetricTransitionDownBlock(in_planes=feat_channel, out_planes=feat_channel, stride=stride, nsample=nsample)

        self.mlp_l0 = nn.Conv1d(self.pos_channel, mlp[0], 1)
        self.mlp_f0 = nn.Conv1d(feat_channel, mlp[0], 1)
        self.bn_l0 = nn.BatchNorm1d(mlp[0])
        self.bn_f0 = nn.BatchNorm1d(mlp[0])

        last_channel = mlp[0]
        for out_channel in mlp[1:]:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, pos_nor_feat_off):
        center, normal, feature, offset = pos_nor_feat_off  # [N, 3], [N, 10], [N, C], [B]

        new_center, new_normal, new_feature, new_offset = sample_and_group(self.stride, self.nsample, center,
                                                                           normal, feature, offset,
                                                                           return_polar=self.return_polar,
                                                                           num_sector=self.num_sector,
                                                                           training=self.training)

        # new_center: sampled points position data, [M, 3]
        # new_normal: sampled normal feature data, [M, 3/10]
        # new_feature: sampled feature, [M, nsample, 3+3/10+C]
        #print(new_center.shape, new_normal.shape, new_feature.shape) #torch.Size([53274, 3]) torch.Size([53274, 10]) torch.Size([53274, 32, 19])
        
        #new_feature = self.transdown()
        #new_feature = self.transdown(new_feature)
        #print(new_feature.shape)

        new_feature = new_feature.transpose(1, 2).contiguous()  # [M, 3+C, nsample]
        #print(self.feat_channel,new_feature.shape)

        # init layer
        loc = self.bn_l0(self.mlp_l0(new_feature[:, :self.pos_channel]))
        feat = self.bn_f0(self.mlp_f0(new_feature[:, self.pos_channel:]))
        new_feature = loc + feat
        new_feature = F.relu(new_feature) #torch.Size([45062, 32, 32])


        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_feature = F.relu(bn(conv(new_feature)))
        new_feature = torch.max(new_feature, 2)[0] #torch.Size([45062, 64])

        return [new_center, new_normal, new_feature, new_offset]


class SurfaceFeaturePropagationCD(nn.Module):
    """
    Surface FP Module w/ Channel De-differentiation

    """

    def __init__(self, prev_channel, skip_channel, mlp):
        super(SurfaceFeaturePropagationCD, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        self.skip = skip_channel is not None
        self.nsample = 32
        self.transup = SymmetricTransitionUpBlock(prev_channel, skip_channel, 32)

        self.mlp_f0 = nn.Linear(prev_channel, mlp[0])
        self.norm_f0 = nn.BatchNorm1d(mlp[0])
        if skip_channel is not None:
            self.mlp_f1 = nn.Linear(skip_channel, mlp[0])
            self.norm_f1 = nn.BatchNorm1d(mlp[0])
            
            self.mlp_s0 = nn.Linear(skip_channel, mlp[0])
            self.norm_s0 = nn.BatchNorm1d(mlp[0])

        last_channel = mlp[0]
        for out_channel in mlp[1:]:
            self.mlp_convs.append(nn.Linear(last_channel, out_channel))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, pos_feat_off1, pos_feat_off2):
        xyz1, points1, offset1 = pos_feat_off1  # [N, 3], [N, C], [B]
        xyz2, points2, offset2 = pos_feat_off2  # [M, 3], [M, C], [B]
        #print('points1:',points1.shape,points2.shape)
        
        if points1 is not None:
            points2 = self.transup(pos_feat_off1,pos_feat_off2) #torch.Size([3971, 256])
            interpolated_points = self.norm_f1(self.mlp_f1(points2))
        else:        
            #print('xyz1:',xyz1.shape,xyz2.shape) #xyz1: torch.Size([3971, 3]) torch.Size([991, 3])
            # interpolation
            idx, dist = pointops.knnquery(3, xyz2, xyz1, offset2, offset1)  # [M, 3], [M, 3]
            dist_recip = 1.0 / (dist + 1e-8)  # [M, 3]
            norm = torch.sum(dist_recip, dim=1, keepdim=True)
            weight = dist_recip / norm  # [M, 3]

            points2 = self.norm_f0(self.mlp_f0(points2)) #torch.Size([991, 256])
        
            interpolated_points = torch.cuda.FloatTensor(xyz1.shape[0], points2.shape[1]).zero_()
            #print('interpolated_points1:',interpolated_points.shape)
            for i in range(3):
                interpolated_points += points2[idx[:, i].long(), :] * weight[:, i].unsqueeze(-1)
            #print('interpolated_points2:',interpolated_points.shape)

        # init layer
        if self.skip:
            skip = self.norm_s0(self.mlp_s0(points1))
            #print('skip:',skip.shape)
            new_points = F.relu(interpolated_points + skip)
        else:
            new_points = F.relu(interpolated_points)

        # mlp
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))

        return new_points
        
        
class UmbrellaSurfaceConstructor(nn.Module):
    """
    Umbrella Surface Representation Constructor

    """

    def __init__(self, k, in_channel, out_channel, random_inv=True, sort='fix'):
        super(UmbrellaSurfaceConstructor, self).__init__()
        self.k = k
        self.random_inv = random_inv

        self.mlps = nn.Sequential(
            nn.Conv1d(in_channel, out_channel, 1, bias=True),
            nn.BatchNorm1d(out_channel),
            nn.ReLU(True),
            nn.Conv1d(out_channel, out_channel, 1, bias=True),
        )
        self.sort_func = sort_factory(sort)

    def forward(self, center, offset):
        # umbrella surface reconstruction
        group_xyz = self.sort_func(center, center, offset, offset, k=self.k)  # [N, K-1, 3 (points), 3 (coord.)]

        # normal
        group_normal = cal_normal(group_xyz, offset, random_inv=self.random_inv, is_group=True)
        # coordinate
        group_center = cal_center(group_xyz)
        # polar
        group_polar = xyz2sphere(group_center) #torch.Size([213099, 9, 3])
        
        # surface position
        group_pos = cal_const(group_normal, group_center) ##torch.Size([213099, 9, 1])

        group_normal, group_center, group_pos = check_nan_umb(group_normal, group_center, group_pos)
        #print(group_normal.shape,group_center.shape,group_pos) #torch.Size([213099, 9, 3]) torch.Size([213099, 9, 3]) torch.Size([213099, 9, 1])
        
        new_feature = torch.cat([group_polar, group_normal, group_pos, group_center], dim=-1)  # P+N+SP+C: 10
        new_feature = new_feature.transpose(1, 2).contiguous()  # [N, C, G]

        # mapping
        new_feature = self.mlps(new_feature)

        # aggregation
        new_feature = torch.sum(new_feature, 2)

        return new_feature


class BilinearFeedForward(nn.Module):

    def __init__(self, in_planes1, in_planes2, out_planes):
        super().__init__()
        self.bilinear = nn.Bilinear(in_planes1, in_planes2, out_planes)

    def forward(self, x):
        x = x.contiguous()
        x = self.bilinear(x, x)
        return x


class LocalPointFusionLayer(nn.Module):
    expansion = 1

    def __init__(self, in_planes, out_planes, share_planes=8, nsample=16):
        super().__init__()
        self.mid_planes = mid_planes = out_planes // 1
        self.out_planes = out_planes
        self.share_planes = share_planes
        self.nsample = nsample

        self.channelMixMLPs01 = nn.Sequential( # input.shape = [N, K, C]
            nn.Linear(3+in_planes, nsample),
            nn.ReLU(inplace=True),
            BilinearFeedForward(nsample, nsample, nsample))
        
        self.linear_p = nn.Sequential( # input.shape = [N, K, C]
            nn.Linear(3, 3, bias=False),
            nn.Sequential(
                Rearrange('n k c -> n c k'),
                nn.BatchNorm1d(3),
                Rearrange('n c k -> n k c')),
            nn.ReLU(inplace=True), 
            nn.Linear(3, out_planes))
        self.shrink_p = nn.Sequential(
            Rearrange('n k (a b) -> n k a b', b=nsample),
            Reduce('n k a b -> n k b', 'sum', b=nsample))

        self.channelMixMLPs02 = nn.Sequential( # input.shape = [N, K, C]
            Rearrange('n k c -> n c k'),
            nn.Conv1d(nsample+nsample, mid_planes, kernel_size=1, bias=False),
            nn.BatchNorm1d(mid_planes), 
            nn.ReLU(inplace=True),
            nn.Conv1d(mid_planes, mid_planes//share_planes, kernel_size=1, bias=False),
            nn.BatchNorm1d(mid_planes//share_planes),
            nn.ReLU(inplace=True),
            nn.Conv1d(mid_planes//share_planes, out_planes//share_planes, kernel_size=1),
            Rearrange('n c k -> n k c'))

        self.channelMixMLPs03 = nn.Linear(in_planes, out_planes)
        
        self.softmax = nn.Softmax(dim=1)

    def forward(self, pxo) -> torch.Tensor:
        p, x, o = pxo  # (n, 3), (n, c), (b)

        x_knn, knn_idx = pointops2.queryandgroup(
            self.nsample, p, p, x, None, o, o, use_xyz=True, return_idx=True)  # (n, k, 3+c)
        p_r = x_knn[:, :, 0:3]

        energy = self.channelMixMLPs01(x_knn) # (n, k, k)
        
        #print(p_r)
        p_embed = self.linear_p(p_r) # (n, k, out_planes)
        p_embed_shrink = self.shrink_p(p_embed) # (n, k, k)

        energy = torch.cat([energy, p_embed_shrink], dim=-1)
        energy = self.channelMixMLPs02(energy) # (n, k, out_planes/share_planes)
        w = self.softmax(energy)

        x_v = self.channelMixMLPs03(x)  # (n, in_planes) -> (n, k)
        n = knn_idx.shape[0]; knn_idx_flatten = knn_idx.flatten()
        x_v  = x_v[knn_idx_flatten, :].view(n, self.nsample, -1)

        n, nsample, out_planes = x_v.shape
        x_knn = (x_v + p_embed).view(
            n, nsample, self.share_planes, out_planes//self.share_planes)
        x_knn = (x_knn * w.unsqueeze(2))
        x_knn = x_knn.reshape(n, nsample, out_planes)

        x = x_knn.sum(1)
        return (x, x_knn, knn_idx, p_r)
        
