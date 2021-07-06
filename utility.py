import struct
import numpy as np
import os
import matplotlib.pyplot as plt
from BirdsEyeView import BirdsEyeView
from BirdsEyeView import readKittiCalib

# read binary and convert to cylindsrical view
def bin2Pcd(binFileName):
    size_float = 4
    # x, y, z: geometric coordinates
    # t(theta), p(phi), r(rho): spherical coordinates
    # h: intensity
    xyztprh = []
    # open file as binary
    with open(binFileName, "rb") as f:
        byte = f.read(size_float * 4)
        while byte:
            x, y, z, intensity = struct.unpack("ffff", byte)
            r = np.sqrt(x * x + y * y + z * z)
            t = np.rad2deg(np.arccos(z / r))
            p = np.rad2deg(np.arctan2(y, x))
            xyztprh.append([x, y, z, t, p, r, intensity])
            byte = f.read(size_float * 4)
    return np.array(xyztprh)

# project point cloud data onto cylindrical view [p_len, t_len, channel]
# channel = (xyztprh for nearest point) + (xyztprh for farthest point) = 7 + 7 = 14
# (t_len, p_len): number of quantized values in (theta, phi) direction
# (p_min, p_max): (min, max) value of phi used for quantization
# if p_min, p_max is None, use the actual minimum and maximum values of phi
def pcd2Cyl(xyztprh, t_len=64, p_len=180, p_min=-45, p_max=45):
    if p_min is None:
        p_min = np.min(xyztprh[:, 4])
    else:
        xyztprh = xyztprh[p_min <= xyztprh[:, 4]]
    if p_max is None:
        p_max = np.max(xyztprh[:, 4])
    else:
        xyztprh = xyztprh[xyztprh[:, 4] < p_max]

    # use the actual minimum and maximum values of theta for t_min, t_max
    t_min = np.min(xyztprh[:, 3])
    t_max = np.max(xyztprh[:, 3])

    # p_space: quantization range in phi direction
    p_space = np.linspace(p_min, p_max, p_len+1)
    # p_quantized: quantized data in phi direction [p_len, number of data in each range, 7]
    p_quantized = [xyztprh[(p_space[i] <= xyztprh[:, 4]) & (xyztprh[:, 4] < p_space[i+1])] for i in range(p_len)]

    # t_space: quantization range in theta direction
    t_space = np.linspace(t_min, t_max, t_len+1)
    # tp_quantized: quantized data in theta and phi direction  [p_len, t_len, number of data in each range, 7]
    pt_quantized = [[td[(t_space[i] <= td[:, 3]) & (td[:, 3] < t_space[i+1])] for i in range(t_len)] for td in p_quantized]

    # ept: value for range with no data
    ept = np.array([-1] * 7 * 2)
    # ept = np.array([0] * 7 * 2)
    # ept = np.array(([-5, 0, 0] + [90, 0, -5] + [0]) * 2)
    # projected: cylindrical view [p_len, t_len, channel=14]
    projected = [[np.concatenate([td[np.argmin(td[:, 5])], td[np.argmax(td[:, 5])]]) if len(td) > 0 else ept for td in ptd] for ptd in pt_quantized]
    # return np.array(projected)

    # filtering only empty areas in cylindrical view to minimize empty areas
    kernel = np.array([[1, 4, 6, 4, 1], [4, 16, 24, 16, 4], [6, 24, 36, 24, 6], [4, 16, 24, 16, 4], [1, 4, 6, 4, 1]])
    r = kernel.shape[0] // 2
    filtered = np.array(projected)
    for i in range(filtered.shape[0]):
        for j in range(filtered.shape[1]):
            if np.all(projected[i][j] == ept):
                for wi in range(kernel.shape[0]):
                    for wj in range(kernel.shape[1]):
                        y, x = i + wi - r, j + wj - r
                        val, weight = [0] * 14, 0
                        pos = np.array([y, x])
                        if np.all((0 <= pos) & (pos < filtered.shape[:2])):
                            if np.any(projected[y][x] != ept):
                                val += projected[y][x] * kernel[wi][wj]
                                weight += kernel[wi][wj]

                if weight != 0:
                    filtered[i][j] = val / weight
                filtered[i][j][3] = filtered[i][j][10] = (t_space[j] + t_space[j+1]) / 2
                filtered[i][j][4] = filtered[i][j][11] = (p_space[i] + p_space[i+1]) / 2

    return filtered

# L: LiDAR coordinates [3]
# K: calibulation matrix [3, 4]
# output: Camera coordinates [2]
def lidar2Cam(L, K):
    xyz = K.dot(np.append(L, 1))
    return xyz[:2] / xyz[2]

# get mat values at pos if pos is in the matrix space
# otherwise return 0
def getMatVal(mat, pos):
    pos = np.round(pos).astype('int')
    if np.all((0 <= pos) & (pos < mat.shape[:2])):
        return mat[tuple(pos)]
    else:
        return 0

# set val at pos in mat if pos is in the matrix space
# otherwise do nothing
def setMatVal(mat, pos, val):
    pos = np.round(pos).astype('int')
    if np.all((0 <= pos) & (pos < mat.shape[:2])):
        mat[tuple(pos)] = val

def gt2Cyl(num, path='data_road', img_type='um', pcd=None):
    basePath = os.path.join(path, 'training')

    # load cylindrical view if not provided by pcd
    L = pcd
    if pcd is None:
        binFileName = os.path.join(basePath, 'velodyne/%s_%06d.bin' % (img_type, num))
        L = pcd2Cyl(bin2Pcd(binFileName))

    # load ground truth image
    gtFileName = os.path.join(basePath, 'gt_image_2/%s_road_%06d.png' % (img_type, num))
    B = plt.imread(gtFileName)[:, :, 2]

    # load calibration data
    calibFileName = os.path.join(basePath, 'calib/%s_%06d.txt' % (img_type, num))
    calib = readKittiCalib(calibFileName)
    P2 = np.array(calib['P2']).reshape(3,4)
    R0 = np.array(calib['R0_rect']).reshape(3,3)
    T = np.array(calib['Tr_velo_to_cam']).reshape(3,4)
    # K: calibration matrix
    K = P2.dot(np.vstack([R0.dot(T), [0, 0, 0, 1]]))

    # if both projected pcds of nearest and farthest on ground truth image have label 1,
    # set 1 on return view, otherwise 0
    G = np.array([
            [getMatVal(B, lidar2Cam(L[i][j][0:3], K)[::-1]) * getMatVal(B, lidar2Cam(L[i][j][7:10], K)[::-1])
            for j in range(L.shape[1])]
            for i in range(L.shape[0])])

    return G

def pcdSeg2Cam(num, coords, seg, path='data_road', img_type='um', color=[1, 0, 1]):
    basePath = os.path.join(path, 'training')

    # load original image
    imgFileName = os.path.join(basePath, 'image_2/%s_%06d.png' % (img_type, num))
    img = plt.imread(imgFileName)

    # load calibration data
    calibFileName = os.path.join(basePath, 'calib/%s_%06d.txt' % (img_type, num))
    calib = readKittiCalib(calibFileName)
    P2 = np.array(calib['P2']).reshape(3,4)
    R0 = np.array(calib['R0_rect']).reshape(3,3)
    T = np.array(calib['Tr_velo_to_cam']).reshape(3,4)
    # K: calibration matrix
    K = P2.dot(np.vstack([R0.dot(T), [0, 0, 0, 1]]))

    # set color to the 3 x 3 pixels around the pixel in img
    # whose value in predicted result equals to 1
    for i in range(seg.shape[0]):
        for j in range(seg.shape[1]):
            if seg[i][j] == 1:
                p1 = lidar2Cam(coords[i][j][0:3], K)[::-1]
                p2 = lidar2Cam(coords[i][j][7:10], K)[::-1]
                for ii in range(3):
                    for jj in range(3):
                        offset = [ii-1, jj-1]
                        setMatVal(img, p1+offset, color)
                        setMatVal(img, p2+offset, color)

    return img

def cam2BEV(num, img, path='data_road', img_type='um'):
    '''
    Main method of cam2BEV
    :param dataFiles: the files you want to transform to BirdsEyeView, e.g., /home/elvis/kitti_road/data/*.png
    :param pathToCalib: containing calib data as txt-files, e.g., /home/elvis/kitti_road/calib/
    :param outputPath: where the BirdsEyeView data will be saved, e.g., /home/elvis/kitti_road/data_bev
    :param calib_end: file extension of calib-files (OPTIONAL)
    '''

    calibFileName = os.path.join(path, 'training/calib/%s_%06d.txt' % (img_type, num))

    # BEV class
    bev = BirdsEyeView()
    bev.setup(calibFileName)

    # Compute Birds Eye View
    data_bev = bev.compute(img)

    return data_bev


def gt2CamSeg(num, path='data_road', img_type='um', color=[1, 0, 1]):
    basePath = os.path.join(path, 'training')

    # load original image
    imgFileName = os.path.join(basePath, 'image_2/%s_%06d.png' % (img_type, num))
    img = plt.imread(imgFileName)

    # load ground truth image
    gtFileName = os.path.join(basePath, 'gt_image_2/%s_road_%06d.png' % (img_type, num))
    gt = plt.imread(gtFileName)[:, :, 2]

    # set color to the pixel in img whose value in ground truth equals to 1
    for i in range(gt.shape[0]):
        for j in range(gt.shape[1]):
            if gt[i][j] == 1:
                img[i][j] = color

    return img


def visualizeCyl(mat, file=None):
    fig, axes = plt.subplots(nrows=2, ncols=7, sharex=False, figsize=(16, 8))
    t = [r'$x$', r'$y$', r'$z$', r'$\theta$', r'$\phi$', r'$\rho$', 'intensity']
    for i in range(14):
        axes[i // 7, i % 7].set_title(t[i % 7])
        axes[i // 7, i % 7].imshow(mat[:, :, i], cmap='gray')
    fig.tight_layout()
    if file is not None:
        plt.savefig(file)

def visualizeOutput(inputs, targets, outputs, num, img_type, isMSE=False, file=None):
    fig, axes = plt.subplots(nrows=3, ncols=2, sharex=False, figsize=(16, 16))

    axes[0, 0].set_title('Ground Truth for Point Cloud Data')
    axes[0, 0].imshow(targets[0][0].cpu().numpy(), cmap='gray')

    axes[0, 1].set_title('Neural Net Output')
    if isMSE:
        out_img = np.where(outputs[0][0].cpu().numpy() > 0.5, 1, 0)
    else:
        out_img = np.argmax(outputs[0].cpu().numpy(), axis=0)
    axes[0, 1].imshow(out_img, cmap='gray')

    axes[1, 0].set_title('Segmentation Prediction of Camera View')
    seg_img = pcdSeg2Cam(num, inputs.permute(0, 2, 3, 1)[0].cpu().numpy(), out_img, img_type=img_type)
    axes[1, 0].imshow(seg_img)

    axes[1, 1].set_title('Segmentation Prediction of Birds Eye View')
    bev_img = cam2BEV(num, seg_img, img_type=img_type)
    axes[1, 1].imshow(bev_img)

    axes[2, 0].set_title('Segmentation Ground Truth of Camera View')
    gt_seg_img = gt2CamSeg(num, img_type=img_type)
    axes[2, 0].imshow(gt_seg_img)

    axes[2, 1].set_title('Segmentation Ground Truth of Birds Eye View')
    gt_bev_img = cam2BEV(num, gt_seg_img, img_type=img_type)
    axes[2, 1].imshow(gt_bev_img)

    fig.tight_layout()
    if file is not None:
        plt.savefig(file)