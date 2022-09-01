import torch
import pickle
import numpy as np
from scipy.spatial import ConvexHull
from numpy import *
#  load the evaluation results
result_pkl0 = open(r"/home/yml/second.pytorch/second/result_evaluate/model1/eval_results/step_296960/result.pkl",'rb')   # 1, 3, 4, 5, 7
result_pkl1 = open(r"/home/yml/second.pytorch/second/result_evaluate/model3/eval_results/step_408320/result.pkl",'rb')
result_pkl2 = open(r"/home/yml/second.pytorch/second/result_evaluate/model4/eval_results/step_334080/result.pkl",'rb')
result_pkl3 = open(r"/home/yml/second.pytorch/second/result_evaluate/model5/eval_results/step_296960/result.pkl",'rb')
result_pkl4 = open(r"/home/yml/second.pytorch/second/result_evaluate/model7/eval_results/step_296960/result.pkl",'rb')
result0 = pickle.load(result_pkl0)
result1 = pickle.load(result_pkl1)
result2 = pickle.load(result_pkl2)
result3 = pickle.load(result_pkl3)
result4 = pickle.load(result_pkl4)

results = [result0,result1,result2,result3,result4]

########### loop to data
m = len(results)  # m is Net_numbers
det_bbox =[0] * m
det_score =[0] * m
det_dimension =[0] * m
det_location =[0] * m
det_rotation =[0] * m
bbox = [0] * m
for k in range(m):
    # det_bbox[k] = [0] * 3769
    # det_score[k] = [0] * 3769
    # det_dimension[k] = [0] * 3769
    # det_location[k] = [0] * 3769
    # det_rotation[k] = [0] * 3769
    # bbox[k] = [0] * 3769

    det_bbox[k] = [0] * 5  # 5代表什么？就是我找前5帧图像就行，没有全部遍历
    det_score[k] = [0] * 5
    det_dimension[k] = [0] * 5
    det_location[k] = [0] * 5
    det_rotation[k] = [0] * 5
    bbox[k] = [0] * 5

    for i in range(5):
        det_bbox[k][i] = results[k][i].get('bbox')  # 4
        det_score[k][i] = results[k][i].get('score')  # 1
        det_dimension[k][i] = results[k][i].get('dimensions')  # 3
        det_location[k][i] = results[k][i].get('location')  # 3
        det_rotation[k][i] = results[k][i].get('rotation_y')  # 1
        # the numbers to > 0.5
        n = 0
        object =[]
        for j in range(len(det_score[k][i])):
            if det_score[k][i][j] > 0.05:
                object.append(j)
                n = n+1
        # print(object)
        # bbox[k][i] = [0] * len(det_score[k][i])
        bbox[k][i] = [0] * n

        # for j in range(len(det_score[k][i])):
        # for j in range(n):
        for j in object:
            bbox[k][i][j] = torch.from_numpy(np.hstack(
                [det_bbox[k][i][j], det_score[k][i][j], det_dimension[k][i][j], det_location[k][i][j], det_rotation[k][i][j]]))

Bboxes = [bbox[0][3],bbox[1][3],bbox[2][3],bbox[3][3],bbox[4][3]]   #  第i和网络的第j帧   5  nets
# Bboxes = [bbox[0][3],bbox[1][3],bbox[2][3],bbox[4][3]]   #  第i和网络的第j帧   4  nets
# Bboxes = [bbox[0][3],bbox[1][3],bbox[4][3]]   #  第i和网络的第j帧   3  nets
# Bboxes = [bbox[0][3],bbox[1][3]]   #  第i和网络的第j帧   2  nets
# Bboxes = [bbox[0][3]]   #  第i和网络的第j帧   1  nets

print(bbox[0][3][0])  #第1个网络（编号为0）第4帧的第一个物体的信息
# print(Bboxes[0][0])




#  3D_IOU calculation
def polygon_clip(subjectPolygon, clipPolygon):
    """ Clip a polygon with another polygon.
    Ref: https://rosettacode.org/wiki/Sutherland-Hodgman_polygon_clipping#Python
    Args:
      subjectPolygon: a list of (x,y) 2d points, any polygon.
      clipPolygon: a list of (x,y) 2d points, has to be *convex*
    Note:
      **points have to be counter-clockwise ordered**
    Return:
      a list of (x,y) vertex point for the intersection polygon.
    """

    def inside(p):
        return (cp2[0] - cp1[0]) * (p[1] - cp1[1]) > (cp2[1] - cp1[1]) * (p[0] - cp1[0])

    def computeIntersection():
        dc = [cp1[0] - cp2[0], cp1[1] - cp2[1]]
        dp = [s[0] - e[0], s[1] - e[1]]
        n1 = cp1[0] * cp2[1] - cp1[1] * cp2[0]
        n2 = s[0] * e[1] - s[1] * e[0]
        n3 = 1.0 / (dc[0] * dp[1] - dc[1] * dp[0])
        return [(n1 * dp[0] - n2 * dc[0]) * n3, (n1 * dp[1] - n2 * dc[1]) * n3]

    outputList = subjectPolygon
    cp1 = clipPolygon[-1]

    for clipVertex in clipPolygon:
        cp2 = clipVertex
        inputList = outputList
        outputList = []
        s = inputList[-1]

        for subjectVertex in inputList:
            e = subjectVertex
            if inside(e):
                if not inside(s):
                    outputList.append(computeIntersection())
                outputList.append(e)
            elif inside(s):
                outputList.append(computeIntersection())
            s = e
        cp1 = cp2
        if len(outputList) == 0:
            return None
    return outputList


def poly_area(x, y):
    """ Ref: http://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates """
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def convex_hull_intersection(p1, p2):
    """ Compute area of two convex hull's intersection area.
        p1,p2 are a list of (x,y) tuples of hull vertices.
        return a list of (x,y) for the intersection and its volume
    """
    inter_p = polygon_clip(p1, p2)
    if inter_p is not None:
        hull_inter = ConvexHull(inter_p)
        return inter_p, hull_inter.volume
    else:
        return None, 0.0


def box3d_vol(corners):
    """
    corners: (8,3) no assumption on axis direction
    """
    # from tensor to numpy
    corners = corners.numpy()
    a = np.sqrt(np.sum((corners[0, :] - corners[1, :]) ** 2))
    b = np.sqrt(np.sum((corners[1, :] - corners[2, :]) ** 2))
    c = np.sqrt(np.sum((corners[0, :] - corners[4, :]) ** 2))
    return a * b * c


def is_clockwise(p):
    x = p[:, 0]
    y = p[:, 1]
    return np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)) > 0


def box3d_iou(corners1, corners2):
    """ Compute 3D bounding box IoU.
    Input:
        corners1: numpy array (8,3), assume up direction is negative Y
        corners2: numpy array (8,3), assume up direction is negative Y
    Output:
        iou: 3D bounding box IoU
        iou_2d: bird's eye view 2D bounding box IoU
    todo (kent): add more description on corner points' orders.
    """
    # corner points are in counter clockwise order
    rect1 = [(corners1[i, 0], corners1[i, 2]) for i in range(3, -1, -1)]
    rect2 = [(corners2[i, 0], corners2[i, 2]) for i in range(3, -1, -1)]

    area1 = poly_area(np.array(rect1)[:, 0], np.array(rect1)[:, 1])
    area2 = poly_area(np.array(rect2)[:, 0], np.array(rect2)[:, 1])

    inter, inter_area = convex_hull_intersection(rect1, rect2)
    iou_2d = inter_area / (area1 + area2 - inter_area)
    ymax = min(corners1[0, 1], corners2[0, 1])
    ymin = max(corners1[4, 1], corners2[4, 1])

    inter_vol = inter_area * max(0.0, ymax - ymin)

    vol1 = box3d_vol(corners1)
    vol2 = box3d_vol(corners2)
    iou = inter_vol / (vol1 + vol2 - inter_vol)
    # return iou, iou_2d
    return iou


# ----------------------------------
# Helper functions for evaluation
# ----------------------------------

def get_3d_box(box_size, heading_angle, center):
    """ Calculate 3D bounding box corners from its parameterization.
    Input:
        box_size: tuple of (length,wide,height)
        heading_angle: rad scalar, clockwise from pos x axis
        center: tuple of (x,y,z)
    Output:
        corners_3d: numpy array of shape (8,3) for 3D box cornders
    """

    def roty(t):
        c = np.cos(t)
        s = np.sin(t)
        return np.array([[c, 0, s],
                         [0, 1, 0],
                         [-s, 0, c]])

    R = roty(heading_angle)
    l, w, h = box_size
    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    y_corners = [h / 2, h / 2, h / 2, h / 2, -h / 2, -h / 2, -h / 2, -h / 2]
    z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
    corners_3d = torch.from_numpy(np.dot(R, np.vstack([x_corners, y_corners, z_corners])))

    corners_3d[0, :] = corners_3d[0, :] + center[0]
    corners_3d[1, :] = corners_3d[1, :] + center[1]
    corners_3d[2, :] = corners_3d[2, :] + center[2]
    corners_3d = np.transpose(corners_3d)
    return corners_3d

# if __name__ == '__main__':
#     print('------------------')
#     # get_3d_box(box_size, heading_angle, center)
#     corners_3d_ground = get_3d_box((1.497255, 1.644981, 3.628938), -1.531692, (2.882992, 1.698800, 20.785644))
#     corners_3d_predict = get_3d_box((1.458242, 1.604773, 3.707947), -1.549553, (2.756923, 1.661275, 20.943280))
#     (IOU_3d, IOU_2d) = box3d_iou(corners_3d_predict, corners_3d_ground)
#     print(IOU_3d, IOU_2d)  # 3d IoU/ 2d IoU of BEV(bird eye's view)



# 求解聚类结果
def bbox_in_cluster(cluster, bbox_tp, iou_thre=0.55):  # cluster已经分类的，bbox_tp未分类的
    # if a bbox should be in a cluster, return True
    # else return False
    # box = bbox_tp[:4] #数组在后面未参与比较的

    #函数接口，参数输入
    box = get_3d_box((bbox_tp[5], bbox_tp[7], bbox_tp[6]), bbox_tp[11], (bbox_tp[8], bbox_tp[9], bbox_tp[10]))
    for bbox_tuple in cluster:
        # checkbox = bbox_tuple[:4]
        checkbox = get_3d_box((bbox_tuple[5], bbox_tuple[7], bbox_tuple[6]), bbox_tuple[11], (bbox_tuple[8], bbox_tuple[9], bbox_tuple[10]))

        #检查两个框架的交并比
        # if box3d_iou(box, checkbox) >0.55:
        #     print(box3d_iou(box, checkbox))

        if box3d_iou(box,checkbox) > iou_thre:

            # if bbox_iou(box, checkbox) > iou_thre:
            return True
    return False

def BSAS_excl(Bboxes, iou_thre=0.55):
    # use BSAS_intra-sample exlusivity method to cluster bboxes detected in different ensemble models
    # Bboxes includes bboxes detected from different ensemble models
    # bboxes should be a list of bbox
    # a bbox is in format:
    # [box(4), scores，dimension，location，rotation]
    clusters = []
    for bboxes in Bboxes:
        # clusters from the first ensemble model
        c = len(clusters)
        if not c:
            for bbox_tp in bboxes:
                clusters.append([bbox_tp])  #bbox_tp从数据中拿出来要比较的
            continue

        cluster_flag = torch.zeros(c)

        for bbox_tp in bboxes:  #从第二组开始查询
            # check if has been clustered
            deal_flag = False
            i = int(-1)

            # clustering
            for cluster in clusters:
                i += 1
                if bbox_in_cluster(cluster, bbox_tp, iou_thre) and not cluster_flag[i]:
                    cluster_flag[i] = 1
                    cluster.append(bbox_tp)
                    deal_flag = True
                    break

            # create a new cluster for the new object
            if not deal_flag:
                clusters.append([bbox_tp])
                cluster_flag = torch.cat((cluster_flag, torch.zeros(1)), 0)  #  维数不增加
    return clusters

def stats_cluster(cluster):
    # calculate mean and variance of the cluster
    mean = cluster[0]  #分类中第一个元素
    var = torch.zeros(len(mean)+1)
    n = len(cluster) #某一个障碍物有几个网络识别到
    var[5] = n

    if n > 1:
        # classes_num = torch.zeros(len(mean[6:]))    # x1 y1 x2 y2 conf cls *scores   障碍物识别种类数量，scores数量
        classes_num = torch.zeros(1)    # x1 y1 x2 y2 conf cls *scores   障碍物识别种类数量，scores数量

        bboxes = []
        scores = []
        dimensions = []
        locations = []
        rotations = []

        for bbox_tp in cluster:
            # bbox_tp[:4] = xywh2xyxy(bbox_tp[:4])
            bboxes.append(bbox_tp[:4])
            # scores.append(bbox_tp[6:])
            scores.append(bbox_tp[4])
            dimensions.append(bbox_tp[5:8])
            locations.append(bbox_tp[8:11])
            rotations.append(bbox_tp[11])

            # classes_num[int(bbox_tp[5])] += 1  #  5 class 类别

        bbox_matrix = torch.stack(bboxes, 0)
        # print(bbox_matrix)
        score_matrix = torch.stack(scores, 0)
        dimension_matrix = torch.stack(dimensions, 0)
        location_matrix = torch.stack(locations, 0)
        rotation_matrix = torch.stack(rotations, 0)

        mean[:4] = torch.mean(bbox_matrix, dim=0)
        # mean[6:] = torch.mean(score_matrix, dim=0)
        mean[4] = torch.mean(score_matrix, dim=0)
        mean[5:8] = torch.mean(dimension_matrix, dim=0)
        mean[8:11] = torch.mean(location_matrix, dim=0)
        mean[11] = torch.mean(rotation_matrix, dim=0)

        # mean[5] = torch.argmax(classes_num, 0)   # 类别对应的数量最多，就属于这个类别
        # mean[4] = mean[6+int(mean[5])]   # 新的conf
        var[:4] = torch.var(bbox_matrix, dim=0)
        var[4] = torch.var(score_matrix, dim=0)
        # var[6:] = torch.std(score_matrix, dim=0)
        var[6:9] = torch.var(dimension_matrix, dim=0)
        var[9:12] = torch.var(location_matrix, dim=0)
        var[12] = torch.var(rotation_matrix, dim=0)

        # var[4] = var[6+int(mean[5])]
    else:
        # mean[:4] = xywh2xyxy(mean[:4])
        mean[:4] = mean[:4]

    xyxy = mean[:4]
    # pe = calculate_pe(mean[6:], normalize=False, normalize_method=None)
    pe = calculate_pe(mean[4], normalize=False, normalize_method=None)

    cpe = pe * (1 + (5 - n) * 0.2)
    # print(cpe)
    # print(n)
    # print(cpe)
    # cpe = pe
    return mean, var, xyxy, cpe

# def xyxy2xywh(x):
#     # Convert the box in mean to xyxy style
#     y = x.clone()
#     # y = [0] * 4
#     y[0] = x[0] / 2 + x[2] / 2
#     y[1] = x[1] / 2 + x[3] / 2
#     y[2] = x[2] - x[0]
#     y[3] = x[3] - x[1]
#     return y
#
#
# def xywh2xyxy(x):
#     # Convert the box in mean to xyxy style
#     y = x.clone()
#     y[0] = x[0] - x[2] / 2  # top left x
#     y[1] = x[1] - x[3] / 2  # top left y
#     y[2] = x[0] + x[2] / 2  # bottom right x
#     y[3] = x[1] + x[3] / 2  # bottom right y
#     return y

def plogp(p):
    # assert(p>=0, "wrong p given in plogp")
    if p < 1e-9:
        return 0
    else:
        return p * torch.log2(p)


def calculate_pe(mean, normalize=True, normalize_method=None):
    # calculate predict entropy
    # if normalize, the raw class vector will be normalized
    pe = 0
    if normalize:
        if normalize_method == 'average':
            mean = mean / torch.sum(mean)
        if normalize_method == 'softmax':
            mean = torch.exp(mean)
            mean = mean / torch.sum(mean)
        for value in mean:
            pe += - plogp(value)
    else:
        # for value in mean:
        #     # pe += - plogp(value) - plogp(1 - value)
        #     pe = - (plogp(value) + plogp(1 - value))
        #     print(pe)

        pe = - (plogp(mean) + plogp(1 - mean))
     #   print(pe)

    return pe

if len(Bboxes):
    clusters = BSAS_excl(Bboxes, iou_thre=0.55)
    for cluster in clusters:
        mean, var, xyxy, pe = stats_cluster(cluster)

        # print(len(clusters)) #识别总障碍物的数量
        #
        # # print(xyxy)
        if var[5] > 3:
            print(var[5])  # 每一障碍物识别的网络数量
        # print(pe)
        # mean[0:4] xyxy    mean[4] score ,mean[5:8] dimensioms  mean[8:11] location mean[11] rotation

        # with open('/home/yml/second.pytorch/second/result_evaluate/result_8.txt', 'a') as f:
        #     f.write('xyxy: ' + ('%g ' * len(mean[:4])).rstrip() % tuple(mean[:4]) + ' ; ')
            # f.write('scores: '+ '%g' % mean[4] + ' , ')
            # f.write('%g' % mean[4] + ' , ')
        #
        #     # f.write('dimensions: ' + ('%g ' * len(mean[5:8])).rstrip() % tuple(mean[5:8]) + ' ; ')
        #     # f.write('location: ' + ('%g ' * len(mean[8:11])).rstrip() % tuple(mean[8:11]) + ' ; ')
        #     # f.write('rotations: '+ '%g' % mean[11] + ' ; ')
        #
        #     # f.write('var_box: ' + ('%g ' * len(var[:4])).rstrip() % tuple(var[:4]) + ' ; ')
        #     # f.write('var_scores: ' + '%g' % var[4] + ' ; ')
        #     # f.write('detected: ' + '%g' % var[5] + ' ; ')
        #     # f.write('var_dimensions: ' + ('%g ' * len(var[6:9])).rstrip() % tuple(var[6:9]) + ' ; ')
        #     # f.write('var_locations: ' + ('%g ' * len(var[9:12])).rstrip() % tuple(var[9:12]) + ' ; ')
        #     # f.write('var_rotations: ' + '%g' % var[12] + ' ; ')
        #     f.write('%g' % var[6] + ' , ')
        #     # f.write('predict_entropy: ' + '%g' % pe + '\n' + '\n')
        #     # f.write('%g' % pe + '\n' + '\n')
# 批量打印
        # with open('/home/yml/second.pytorch/second/result_evaluate/result_51.txt', 'a') as f:
        #     if mean[8] < 26.75 and mean[9] < 2.3 and mean[10] < 43.5:
        #         f.write('%g' % var[5] + ' '+ '%g' % pe + ' '+'%g' % var[7] + ' ' + '%g' % var[8]+ ' ' + '%g' % var[6]+ ' '
        #                 + '%g' % var[9]+ ' ' + '%g' % var[10]+ ' ' + '%g' % var[11]+ ' ' + '%g' % var[12]+ ' '
        #                 + '%g' % mean[6] + ' ' + '%g' % mean[7] + ' ' + '%g' % mean[5] + ' '
        #                + '%g' % mean[8] + ' ' + '%g' % mean[9] + ' ' + '%g' % mean[10] +' ' + '%g' % mean[11] +'\n')







        #
        # with open('/home/yml/second.pytorch/second/result_evaluate/result_82.txt', 'a') as f:
        #     f.write('%g' % pe + ' , ') # entropy
        #
        # with open('/home/yml/second.pytorch/second/result_evaluate/result_53.txt', 'a') as f:
        #     # mean_score
        #     f.write('%g' % mean[4] + ' , ')
        #
        # with open('/home/yml/second.pytorch/second/result_evaluate/result_81.txt', 'a') as f:
        #     # mean_w
        #     f.write('%g' % mean[5] + ' , ')
        #
        # with open('/home/yml/second.pytorch/second/result_evaluate/result_81.txt', 'a') as f:
        #     # mean_h
        #     f.write('%g' % mean[6] + ' , ')
        #
        # with open('/home/yml/second.pytorch/second/result_evaluate/result_81.txt', 'a') as f:
        #     # mean_w
        #     f.write('%g' % mean[7] + ' , ')
        #
        # with open('/home/yml/second.pytorch/second/result_evaluate/result_82.txt', 'a') as f:
        #     # var_dimen
        #     f.write('%g' % mean[2] + ' , ')
        #
        # with open('/home/yml/second.pytorch/second/result_evaluate/result_53.txt', 'a') as f:
        #     # var_locat
        #     f.write('%g' % mean[8] + ' , ')
        #




