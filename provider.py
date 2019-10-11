import os
import sys
import numpy as np
import h5py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

# Download dataset for point cloud classification
DATA_DIR = os.path.join(BASE_DIR, 'data')
if not os.path.exists(DATA_DIR):
    os.mkdir(DATA_DIR)
# if not os.path.exists(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048')):
#     www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
#     zipfile = os.path.basename(www)
#     os.system('wget %s; unzip %s' % (www, zipfile))
#     os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
#     os.system('rm %s' % (zipfile))


def shuffle_data(data, labels):
    """ Shuffle data and labels.
        Input:
          data: B,N,... numpy array
          label: B,... numpy array
        Return:
          shuffled data, label and shuffle indices
    """
    idx = np.arange(len(labels))
    np.random.shuffle(idx)
    return data[idx, ...], labels[idx], idx
def linear_norm_point_cloud(batch_data):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    normed_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        maxv = max(batch_data[k, ...].flatten())
        minv = min(batch_data[k, ...].flatten())
        normed_data[k,...] = (batch_data[k, ...] - minv)*255 / (maxv - minv)
    return normed_data
def add_square_to_point_cloud(batch_data):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    added_data = np.zeros((batch_data.shape[0],batch_data.shape[1],4), dtype=np.float32)
    # added_data[:,:,3:6] = np.square(batch_data)
    # added_data[:,:,3] = np.square(batch_data)
    for k in range(batch_data.shape[0]):
        for j in range(batch_data.shape[1]):
            added_data[k,j,3] = np.sqrt(batch_data[k, j, 0]**2 + batch_data[k, j, 1]**2 + batch_data[k, j, 2]**2)
            # added_data[k,j,4] = batch_data[k, j, 1]**2
            # added_data[k,j,5] = batch_data[k, j, 2]**2

    return added_data
def sort_axis_point_cloud(batch_data,num_point,axis=0):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    sorted_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        shape_pc = batch_data[k, ...]
        # gravity = shape_pc.mean(0)
        dis = np.zeros((num_point),dtype=np.float)
        for j in range(num_point):
            dis[j] = (shape_pc[j,axis])
            # dis[j] = (shape_pc[j,0]-gravity[0])**2+(shape_pc[j,1]-gravity[1])**2+(shape_pc[j,2]-gravity[2])**2
        dis_index = dis.argsort()

        for j in range(num_point):
            sorted_data[k,j,:] = shape_pc[dis_index[j],:]
    return sorted_data

def sort_point_cloud(batch_data,num_point):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    sorted_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        shape_pc = batch_data[k, ...]
        # gravity = shape_pc.mean(0)
        dis = np.zeros((num_point),dtype=np.float)
        for j in range(num_point):
            dis[j] = (shape_pc[j,0])**2+(shape_pc[j,1])**2+(shape_pc[j,2])**2
            # dis[j] = (shape_pc[j,0]-gravity[0])**2+(shape_pc[j,1]-gravity[1])**2+(shape_pc[j,2]-gravity[2])**2
        dis_index = dis.argsort()

        for j in range(num_point):
            sorted_data[k,j,:] = shape_pc[dis_index[j],:]
    return sorted_data

def rotate_point_cloud(batch_data):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    rand_arr = np.random.randint(0,2,batch_data.shape[0])
    for k in range(batch_data.shape[0]):
        if rand_arr[k] == 1:
            rotation_angle = np.random.uniform() * 2 * np.pi
            cosval = np.cos(rotation_angle)
            sinval = np.sin(rotation_angle)
            rotation_matrix = np.array([[cosval, 0, sinval],
                                        [0, 1, 0],
                                        [-sinval, 0, cosval]])
            shape_pc = batch_data[k, ...]
            rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
        else:
            rotated_data[k, ...] = batch_data[k, ...]
    return rotated_data

def multi_rotate_point_cloud(batch_data,times=4):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros((batch_data.shape[0]*(times+1),batch_data.shape[1],batch_data.shape[2]), dtype=np.float32)
    rotated_data[0:batch_data.shape[0],...] = batch_data

    for i in range(1,times):
        for k in range(batch_data.shape[0]):
            rotation_angle = np.random.uniform() * 2 * np.pi
            cosval = np.cos(rotation_angle)
            sinval = np.sin(rotation_angle)
            rotation_matrix = np.array([[cosval, 0, sinval],
                                        [0, 1, 0],
                                        [-sinval, 0, cosval]])
            shape_pc = batch_data[k, ...]
            rotated_data[batch_data.shape[0]*i+k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data


def rotate_point_cloud_by_angle(batch_data, rotation_angle):
    """ Rotate the point cloud along up direction with certain angle.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        #rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data


def jitter_point_cloud(batch_data, sigma=0.01, clip=0.05):
    """ Randomly jitter points. jittering is per point.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, jittered batch of point clouds
    """
    if np.random.randint(0,2) == 1:
        B, N, C = batch_data.shape
        assert(clip > 0)
        jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1*clip, clip)
        jittered_data += batch_data
    else:
        jittered_data = batch_data
    return jittered_data

def getDataFiles(list_filename):
    return [line.rstrip() for line in open(list_filename)]

def load_h5(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    return (data, label)

def loadDataFile(filename):
    return load_h5(filename)

def load_h5_data_label_seg(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    seg = f['pid'][:]
    return (data, label, seg)


def loadDataFile_with_seg(filename):
    return load_h5_data_label_seg(filename)
