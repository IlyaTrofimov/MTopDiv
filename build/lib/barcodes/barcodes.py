from sklearn.metrics.pairwise import pairwise_distances
import subprocess
from copy import copy
import numpy as np
import matplotlib.pyplot as plt
import torch

def pdist_gpu(a, b, device = 'cuda:0'):
    A = torch.tensor(a, dtype = torch.float64)
    B = torch.tensor(b, dtype = torch.float64)

    size = (A.shape[0] + B.shape[0]) * A.shape[1] / 1e9
    max_size = 0.2

    if size > max_size:
        parts = int(size / max_size) + 1
    else:
        parts = 1

    pdist = np.zeros((A.shape[0], B.shape[0]))
    At = A.to(device)

    for p in range(parts):
        i1 = int(p * B.shape[0] / parts)
        i2 = int((p + 1) * B.shape[0] / parts)
        i2 = min(i2, B.shape[0])

        Bt = B[i1:i2].to(device)
        pt = torch.cdist(At, Bt)
        pdist[:, i1:i2] = pt.cpu()

        del Bt, pt
        torch.cuda.empty_cache()

    del At

    return pdist

def sep_dist(a, b, pdist_device = 'cpu'):
    if pdist_device == 'cpu':
        d1 = pairwise_distances(b, a, n_jobs = 40)
        d2 = pairwise_distances(b, b, n_jobs = 40)
    else:
        d1 = pdist_gpu(b, a, device = pdist_device)
        d2 = pdist_gpu(b, b, device = pdist_device)

    s = a.shape[0] + b.shape[0]

    apr_d = np.zeros((s, s))
    apr_d[a.shape[0]:, :a.shape[0]] = d1
    apr_d[a.shape[0]:, a.shape[0]:] = d2

    return apr_d

lines10k = []
for i in range(10000):
    lines10k.append('0,' * i + '\n')

def count_cross_barcodes(cloud_1, cloud_2, dim, title = '', cuda = 0, is_plot = True, pdist_device = 'cpu'):

    if pdist_device != 'cpu':
        pdist_device = 'cuda:%d' % cuda

    d = sep_dist(cloud_1, cloud_2, pdist_device = pdist_device)
    m = d[cloud_1.shape[0]:, :cloud_1.shape[0]].mean()
    d[:cloud_1.shape[0]][:cloud_1.shape[0]] = 0
    d[d < m*(10e-10)] = 0

    matrix_file = 'cloud_distance.lower_triangular'

    lines = []
    start_idx = min(10000, cloud_1.shape[0])
    lines = copy(lines10k[0:start_idx])

    for i in range(start_idx, d.shape[0]):
        for j in range(i):
            lines.append(str(d[i][j]) + ',')
        lines.append('\n')

    f = open(matrix_file, 'w')
    f.write(''.join(lines))
    f.close()

    command = 'CUDA_VISIBLE_DEVICES={2} /ripser-plusplus/build/ripser++ --dim {0} {1} 1> outrp.gpu.barcodes 2> outrp.gpu.perf'.format(dim, matrix_file, cuda)
    subprocess.call(command, shell=True)

    f = open('outrp.gpu.perf', 'r')
    found = False
    for line in f:
        if line.startswith('recalculated dim_max'):
            found = True
            dim_max = int(line.split(' ')[-1])

            #print('dim_max', dim_max)
            if dim_max < dim:
                raise ValueError('Not enough GPU memory!')

    if not found:
        raise ValueError('Not enough GPU memory!')

    f = open('outrp.gpu.barcodes', 'r')
    barcodes = []
    b = []
    for line in f:
        if line[1] == '[':
            x, y = line[2:-2].split(',')
            if y == ' ':
                continue
                #y = float('inf')
            b.append([float(x), float(y)])
        if line[0:11] == 'persistence':
            if (len(b) == 0):
                continue
            barcodes.append(np.array(b))
            b = []
    if(len(b) != 0):
        barcodes.append(np.array(b))

    f.close()

    barcodes = np.array(barcodes, dtype=object)

    if is_plot:
      plot_barcodes(barcodes, title = title)
      plt.show()
    return barcodes
   
def plot_barcodes(arr, color_list = ['deepskyblue', 'limegreen', 'darkkhaki'], dark_color_list = None, title = '', hom = None):

    if dark_color_list is None:
        dark_color_list = color_list
        #dark_color_list = ['b', 'g', 'orange']

    sh = arr.shape[0]
    step = 0
    if (len(color_list) < sh):
        color_list *= sh

    for i in range(sh):

        if not (hom is None):
            if i not in hom:
                continue

        barc = arr[i].copy()
        arrayForSort = np.subtract(barc[:,1],barc[:,0])

        bsorted = np.sort(arrayForSort)
        nbarc = bsorted.shape[0]
        print(nbarc)
        print('max0,976Barcode',i,'=',bsorted[nbarc*976//1000])
        print('maxBarcode',i,'=',bsorted[-1])
        print('middleBarcode',i,'=',bsorted[nbarc//2])
        #print('minbarcode',i,'=',bsorted[0])
        max = bsorted[-3:]
        plt.plot(barc[0], np.ones(2)*step, color = color_list[i], label = 'H{}'.format(i))
        for b in barc:
            if b[1] - b[0] in max :
                plt.plot(b, np.ones(2)*step, dark_color_list[i])
            else:
                plt.plot(b, np.ones(2)*step, color = color_list[i])
            step += 1

    plt.xlabel('$\epsilon$ (time)')
    plt.ylabel('segment')
    plt.title(title)
    plt.legend(loc = 'lower right')
    plt.rcParams["figure.figsize"] = [6, 4]

def calc_cross_barcodes(cloud_1, cloud_2, batch_size1 = 4000, batch_size2 = 200, cuda = 0, pdist_device = 'cpu', dim = 1, is_plot = True):
    dark_color_list = ['deepskyblue', 'maroon', 'g', 'darkorange']
    color_list = ['paleturquoise', 'lightcoral', 'lightgreen', 'lightsalmon']

    batch_size1 = min(batch_size1, cloud_1.shape[0])
    batch_size2 = min(batch_size2, cloud_2.shape[0])

    indexes_1 = np.random.choice(cloud_1.shape[0], batch_size1, replace=False)
    indexes_2 = np.random.choice(cloud_2.shape[0], batch_size2, replace=False)
    cl_1 = cloud_1[indexes_1]
    cl_2 = cloud_2[indexes_2]
    barc = count_cross_barcodes(cl_1, cl_2, dim, is_plot = is_plot, title = '', cuda = cuda, pdist_device = pdist_device)

    return barc

def get_score(elem, h_idx, kind = ''):
    if elem.shape[0] >= h_idx + 1:

        barc = elem[h_idx]
        arrayForSort = np.subtract(barc[:,1], barc[:,0])

        bsorted = np.sort(arrayForSort)

        # number of barcodes
        if kind == 'nbarc':
            return bsorted.shape[0]

        # largest barcode
        if kind == 'largest':
            return bsorted[-1]

        # quantile
        if kind == 'quantile':
            idx = int(0.976 * len(bsorted))
            return bsorted[idx]

        # sum of length
        if kind == 'sum_length':
            return np.sum(bsorted)

        # sum of squared length
        if kind == 'sum_sq_length':
            return np.sum(bsorted**2)

        raise ValueError('Unknown kind of score')

    return 0

def mtopdiv(P, Q, batch_size1 = 1000, batch_size2 = 10000, n = 20, cuda = 0, pdist_device = 'cuda:0', is_plot = False):
    barcs = [calc_cross_barcodes(Q, P, batch_size2, batch_size1, cuda = cuda, pdist_device = pdist_device, is_plot = is_plot) for _ in range(n)]
    return np.mean([get_score(x, 1, 'sum_length') for x in barcs])
