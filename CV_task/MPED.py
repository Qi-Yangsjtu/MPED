
import torch
from pytorch3d.ops.knn import knn_gather, knn_points
from pytorch3d.loss.chamfer import _handle_pointcloud_input
def SPED(x,x_nn,y_nn,T2,N,P1,neighbors,x_coor_near,y_coor_near,D):
    if neighbors>1:
        weight_x = 1/torch.sqrt(x_nn.dists[:,:,0:neighbors]+ T2) 
        weight_y = 1/torch.sqrt(y_nn.dists[:,:,0:neighbors]+ T2)

        weight_x = weight_x.reshape(N, P1, neighbors, 1)
        weight_y = weight_y.reshape(N, P1, neighbors, 1)
    else:
        weight_x=1
        weight_y=1

    x_copy = x.view(N,P1,1,D)
    x_Euclidean_difference_ori = (x_coor_near[:,:,0:neighbors,:] - x_copy)
    y_Euclidean_difference_ori = (y_coor_near[:,:,0:neighbors,:] - x_copy)
    # 2-norm
    x_Euclidean_difference = x_Euclidean_difference_ori**2
    y_Euclidean_difference = y_Euclidean_difference_ori**2

    Graph_mess_x_near = weight_x * x_Euclidean_difference
    Graph_mess_y_near = weight_y * y_Euclidean_difference

    energy_x = torch.sum(Graph_mess_x_near, dim=3)
    energy_y = torch.sum(Graph_mess_y_near, dim=3)
    energy_x = torch.sum(energy_x,dim = 2)
    energy_y = torch.sum(energy_y,dim = 2)
    dis_energy = (energy_x - energy_y).abs()  
    dis_energy = dis_energy.sum(dim=1) #batch
    return dis_energy
def feeature_pooling(x, x_lengths,y, y_lengths,neighbors):
    x = x
    x_lengths = x_lengths
    y = y
    y_lengths = y_lengths
    N, P1, D = x.shape
    P2 = y.shape[1]
    # T = 0.001
    T2 = 1e-10
    if P1<15 or P2<15:
        raise ValueError("x or y does not have the enough points (at lest 15 points).")

    x_nn = knn_points(x, x, lengths1=x_lengths, lengths2=x_lengths, K=neighbors)
    y_nn = knn_points(x, y, lengths1=x_lengths, lengths2=y_lengths, K=neighbors)

    x_coor_near = knn_gather(x, x_nn.idx, x_lengths)#batch,points,nei,3
    y_coor_near = knn_gather(y, y_nn.idx, x_lengths)

    if y.shape[0] != N or y.shape[2] != D:
        raise ValueError("y does not have the correct shape.")

    # three scales
    SPED1 = SPED(x, x_nn, y_nn, T2, N, P1, 10, x_coor_near, y_coor_near, D)
    SPED2 = SPED(x, x_nn, y_nn, T2, N, P1, 5, x_coor_near, y_coor_near, D)
    SPED3 = SPED(x, x_nn, y_nn, T2, N, P1, 1, x_coor_near, y_coor_near, D)

    SPED1 = SPED1.sum()
    SPED2 = SPED2.sum()
    SPED3 = SPED3.sum()
    MPED_SCORE = (SPED1+SPED2+SPED3)
    MPED_SCORE = MPED_SCORE / P1
    MPED_SCORE = MPED_SCORE / N
    return MPED_SCORE

def MPED_VALUE(
        x,
        y,
        x_lengths=None,
        y_lengths=None,
        x_normals=None,
        y_normals=None,
        neighbors=10,
):
    """
    MPED between two pointclouds x and y.

    Args:
        x: FloatTensor of shape (N, P1, D) or a Pointclouds object representing
            a batch of point clouds with at most P1 points in each batch element,
            batch size N and feature dimension D.
        y: FloatTensor of shape (N, P2, D) or a Pointclouds object representing
            a batch of point clouds with at most P2 points in each batch element,
            batch size N and feature dimension D.
        x_lengths: Optional LongTensor of shape (N,) giving the number of points in each
            cloud in x.
        y_lengths: Optional LongTensor of shape (N,) giving the number of points in each
            cloud in x.

    Returns:

        - **loss**: Tensor giving the MPED between the pointclouds
          in x and the pointclouds in y.
    """
    x, x_lengths, x_normals = _handle_pointcloud_input(x, x_lengths, x_normals)
    y, y_lengths, y_normals = _handle_pointcloud_input(y, y_lengths, y_normals)
    MPED1 = feeature_pooling(x, x_lengths, y, y_lengths, neighbors)
    MPED2 = feeature_pooling(y, y_lengths, x, x_lengths, neighbors)
    MPED_SCORE = MPED1+MPED2
    return MPED_SCORE
