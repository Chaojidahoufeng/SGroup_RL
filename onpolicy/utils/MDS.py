from matplotlib import pyplot as plt
import numpy as np

# rotation and transit in LYP paper
def f_Relative(groundtruth, P, N):
    '''
    Inputs:
        groundtruth
        P: predicted topo
        N: number of nodes
    Outputs:
        P_final: topo rotated and transited
        Gamma: rotation mat
        t: transit argument
    '''
    I_Na = np.eye(N)
    # SVD
    M_h = P.T
    L = I_Na - np.ones([N, N]) * 1/N
    M = groundtruth.T
    S = np.matmul( np.matmul(M_h.T, L), M )
    U1, Lambda1, V1 = np.linalg.svd(S)
    
    #TODO: why should we use.T
    U1 = U1.T
    V1 = V1.T
    
    # calculate gamma mat and T mat
    gamma = np.matmul(V1, U1.T)
    t = np.matmul((M.T - np.matmul(gamma, M_h.T)), np.ones([N,1])) * 1 / N

    # V mat
    v_x = np.kron(np.ones([N,1]), np.expand_dims(np.array([1,0]),axis=-1))
    v_y = np.kron(np.ones([N,1]), np.expand_dims(np.array([0,1]),axis=-1))

    # rotation correction and transit correction
    P_array = P
    P_final = np.expand_dims(np.matmul(np.kron(I_Na, gamma), P_array.flatten(order='F')), axis=-1) + np.matmul(np.concatenate([v_x, v_y], axis=1), t)

    return P_final, gamma, t


def error_rel_g(p1, p2, n):
    p2_rel,_,_ = f_Relative(p1, p2, n)
    error2 = p2_rel - np.expand_dims(p1.reshape(-1, order='F'), axis=-1)
    error_rel = np.sqrt(1/n * np.sum(error2 ** 2))

    return error_rel
    

def MDS_relative_loc(N, X_groundtruth, M):
    '''
    Inputs:
        N: number of nodes; int
        X_groundtruth: X_groundtruth for rotation and transit; numpy array: [2, N]
        M: number of repeats
    
    Outputs:

    '''
    dsigma = 0.1
    RMSE = 0

    plt.scatter(X_groundtruth[0,:], X_groundtruth[1,:])

    for round_num in range(M):
    # MDS algorithm
        D = np.zeros([N, N])
        for i in range(N):
            for j in range(N):
                D[i,j] = (np.linalg.norm(X_groundtruth[:,i] - X_groundtruth[:,j]) + dsigma * np.random.randn())**2
                #D[i,j] = (np.linalg.norm(X_groundtruth[:,i] - X_groundtruth[:,j])) ** 2
                D[j,i] = D[i,j]
                if i == j:
                    D[i,j] = 0

        C = np.eye(N) - 1/N * np.ones([N,N])
        B = -1/2 * np.matmul(np.matmul(C, D), C)
        U, Lambda, V = np.linalg.svd(B)
        
        # normalize the SVD mat
        for i in range(N):
            if U[0,i]<0:
                U[:, i] = -U[:, i]
                V[:, i] = -V[:, i]
        
        U_norm = np.zeros([N,N])
        for i in range(N):
            U_norm[:, i] = U[:, i]/np.linalg.norm(U[:, i])
        
        V1 = U_norm * np.sqrt(Lambda)
        P_rel_ori = V1[:N, :2]
        P_rel_ori = P_rel_ori.T

        plt.scatter(P_rel_ori[0,:],P_rel_ori[1,:]);

        P_final, gamma, t = f_Relative(X_groundtruth, P_rel_ori, N)
        P_final_results = np.reshape(P_final, [2,4], order='F')

        plt.scatter(P_final_results[0,:],P_final_results[1,:])

        Error_rel = error_rel_g(X_groundtruth, P_rel_ori, N)
        RMSE = RMSE + Error_rel/M
        plt.show()

# N = 4
# l = 15
# lenth = 15*np.sqrt(2)/2

# X_groundtruth = [[-15, 15, 0, 0],
#                           [0, 0, 15, -15]]
# X_ideal = [[lenth, lenth, -lenth, -lenth],
#                     [lenth, -lenth, lenth, -lenth]]

# # plt.scatter(X_groundtruth[0], X_groundtruth[1])
# # plt.scatter(X_ideal[0], X_ideal[1])

# # plt.show()

# X_groundtruth = np.array([[-15, 15, 0, 0],
#                           [0, 0, 15, -15]])
# X_ideal = np.array([[lenth, lenth, -lenth, -lenth],
#                     [lenth, -lenth, lenth, -lenth]])
# M = 1
# # MDS_relative_loc(4, X_groundtruth, M)
# error_rel_g(X_groundtruth, X_ideal, 4)
        
  



