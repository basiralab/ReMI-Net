import torch
import numpy as np

from sklearn.model_selection import KFold
from torch_geometric.data import Data

# We used 35813 (part of the Fibonacci Sequence) as the seed.
np.random.seed(35813)
# Check device. You can manually change this line to use cpu only, do not forget to change it in all other files.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Antivectorize given vector
def antiVectorize(vec, m):
    M = np.zeros((m,m))
    M[np.tril_indices(m,k=-1)] = vec
    M= M.transpose()
    M[np.tril_indices(m,k=-1)] = vec
    return M

# CV splits
def get_nfold_split(X, number_of_folds, current_fold_id):
    kf = KFold(n_splits=number_of_folds,shuffle=True)
    split_indices = kf.split(range(X.shape[0]))
    train_indices, test_indices = [(list(train), list(test)) for train, test in split_indices][current_fold_id]
    #Split train and test
    X_train = X[train_indices]
    X_test = X[test_indices]
    return X_train, X_test

# Utility function to create a single multigraph from given numpy tensor: (n_rois, n_rois, n_views)
def create_graph_obj(mat):
    N_ROI = mat.shape[0] 
    VIEWS = mat.shape[-1]
    # There are not any node features, so all are assigned as 1.
    x = np.ones((N_ROI, 1))
    # We do not use any labels, so all are assigned as 0.
    y = np.zeros((1,))
    y[0] = None

    # Torch operations execute faster to create source-destination pairs.
    edge_attr = mat.reshape((N_ROI*N_ROI,VIEWS)) # Edge weights
    src_index = torch.arange(N_ROI).expand(N_ROI,N_ROI).transpose(0,1).reshape(N_ROI*N_ROI) # [0,0,0,0,1,1,1,1...]
    dst_index = torch.arange(N_ROI).repeat(N_ROI) # [0,1,2,3,0,1,2,3...]
    edge_index = torch.stack([src_index,dst_index]) # COO Matrix for index src-dst pairs.

    edge_attr = torch.from_numpy(edge_attr).float()
    x = torch.from_numpy(x).float()
    y = torch.from_numpy(y).float()
    con_mat = torch.from_numpy(mat).float()
    return Data(x = x, edge_index=edge_index, edge_attr=edge_attr, con_mat = con_mat,  y=y, label = None).to(device)

# Create torch-geometric data objects.
def cast_data(subjects):
    dataset = []
    for subject in subjects:
        all_times = []
        for timepoint in subject:
            # Each subject has one multigraph for each timepoint.
            all_times.append(create_graph_obj(timepoint))
        dataset.append(all_times)
    return dataset

if __name__=="__main__":
    pass