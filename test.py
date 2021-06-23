import torch
import numpy as np

from dataset import prepare_data
from utils import get_nfold_split, cast_data
from plotting import plot_cbt
from loss import MultiFrobLoss, TestFrobLoss
from model import RemiNet

# These two options should be seed to ensure reproducible (If you are using cudnn backend)
# https://pytorch.org/docs/stable/notes/randomness.html
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

np.random.seed(35813)
torch.manual_seed(35813)

# Check device. You can manually change this line to use cpu only, do not forget to change it in all other files.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("RUNS ON:",device)

# MODEL PARAMETERS (must be same with the trained model)
h1_size = 12
h2_size = 36
h3_size = 24

def test_reminet(X, n_folds=5,timepoints=2,data_name="sim",recursion="cyclic",norm="sigmoidnorm"):
    test_losses = []
    reg_losses = []
    print("Testing ReMI-Net...")
    for i in range(n_folds):
        print(f"********* FOLD {i} *********")
        torch.cuda.empty_cache() 
        
        # Prepare CV splits and torch-geometric data.
        train_data, test_data = get_nfold_split(X, number_of_folds=n_folds, current_fold_id=i)
        train_casted = cast_data(train_data)
        test_casted = cast_data(test_data)
        test_targets = torch.from_numpy(test_data).to(device)
        shape = train_data.shape

        # Load trained model.
        model = RemiNet(shape[-2],shape[-1],[h1_size,h2_size,h3_size],recursion,norm) # All parameters must be exactly the same with the model that you are trying to load.
        model.load_state_dict(torch.load(f"./models/{data_name}data_{recursion}_{norm}_model_fold{i}"))
        model.to(device)
        # Change model mode to evaluation before running the test.
        model.eval()

        # Population CBTs.
        generated_cbts = model.generate_cbt_median(test_casted,t=timepoints)
        # Calculate loss.
        test_loss, reg_loss = TestFrobLoss(generated_cbts,test_targets,sample_size=len(test_targets))
        
        test_loss = np.round(test_loss.detach().cpu().numpy(),3)
        reg_loss = np.round(reg_loss.detach().cpu().numpy(),3)

        test_losses.append(test_loss)
        reg_losses.append(reg_loss)

        # Plot population CBTs.
        for t, cbt in enumerate(generated_cbts):
            plot_cbt(cbt.detach().cpu().numpy(),i+1,t)

    return np.array(test_losses), np.array(reg_losses)

        
# --------------------------------------------------------------
# SHAPE: (n_subjects, n_timepoints, n_roi, n_roi, n_views)
# --------------------------------------------------------------
if __name__=="__main__":
    X = prepare_data()
    # data_name can be anything that you used for the name of the pretrained model.
    # model type must be "cyclic" or "vanilla"
    # norm_method must be "no_norm", "sigmoidnorm" or "minmax"
    test_losses, reg_losses = test_reminet(X,timepoints=X.shape[1],data_name="sim",recursion="cyclic",norm="sigmoidnorm")