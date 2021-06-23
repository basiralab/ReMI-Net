import torch
import numpy as np
import time
import matplotlib.pyplot as plt

from dataset import prepare_data
from loss import MultiFrobLoss, TestFrobLoss
from model import RemiNet
from utils import get_nfold_split, cast_data

# We used 35813 (part of the Fibonacci Sequence) as the seed when we conducted experiments
np.random.seed(35813)
torch.manual_seed(35813)

# These two options should be seed to ensure reproducible (If you are using cudnn backend)
# https://pytorch.org/docs/stable/notes/randomness.html
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Check device. You can manually change this line to use cpu only, do not forget to change it in all other files.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("RUNS ON:",device)

# MODEL PARAMETERS
h1_size = 12
h2_size = 36
h3_size = 24

learning_rate = 0.0008

def train(X, n_max_epochs, time_points = 3, sampling_size = 10, n_folds = 5, dataset="sim", model_type="cyclic",norm_method="sigmoidnorm"):

    all_train_Lc = []
    all_train_Lt = []
    all_test_Lc = []
    all_test_Lt = []
    print("Training ReMI-Net...")
    for i in range(n_folds):
        # Train a model for each fold.
        torch.cuda.empty_cache() 
        print(f"********* FOLD {i} *********")
        train_data, test_data = get_nfold_split(X, number_of_folds=n_folds, current_fold_id=i)
        shape = train_data.shape

        # Numpy tensor to list of torch-geometric Data objects.
        train_casted = cast_data(train_data)
        test_casted = cast_data(test_data)

        # Numpy tensor to torch tensor.
        train_targets = torch.from_numpy(train_data).to(device)
        test_targets = torch.from_numpy(test_data).to(device)

        # Define model and optimizer.
        model = RemiNet(shape[-2],shape[-1],[h1_size,h2_size,h3_size],model_type,norm_method).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay= 0.00)
        model.train()

        # Define losses: Centeredness Test, Centeredness Train, Time Regularization Test, Time Regularization Train.
        tst_center = []
        tr_center = []
        tst_reg = []
        tr_reg = []
        tick = time.time()
        for epoch in range(n_max_epochs):
            losses = []
            for data in train_casted:
                # Forward pass of each subject.
                cbts = model.forward(data,time_points,cycles=1,stop_point=0)
                # MultiFrobLoss returns a single loss value (Lt + Lc).
                losses.append(MultiFrobLoss(cbts,train_targets))
            # Total loss.
            loss = torch.mean(torch.stack(losses))

            # Backpropagate.
            optimizer.zero_grad()
            loss.backward()
            # Start cuda sync as earliest as possible to save time.
            torch.cuda.synchronize()
            optimizer.step()

            if epoch % 5 == 0 or epoch == n_max_epochs-1:
                # Test the trained model at each 5 epochs.
                generated_cbts = model.generate_cbt_median(train_casted,time_points)
                generated_tst_cbts = model.generate_cbt_median(test_casted,time_points)

                # TestFrobLoss returns the two losses separately (Lc, Lt)
                train_loss, train_reg_loss = TestFrobLoss(generated_cbts,test_targets,sample_size=len(test_targets)) # Train-to-Test Loss
                test_loss, test_reg_loss = TestFrobLoss(generated_tst_cbts,test_targets,sample_size=len(test_targets)) # Test-to-Test Loss
                
                train_loss = np.round(train_loss.detach().cpu().numpy(),3)
                train_reg_loss = np.round(train_reg_loss.detach().cpu().numpy(),3)

                test_loss = np.round(test_loss.detach().cpu().numpy(),3)
                test_reg_loss = np.round(test_reg_loss.detach().cpu().numpy(),3)

                tock = time.time()
                print(f"Epoch: {epoch}/{n_max_epochs} | Train Loss: {train_loss} | Train Reg Loss: {train_reg_loss} | Test Loss: {test_loss} | Test Reg Loss: {test_reg_loss} | Time: {round(tock-tick,3)}")
                tick = time.time()

                tst_reg.append(test_reg_loss)
                tr_reg.append(train_reg_loss)
                tst_center.append(test_loss)
                tr_center.append(train_loss)

        tock = time.time()
        print(f"FINAL | Train Loss: {train_loss} | Train Reg Loss: {train_reg_loss} | Test Loss: {test_loss} | Test Reg Loss: {test_reg_loss}")

        torch.save(model.state_dict(), f"./models/{dataset}data_{model_type}_{norm_method}_model_fold{i}")
        all_test_Lt.append(tst_reg)
        all_train_Lt.append(tr_reg)

        all_test_Lc.append(tst_center)
        all_train_Lc.append(tr_center)

    test_losses, test_reg, train_losses, train_reg = np.array(all_test_Lc), np.array(all_test_Lt), np.array(all_train_Lc), np.array(all_train_Lt)
    
    Lc_losses = np.stack([train_losses,test_losses])
    Lt_losses = np.stack([train_reg,test_reg])

    # Save all results.
    np.save(f"./experiments/{dataset}data_L1_{model_type}_{norm_method}_model.npy", Lc_losses)
    np.save(f"./experiments/{dataset}data_L2_{model_type}_{norm_method}_model.npy", Lt_losses)
    return Lc_losses, Lt_losses

            
if __name__=="__main__":
    X = prepare_data()
    # You can give any name to dataset you are using, it also changes the model name.
    # model type must be "cyclic" or "vanilla"
    # norm_method must be "no_norm", "sigmoidnorm" or "minmax"
    Lc_losses, Lt_losses = train(X,n_max_epochs=100,time_points=X.shape[1],dataset="sim",model_type="cyclic",norm_method="sigmoidnorm")

