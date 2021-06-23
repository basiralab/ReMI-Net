import torch
torch.manual_seed(35813)

# Check device. You can manually change this line to use cpu only, do not forget to change it in all other files.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Centeredness with Frobenious Distance
def frobenious_distance(mat1,mat2):
    # Utility
    # Return matrix shaped: (n_views, n_subjects)
    # Each View of each subject has corresponding distance calculation.
    return torch.sqrt(torch.square(torch.abs(mat1 - mat2)).sum(dim=(2,3))).transpose(1,0).to(device)

def view_normalization(views):
    # Utility
    # Return matrix shaped: (n_views, n_subjects)
    # Each View of each subject has different view normalizer.
    return views.mean(dim=(1,2,3)).max() / views.mean(dim=(1,2,3))

def FrobLoss(cbt, subjects, sample_size=10, aggr="sum"):
    subset = subjects[torch.randint(len(subjects), (sample_size,))].to(device)
    subset = subset.reshape((subset.size()[3],subset.size()[0],subset.size()[1],subset.size()[2]))
    if aggr == "sum":return (frobenious_distance(cbt,subset)*view_normalization(subset)).sum()
    if aggr == "mean":return (frobenious_distance(cbt,subset)*view_normalization(subset)).mean()

def TestFrobLoss(cbts, targets, sample_size=10,lambda1=0.3):
    # Separate measuring of all losses for ReMI-Net evaluation.
    reg_loss = []
    loss=[]
    for idx,cbt in enumerate(cbts):
        loss.append(FrobLoss(cbt, targets[:,idx,:,:,:],sample_size=sample_size,aggr="mean"))
        if idx != 0: 
            reg_loss.append(lambda1 * torch.sqrt(torch.square((cbt - cbt[idx - 1])).sum()))
        else: continue
        
    return torch.stack(loss).to(device), torch.stack(reg_loss).to(device)

def MultiFrobLoss(cbts, targets, sample_size=10, aggr="sum",lambda1=0.3):
    # Combination of all losses are measured during ReMI-Net training.
    losses = []
    for idx,cbt in enumerate(cbts):
        # Frob Loss is sum of distances: (views - population)
        loss = FrobLoss(cbt, targets[:,idx,:,:,:],sample_size=sample_size, aggr=aggr)
        if idx != 0: 
            loss = loss + lambda1 * torch.sqrt(torch.square((cbt - cbt[idx - 1])).sum())
        losses.append(loss)
    return torch.stack(losses).to(device)

if __name__=="__main__":
    samples = torch.rand(100,3,35,35,4)
    cbt = torch.rand(35,35)
    sub = samples[0,0,:,:,:]