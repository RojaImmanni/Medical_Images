from utilities import *


def train_triangular_policy(model, optimizer, train_dl, valid_dl, valid_dataset, loss_fn, dataset,  
                            binary=None, max_lr=0.04, epochs=10):
    """
    Defining LR policy and training
    """
    idx = 0
    iterations = epochs*len(train_dl)
    lrs = get_cosine_triangular_lr(max_lr, iterations)
    for i in range(epochs):
        start = datetime.now()
        model.train()
        total = 0
        sum_loss = 0
        for i, (x, y) in enumerate(train_dl):
            lr = lrs[idx]
            update_optimizer(optimizer, [lr/9, lr/3, lr])
            batch = y.shape[0]
            
            x = x.cuda().float()
            y = y.cuda()
            
            if binary: out = model(x).view(-1)
            else: out = model(x)
                
            loss = loss_fn(out, y.float()) 

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            idx += 1
            total += batch
            sum_loss += batch*(loss.item())
        train_loss = sum_loss/total
        
        if dataset=='mura':
            val_loss, val_score = val_metrics_mura(model, valid_dl, valid_dataset)
        elif dataset=='rsna':
            val_loss, val_score = val_metrics_rsna(model, valid_dl)
        elif dataset=="chexpert":
            val_loss, val_score = val_metrics_chexpert(model, valid_dl)
            
        print("train_loss %.3f val_loss %.3f val_auc_score %.3f" % (train_loss, val_loss, val_score))
        end = datetime.now()
        print("----End of step", (end - start))
    return val_score, (end - start)


def LR_range_finder(model, train_dl, loss_fn,
                    binary=None, lr_low=1e-6, lr_high=0.5, epochs=1):
    
    PATH = Path('/home/rimmanni/data/')
    losses = []
    p = PATH/"model_temp.pth"
    save_model(model, str(p))
    iterations = epochs * len(train_dl)
    delta = (lr_high - lr_low)/iterations
    lrs = [lr_low + i*delta for i in range(iterations)]
    optimizer = create_optimizer(model, lrs[0])
    model.train()
    ind = 0
    for i in range(epochs):
        for x,y in train_dl:
            lr = lrs[ind]
            update_optimizer(optimizer, [lr/9, lr/3, lr])
            x = x.cuda().float()
            y = y.cuda()
            
            if binary: out = model(x).view(-1)
            else: out = model(x)

            loss = loss_fn(out, y.float())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            ind +=1

    load_model(model, str(p))
    return lrs, losses


def val_metrics_rsna(model, valid_dl):

    model.eval()
    total = 0
    sum_loss = 0
    probs = []
    ys = []

    for x, y in valid_dl:
        batch = y.shape[0]
        x = x.cuda().float()
        y = y.cuda()
        out = model(x).view(-1)

        loss = F.binary_cross_entropy_with_logits(out, y.float())  # pos_weight=weights
        sum_loss += batch*(loss.item())
        total += batch

        probs += list(out.cpu().detach().numpy())
        ys += list(y.cpu().numpy())


    auc_score = metrics.roc_auc_score(ys, probs)

    return sum_loss/total, auc_score


## Mura Dataset
def val_metrics_mura(model, valid_dl, valid_dataset):

    model.eval()
    total = 0
    sum_loss = 0
    probs = []
    ys = []
    weights = torch.Tensor([0.404]).cuda()

    for x, y in valid_dl:
        batch = y.shape[0]
        x = x.cuda().float()
        y = y.cuda()
        out = model(x).view(-1)

        loss = F.binary_cross_entropy_with_logits(out, y.float())  # pos_weight=weights
        sum_loss += batch*(loss.item())
        total += batch

        probs += list(out.cpu().detach().numpy())
        ys += list(y.cpu().numpy())

    valid_df = pd.DataFrame()
    valid_df['name'] = ['_'.join(f[0].split('_')[3:7]) for f in valid_dataset.samples]
    valid_df['ys'] = ys
    valid_df['probs'] = probs
    valid_df = valid_df.groupby('name').mean().reset_index()
    auc_score = metrics.roc_auc_score(valid_df["ys"], valid_df["probs"])

    return sum_loss/total, auc_score


def val_metrics_chexpert(model, valid_dl):

    model.eval()
    total = 0
    sum_loss = 0
    probs = []
    ys = []

    for x, y in valid_dl:
        batch = y.shape[0]
        x = x.cuda().float()
        y = y.cuda()
        out = model(x).squeeze()

        loss = F.binary_cross_entropy_with_logits(out, y.float()) 
        sum_loss += batch*(loss.item())
        total += batch

        probs += list(out.squeeze().cpu().detach().numpy())
        ys += list(y.long().cpu().numpy())
    
    probs = np.vstack(probs)
    ys = np.vstack(ys)
    
    aucs = [metrics.roc_auc_score(ys[:, i], probs[:, i]) for i in range(probs.shape[1])]

    return sum_loss/total, np.mean(aucs)