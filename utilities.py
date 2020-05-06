from imports import *

def read_image(path):
    im = cv2.imread(str(path))
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

def plot_lr(lrs, losses):
    """
    Given list of learning rates and corresponding losses, gives a smoothened line plot of losses to find the optimal learning
    rate
    """
    losses = smoothen_by_spline(lrs, losses)
    fig, ax = plt.subplots(1,1)
    ax.plot(lrs, losses)
    ax.set_ylabel("Loss")
    ax.set_xlabel("Learning Rate")
    ax.set_xscale('log')
    
def smoothen_by_spline(xs, ys):
    xs = np.arange(len(ys))
    spl = interpolate.UnivariateSpline(xs, ys)
    ys = spl(xs)
    return ys


def create_optimizer(model, lr0):
    """
    Creates an optimizer with layer-wise learning rate but fixed ratios
    """
    params = [{'params': model.features1.parameters(), 'lr': lr0/9},
              {'params': model.features2.parameters(), 'lr': lr0/3},
              {'params': model.classifier.parameters(), 'lr': lr0}]
    return optim.Adam(params, weight_decay=1e-5)

def update_optimizer(optimizer, group_lrs):
    for i, param_group in enumerate(optimizer.param_groups):
        param_group["lr"] = group_lrs[i]

def save_model(m, p): torch.save(m.state_dict(), p)

def load_model(m, p): m.load_state_dict(torch.load(p))
 
    
## Design traingular learning rates with annealing
def cosine_segment(start_lr, end_lr, iterations):
    i = np.arange(iterations)
    c_i = 1 + np.cos(i*np.pi/iterations)
    return end_lr + (start_lr - end_lr)/2 *c_i

def get_cosine_triangular_lr(max_lr, iterations):
    min_start, min_end = max_lr/10, max_lr/(100)
    iter1 = int(0.2*iterations)
    iter2 = iterations - iter1
    segs = [cosine_segment(min_start, max_lr, iter1), cosine_segment(max_lr, min_end, iter2)]
    return np.concatenate(segs)
    
