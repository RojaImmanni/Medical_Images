from training import *


def prepare_rsna_datasamples():
    path = Path('/mnt/wamri/rsna/rsna_retro')
    paths = (path/'nocrop_jpg256').iterdir()
    paths = [str(path) for path in paths]
    image_paths = [path.split('/')[-1].strip('.jpg') for path in paths]
    
    df = pd.read_csv(path/'stage_2_train.csv')
    df = df.loc[df.ID.str.contains('any')]
    df['ID'] = df['ID'].map(lambda x: x.split('_')[:2])
    df['ID'] = df['ID'].map(lambda x: ('_'.join(p for p in x)))
    df = df.loc[df.ID.isin(image_paths)]
    
    normal_list = list(df.loc[df.Label==0].ID)
    abnormal_list = list(df.loc[df.Label==1].ID)

    normal_samples = resample(normal_list, n_samples=40000, replace=False, random_state=1)
    abnormal_samples = resample(abnormal_list, n_samples=20000, replace=False, random_state=1)

    train_normal, test_normal = normal_samples[:20000], normal_samples[20000:]
    train_abnormal, test_abnormal = abnormal_samples[:10000], abnormal_samples[10000:]


    train_set, test_set = pd.DataFrame(), pd.DataFrame()
    train_set['ID'] = train_normal + train_abnormal
    train_set['Label'] = [0]*20000 + [1]*10000
    test_set['ID'] = test_normal + test_abnormal
    test_set['Label'] = [0]*20000 + [1]*10000
    
    return train_set, test_set
    
class rsna_dataset(Dataset):
    def __init__(self, train_set, transform=True):
        
        self.paths = ['/mnt/wamri/rsna/rsna_retro/nocrop_jpg256/'+str(p)+'.jpg' for p in train_set['ID']]
        self.labels = [p for p in train_set['Label']]
        self.len = len(self.paths)
        
        # Transformations
        if transform:
            self.tfms = transforms.Compose([transforms.RandomResizedCrop((256, 256), 
                                                                         scale= (0.8, 1), ratio=(0.7,1.3)),
                                            transforms.RandomHorizontalFlip(), 
                                            transforms.RandomRotation(30),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                           ])
            
        else:
            self.tfms = transforms.Compose([transforms.ToTensor(), 
                                            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                           ])
        
        
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        path = self.paths[idx] 
        x = Image.open(path)
        x = self.tfms(x)
        
        y = self.labels[idx]

        return x, y
    

def rsna_dataloaders(batch_size):
    train_set, test_set = prepare_rsna_datasamples()
    train_ds = rsna_dataset(train_set, transform=True)
    valid_ds = rsna_dataset(test_set, transform=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size,num_workers=4, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size,num_workers=4)
    
    return train_loader, valid_loader, valid_ds



## Mura dataset
def mura_dataloaders(batch_size):
    PATH = Path('/home/rimmanni/data/mura')
    train_path = PATH/"train_250_200"
    valid_path = PATH/"valid_250_200"
    
    train_dataset = datasets.ImageFolder(
            train_path,
            transforms.Compose([
                transforms.RandomCrop((250,200)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(30),
                transforms.ToTensor()
#                 transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ]))


    valid_dataset = datasets.ImageFolder(valid_path, transforms.Compose([
                transforms.ToTensor()
#                 transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ]))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size,num_workers=4, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size,num_workers=4)
    
    
    return train_loader, valid_loader, valid_dataset


#Chexpert Dataset
class chexpert_dataset(Dataset):
    def __init__(self, df, image_path, transform=None):
        self.image_files = df["Path"].values
        self.labels = df[['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']].values
        self.labels[self.labels==-1] = 0
        self.image_path = image_path
        if transform:
            self.tfms = transforms.Compose([transforms.RandomResizedCrop((256, 256), 
                                                                         scale= (0.8, 1), ratio=(0.7,1.3)),
                                            transforms.RandomHorizontalFlip(), 
                                            transforms.RandomRotation(30),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                           ])
            
        else:
            self.tfms = transforms.Compose([transforms.ToTensor(), 
                                            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                           ])
        self.len = len(self.image_files)
    
    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        path = self.image_path/self.image_files[index]
        x = Image.open(path)
        x = self.tfms(x)
        y = self.labels[index]
        return x, y

def get_chexpert_dataloaders(batch_size):
    PATH = Path('/mnt/wamri/gilmer/chexpert/ChesXPert-250')
    train_df = pd.read_csv('/home/rimmanni/experiments/chexpert/train_df_chexpert_resized.csv')
    valid_df = pd.read_csv('/home/rimmanni/experiments/chexpert/valid_df_chexpert_resized.csv')
    
    train_ds = chexpert_dataset(train_df, image_path=PATH, transform=True)
    valid_ds = chexpert_dataset(valid_df, image_path=PATH, transform=False)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, num_workers=4, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, num_workers=4)
    
    return train_loader, valid_loader, valid_ds