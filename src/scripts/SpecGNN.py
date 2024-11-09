import torch
from torch import nn
import math
from torch_geometric.nn.conv import GINEConv
from torch_scatter import scatter
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.data import Dataset, Data
from scipy.stats import spearmanr
import pandas as pd
import numpy as np
from torch_geometric.loader import DataLoader

def one_hot_encoding(number,size):
    assert number <= size-1
    encode_list=list(0 for _ in range(size))
    encode_list[number] = 1
    return encode_list

def get_bond_features(bond):
    if bond is not None:
        bt = bond.GetBondType()
        fbond = [
            0,  # bond is not None
            bt == Chem.rdchem.BondType.SINGLE,
            bt == Chem.rdchem.BondType.DOUBLE,
            bt == Chem.rdchem.BondType.TRIPLE,
            bt == Chem.rdchem.BondType.AROMATIC,
            (bond.IsInRingSize(3) if bt is not None else 0),
            (bond.IsInRingSize(4) if bt is not None else 0),
            (bond.IsInRingSize(5) if bt is not None else 0),
            (bond.IsInRingSize(6) if bt is not None else 0),
            (bond.IsInRingSize(7) if bt is not None else 0),
            (bond.IsInRingSize(8) if bt is not None else 0),
            (bond.IsInRingSize(9) if bt is not None else 0),
            (bond.IsInRingSize(10) if bt is not None else 0),
        ]
        fbond.extend(one_hot_encoding(int(bond.IsInRing()), 2))
        fbond.extend(one_hot_encoding(int(bond.GetIsConjugated()), 2))
        fbond.extend(one_hot_encoding(int(bond.GetStereo()),7))
    else:
        fbond=None
        AssertionError('bond is none')
    return fbond


def get_atom_features(atom):
    chi=int(atom.GetChiralTag())
    atomic_number=int(atom.GetAtomicNum())
    hyh=int(atom.GetHybridization())
#     print(hyh)
    fatom=[]
    fatom.extend(one_hot_encoding(atomic_number-1,54))
    fatom.extend(one_hot_encoding(chi,4))
    fatom.extend(one_hot_encoding(atom.GetTotalDegree()-1, 6))
    fatom.extend(one_hot_encoding(atom.GetFormalCharge()+3, 7))
    fatom.extend(one_hot_encoding(hyh, 8))
    fatom.extend(one_hot_encoding(int(atom.GetIsAromatic()),2))
    fatom.extend(one_hot_encoding(atom.GetTotalValence()-1, 8))
    fatom.extend(one_hot_encoding(int(atom.IsInRing()), 2))
    for cy in range(3,11):
        fatom.append(1 if atom.IsInRingSize(cy) else 0)
    return fatom


def smile2graph(smile):
    edge_index=[[],[]]
    node_attr=[]
    edge_attr=[]
    mol=Chem.MolFromSmiles(smile)
    if mol is not None:
        mol=Chem.AddHs(mol)
        Chem.SanitizeMol(mol, Chem.SanitizeFlags.SANITIZE_KEKULIZE)
        atoms=mol.GetAtoms()
        for atom in atoms:
            node_attr.append(get_atom_features(atom))
            bonds=atom.GetBonds()
            atomid = atom.GetIdx()
            for bond in bonds:
                bids=bond.GetBeginAtomIdx()
                eids=bond.GetEndAtomIdx()
                if bids==atomid:
                    edge_index[0].append(bids)
                    edge_index[1].append(eids)
                elif eids==atomid:
                    edge_index[0].append(eids)
                    edge_index[1].append(bids)
                else:
                    AssertionError('edge not in atoms')
                edge_attr.append(get_bond_features(bond))
        return torch.FloatTensor(node_attr),torch.LongTensor(edge_index),torch.FloatTensor(edge_attr)
    else:
        return None


class SpectraDataset(Dataset):
    def __init__(self,
                 smiles,
                 spectra,
                 device=torch.device('cuda'),
                 spectra_process_function=None,
                 **kwargs):
        super(SpectraDataset,self).__init__(**kwargs)
        assert len(smiles)==len(spectra),'The number of spectra and smiles must match'
        self.device=device
        self.smiles=smiles
        self.spectra=spectra
        self.func=spectra_process_function
        
    def len(self):
        return len(self.smiles)

    def get(self,idx):

        SMILES=self.smiles[idx]
        node_attr,edge_index,edge_attr=smile2graph(SMILES)

        if self.func is not None:
            spectra=self.func(torch.tensor(self.spectra[idx],dtype=torch.float32))
        else:
            spectra=torch.tensor(self.spectra[idx],dtype=torch.float32)

        data=Data(x=node_attr,
                  SMILES=SMILES,
                  edge_index=edge_index,
                  edge_attr=edge_attr,
                  spectra=spectra.reshape(1,-1))

        return data.to(device=self.device)


class Linear(nn.Linear):
    def __init__(
        self,
        d_in: int,
        d_out: int,
        bias: bool = True,
        init: str = "default",
    ):
        super(Linear, self).__init__(d_in, d_out, bias=bias)

        self.use_bias = bias

        if self.use_bias:
            with torch.no_grad():
                self.bias.fill_(0)

        if init == "default":
            self._trunc_normal_init(1.0)
        elif init == "relu":
            self._trunc_normal_init(2.0)
        elif init == "glorot":
            self._glorot_uniform_init()
        elif init == "gating":
            self._zero_init(self.use_bias)
        elif init == "normal":
            self._normal_init()
        elif init == "final":
            self._zero_init(False)
        elif init == "jax":
            self._jax_init()
        else:
            raise ValueError("Invalid init method.")

    def _trunc_normal_init(self, scale=1.0):
        # Constant from scipy.stats.truncnorm.std(a=-2, b=2, loc=0., scale=1.)
        TRUNCATED_NORMAL_STDDEV_FACTOR = 0.87962566103423978
        _, fan_in = self.weight.shape
        scale = scale / max(1, fan_in)
        std = (scale**0.5) / TRUNCATED_NORMAL_STDDEV_FACTOR
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std)

    def _glorot_uniform_init(self):
        nn.init.xavier_uniform_(self.weight, gain=1)

    def _zero_init(self, use_bias=True):
        with torch.no_grad():
            self.weight.fill_(0.0)
            if use_bias:
                with torch.no_grad():
                    self.bias.fill_(1.0)

    def _normal_init(self):
        torch.nn.init.kaiming_normal_(self.weight, nonlinearity="linear")

    def _jax_init(self):
        input_size = self.weight.shape[-1]
        std = math.sqrt(1 / input_size)
        nn.init.trunc_normal_(self.weight, std=std, a=-2.0 * std, b=2.0 * std)


class MLP(nn.Module):
    def __init__(self,size,act,bias=True,last_act=False,dropout=0.0,init='glorot'):
        super(MLP,self).__init__()
        assert len(size)>1,'Multilayer perceptrons must be larger than one layer'
        self.in_channels=size[0]
        self.out_channels=size[-1]
        mlp=[]
        for si in range(0,len(size)-1):

            l=Linear(size[si], size[si + 1], bias=bias, init= init)
            mlp.append(l)
            if last_act is False:
                if (si!=len(size)-2):

                    mlp.append(act)
                    mlp.append(nn.Dropout(p=dropout))
            else:
                mlp.append(act)
                mlp.append(nn.Dropout(p=dropout))
        self.mlp=nn.Sequential(*mlp)


    def forward(self,x):
        for f in self.mlp:
            x=f(x)
        return x

class GNNBlock(nn.Module):
    def __init__(self,num_features,act,normal=False,last_act=True,dropout=0.0,reduce='mean'):
        super(GNNBlock, self).__init__()
        self.normal=normal
        if normal:
            self.norm=nn.LayerNorm(normalized_shape=num_features)
            self.graph_norm=nn.LayerNorm(normalized_shape=num_features)
        self.mlp=MLP(size=(num_features,num_features,num_features),act=act,last_act=False)
        self.conv=GINEConv(nn=self.mlp,edge_dim=num_features,aggr=reduce)
        self.last_act=last_act
        self.act=act
        self.drop=nn.Dropout(p=dropout)

    def forward(self,node_feature,edge_index,edge_feature):
        out=self.conv(x=node_feature,edge_index=edge_index,edge_attr=edge_feature)
        if self.normal:
            out=self.norm(out)
            out=self.graph_norm(out)
        if self.last_act:
            out=self.act(out)
        return node_feature+self.drop(out)


class GNN_Model(nn.Module):
    def __init__(self,
                 num_atom_features,
                 num_bond_features,
                 num_hidden_features,
                 num_layers,
                 act,
                 output_size,
                 readout='mean',
                 readout_message='mean',
                 last_act=True,
                 normal=True,
                 dropout=0.0):
        super(GNN_Model, self).__init__()
        self.bond_emb_list=nn.ModuleList()
        self.angle_emb_list=nn.ModuleList()
        self.atom_bond_block_list = nn.ModuleList()

        self.init_atom_emb=MLP(size=(num_atom_features,num_hidden_features,num_hidden_features),
                                          act=act,bias=False,last_act=last_act)

        for _ in range(num_layers):

            self.bond_emb_list.append(MLP(size=(num_bond_features+(num_atom_features*2),
                                                num_hidden_features,
                                                num_hidden_features),
                                          act=act,bias=False,last_act=last_act))

            self.atom_bond_block_list.append(GNNBlock(num_features=num_hidden_features,act=act,
                                                      normal=normal,dropout=dropout,reduce=readout_message))

        self.output_mlp=MLP(size=(num_hidden_features,2*num_hidden_features,output_size),
                                          act=act,bias=True,last_act=False)
        self.readout=readout
        self.num_layer=num_layers

    def forward(self,
                node_attr,
                edge_attr,
                edge_index,
                batch=None):
        atom_hidden=self.init_atom_emb(node_attr)
        i,j=edge_index
        edge_attr_2=torch.concat(tensors=(node_attr[i],edge_attr,node_attr[j]),dim=1)
        for ids in range(self.num_layer):

            bond_hidden=self.bond_emb_list[ids](edge_attr_2)

            atom_hidden = self.atom_bond_block_list[ids](node_feature=atom_hidden, edge_index=edge_index,
                                                         edge_feature=bond_hidden)
        atom_output=self.output_mlp(atom_hidden)

        if batch is None:
            if self.readout=='sum':
                output=torch.sum(atom_output,dim=0)
            elif self.readout=='mean':
                output=torch.mean(atom_output,dim=0)
        else:
            output=scatter(atom_output,index=batch,dim=0,reduce=self.readout)

        return output

def spearman(out, targ):
    out=out.cpu().detach().numpy()
    targ=targ.cpu().detach().numpy()
    spearman=[spearmanr(out[i],targ[i]).correlation for i in range(out.shape[0])]
    return torch.mean(torch.tensor(spearman))


def R2(out,target):
    '''coefficient of determination,Square of Pearson's coefficient, used to assess regression accuracy'''
    mean=torch.mean(target)
    SSE=torch.sum((out-target)**2)
    SST=torch.sum((mean-target)**2)
    return 1-(SSE/SST)


def l2loss(out,target):
    '''Mean Square Error(MSE)'''
    diff = out-target
    return torch.mean(diff ** 2)


def l1loss(out,target):
    '''Mean Absolute Error(MAE)'''
    return torch.mean(torch.abs(out-target))


def rmse(out, target):
    '''Root Mean Square Eroor(rmse) (also known as RMSD)'''
    return torch.sqrt(torch.mean((out - target) ** 2))


def sid(model_spectra,target_spectra,torch_device=torch.device('cpu'),threshold=1e-5):
    nan_mask=torch.isnan(target_spectra)+torch.isnan(model_spectra)
    nan_mask=nan_mask.to(device=torch_device)
    model_spectra[model_spectra < threshold] = threshold
    if not isinstance(target_spectra,torch.Tensor):
        target_spectra = torch.tensor(target_spectra)
    target_spectra = target_spectra.to(torch_device)
    model_spectra=model_spectra.to(torch_device)
    target_spectra[nan_mask]=threshold
    model_spectra[nan_mask]=threshold
    loss = torch.mul(torch.log(torch.div(model_spectra,target_spectra)),model_spectra) \
        + torch.mul(torch.log(torch.div(target_spectra,model_spectra)),target_spectra)
    loss = torch.sum(loss,axis=1)
    return loss.mean()

def combine_loss(out,target, coff=0.001):
    return (coff*sid(out,target))+rmse(out,target)

def sis(out,target):
    loss=sid(out,target)
    return 1/(1+0.001*loss)

model=GNN_Model(num_atom_features=99,
                num_bond_features=24,
                num_hidden_features=256,
                num_layers=3,
                act=nn.SiLU(),
                output_size=3000,
               readout_message='mean',
               normal=True)
        
print(model)

smiles_data=(pd.read_csv(r"5624_SMILES.csv",header=None)).to_numpy()[:,0]
spectra_data=(pd.read_csv(r"5624_ir.csv",header=None)).to_numpy()[:,:]
spectra_data=spectra_data/(spectra_data.max(axis=-1)[:,None])
spectra_data[spectra_data<1e-5]=1e-5
assert len(smiles_data)==len(spectra_data)
lent=len(smiles_data)
dataset_index=np.linspace(0,lent-1,lent,dtype='int')
part_train = dataset_index % 10 < 8
part_val = dataset_index % 10 == 8
train_dataset=SpectraDataset(smiles=smiles_data[part_train],spectra=spectra_data[part_train])
val_dataset=SpectraDataset(smiles=smiles_data[part_val],spectra=spectra_data[part_val])
print(train_dataset[0])
bathes=64
trainloader=DataLoader(train_dataset,batch_size=bathes,shuffle=True)
valloader=DataLoader(val_dataset,batch_size=bathes,shuffle=True)

class Trainer:
    def __init__(self,model,train_loader,val_loader=None,loss_function=l2loss,acc_function=R2,device=torch.device('cuda'),
                 optimizer='Adam_amsgrad',lr=5e-4,weight_decay=0):
        self.opt_type=optimizer
        self.device=device
        self.model=model.to(device)
        self.train_data=train_loader
        self.val_data=val_loader
        self.device=device
        self.opts={'AdamW':torch.optim.AdamW(self.model.parameters(),lr=lr,amsgrad=False,weight_decay=weight_decay),
              'AdamW_amsgrad':torch.optim.AdamW(self.model.parameters(),lr=lr,amsgrad=True,weight_decay=weight_decay),
              'Adam':torch.optim.Adam(self.model.parameters(),lr=lr,amsgrad=False,weight_decay=weight_decay),
              'Adam_amsgrad':torch.optim.Adam(self.model.parameters(),lr=lr,amsgrad=True,weight_decay=weight_decay),
              'Adadelta':torch.optim.Adadelta(self.model.parameters(),lr=lr,weight_decay=weight_decay),
              'RMSprop':torch.optim.RMSprop(self.model.parameters(),lr=lr,weight_decay=weight_decay),
              'SGD':torch.optim.SGD(self.model.parameters(),lr=lr,weight_decay=weight_decay)
        }
        self.optimizer=self.opts[self.opt_type]
        self.loss_function=loss_function
        self.acc_function=acc_function
        self.step=-1
    def train(self,num_train,targ,stop_loss=1e-8, val_per_train=50, print_per_epoch=10):
        self.model.train()
        len_train=len(self.train_data)
        for i in range(num_train):
            val_datas=iter(self.val_data)
            for j,batch in enumerate(self.train_data):
                self.step=self.step+1
                torch.cuda.empty_cache()
                self.optimizer.zero_grad()
                #print(batch.x,batch.batch,batch.edge_attr,batch.edge_index)
                #print(batch.z)
                target = batch[targ].to(self.device)
                target=target.reshape(target.shape[0],-1)
                out = self.model(node_attr=batch.x.to(self.device),edge_index=batch.edge_index.to(self.device),
                                 edge_attr=batch.edge_attr.to(self.device),batch=batch.batch.to(self.device)).reshape(target.shape)
                loss = self.loss_function(out,target)
                loss.backward()
                self.optimizer.step()
                if (self.step%val_per_train==0) and (self.val_data is not None):
                    val_batch = next(val_datas)
                    val_target=val_batch[targ].to(self.device)
                    val_target=val_target.reshape(val_target.shape[0],-1)

                    val_out = self.model(node_attr=val_batch.x.to(self.device),edge_index=val_batch.edge_index.to(self.device),
                                                   edge_attr=val_batch.edge_attr.to(self.device),batch=val_batch.batch.to(self.device)).reshape(val_target.shape)
                    val_loss = self.loss_function(val_out, val_target).item()
                    val_mae=l1loss(val_out, val_target).item()
                    val_acc=self.acc_function(val_out, val_target).item()
                    if self.step % print_per_epoch==0:
                        print('Epoch[{}/{}],loss:{:.8f},val_loss:{:.8f},val_mae:{:.8f},val_acc:{:.8f}'
                              .format(self.step,num_train*len_train,loss.item(),val_loss,val_mae,val_acc))

                    assert (loss > stop_loss) or (val_loss > stop_loss),'Training and prediction Loss is less' \
                                                                        ' than cut-off Loss, so training stops'
                elif (self.step % print_per_epoch == 0) and (self.step%val_per_train!=0):
                    print('Epoch[{}/{}],loss:{:.8f}'.format(self.step,num_train*len_train, loss.item()))
                    
    def load_state_and_optimizer(self,state_path=None,optimizer_path=None):
        if state_path is not None:
            state_dict=torch.load(state_path)
            self.model.load_state_dict(state_dict)
        if optimizer_path is not None:
            self.optimizer=torch.load(optimizer_path)

    def save_param(self,path):
        torch.save(self.model.state_dict(),path)

    def save_model(self,path):
        torch.save(self.model, path)

    def save_opt(self,path):
        torch.save(self.optimizer,path)

    def params(self):
        return self.model.state_dict()

trainer=Trainer(model,train_loader=trainloader,val_loader=valloader,loss_function=combine_loss,acc_function=sis,lr=1e-4,weight_decay=0,optimizer='AdamW')
trainer.train(num_train=1000,targ='spectra',val_per_train=150,print_per_epoch=30)
trainer.save_param('GNN.pth')
