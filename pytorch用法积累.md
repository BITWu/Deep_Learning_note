# pytorch 用法积累
[TOC]
### 初始化模型参数
```python
torch.nn.init
```
### 选择GPU or CPU
```python
device = torch.device('cuda:0' if torch.cuda.isavailable() else 'cpu' )
```
### 将 tensor 和 模型 加到 GPU 上
```python
device = torch.device('cuda:0')
net.to(device)
x.to(device)
```
### 改变 tensor 维度
```python
x.view(-1,2) # 将 tensor x 变为二维的，其中列的数目为2的 tensor，-1 代表第一个维度自适应调整
x.view(-1) # 只有一个-1默认将 x 变为一维数据 
```
### 对 tensor 在指定维度上求均值
```python
x.mean(dim=0, keepdim = True) 
# keepdim = True 时保留x的原来的维度，否则会压缩一个维度
```
### 自定义网络参数
```python
torch.nn.Parameter(tensor) # nn.Parameter 可以将一个 tensor 加入到参数列表中
```
### torchvision.transforms.ToTensor()
ToTensor()方法会将值映射到[0,1]区间，后面常跟着Normalize()方法进行归一化处理

### torchvison.datasets.ImageFolder

### 只训练网络某一部分参数
```python
def get_net(device):
    finetune_net = models.resnet34(pretrained=False)  # 预训练的resnet34网络
    finetune_net.load_state_dict(torch.load('/home/kesci/input/resnet347742/resnet34-333f7ec4.pth'))
    for param in finetune_net.parameters():  # 冻结参数
        param.requires_grad = False # 将特征提取部分的参数设为不可求导，则不会计算这部分参数的梯度，不会对其进行训练
    # 原finetune_net.fc是一个输入单元数为512，输出单元数为1000的全连接层
    # 替换掉原finetune_net.fc，新finetuen_net.fc中的模型参数会记录梯度
    finetune_net.fc = nn.Sequential(
        nn.Linear(in_features=512, out_features=256),
        nn.ReLU(),
        nn.Linear(in_features=256, out_features=120)  # 120是输出类别数
    )
    return finetune_ne
```