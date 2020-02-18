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
