import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from torch.utils.data import DataLoader, Dataset,TensorDataset,ConcatDataset
from sklearn import preprocessing
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


path = 'schneider50k/'
'''
# 加载数据集文件
train_dataset = torch.load(path+"train_dataset.pt")
test_dataset = torch.load(path+"test_dataset.pt")
val_dataset = torch.load(path+"val_dataset.pt")
'''
train_dataset = torch.load(path+"train_filter_dataset.pt")
test_dataset = torch.load(path+"test_filter_dataset.pt")
val_dataset = torch.load(path+"val_filter_dataset.pt")

# 提取所有的y并组成一个array类型的列表
train_features = []
train_labels = []
test_features = []
test_labels = []
val_features = []
val_labels = []
for train_sample in train_dataset:
    train_feature,train_label = train_sample
    train_features.append(train_feature)
    train_labels.append(train_label)

for test_sample in test_dataset:
    test_feature,test_label = test_sample
    test_features.append(test_feature)
    test_labels.append(test_label)

for val_sample in val_dataset:
    val_feature,val_label = val_sample
    val_features.append(val_feature)
    val_labels.append(val_label)

label_num = preprocessing.LabelEncoder()
label_num.fit(train_labels+test_labels+val_labels)
templates_labels = list(label_num.classes_)
templates_nums = list(label_num.transform(templates_labels))
labels_nums = [None]*len(label_num.classes_)
for ind,template in zip(templates_nums,templates_labels):
    labels_nums[int(ind)] = template
with open(path+'labels_nums_filter.csv', "w") as file:
    file.write("\n".join(labels_nums))


train_labels_num = label_num.transform(train_labels)
test_labels_num = label_num.transform(test_labels)
val_labels_num = label_num.transform(val_labels)



# 定义神经网络模型
class Classifer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Classifer, self).__init__()
        self.fc1 = nn.Linear(input_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, output_dim)
        self.dropout = nn.Dropout(p=0.3)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc4(x)
        x = self.softmax(x)
        return x

class ZSLCrossEntropyLoss(nn.Module):
        def __init__(self, weight_decay=0.001):
            super(ZSLCrossEntropyLoss, self).__init__()
            self.loss_fn = nn.CrossEntropyLoss()
            self.weight_decay = weight_decay

        def forward(self, logits, targets, params=None):
            loss = self.loss_fn(logits, targets)

            # 计算L2范数
            l2_reg = 0
            if params is not None:
                for param in params:
                    l2_reg += torch.norm(param, p=2)**2
            else:
                for param in self.parameters():
                    l2_reg += torch.norm(param, p=2)**2

            # 加入L2正则化项
            loss += self.weight_decay * l2_reg

            return loss
        
# 定义数据集及其相关的参数
train_dataset = TensorDataset(torch.tensor(np.array(train_features,dtype=np.float32)),torch.tensor(np.array(train_labels_num)))
test_dataset = TensorDataset(torch.tensor(np.array(test_features,dtype=np.float32)),torch.tensor(np.array(test_labels_num)))
val_dataset = TensorDataset(torch.tensor(np.array(val_features,dtype=np.float32)),torch.tensor(np.array(val_labels_num)))
input_size = train_dataset[0][0].shape[0]
num_classes = len(label_num.classes_)
batch_size = 64
num_epochs = 1000
lr = 0.05

train_dataset = ConcatDataset([train_dataset, val_dataset])
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size)
#val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Classifer(input_size, num_classes).to(device)
loss_fn = ZSLCrossEntropyLoss()
loss_fn = loss_fn.to(device)
optimizer = optim.SGD(model.parameters(), lr=lr)




# 训练模型
loss_list = []
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, (features, labels) in enumerate(train_loader):
        # 将数据移动到GPU
        features = features.to(device)
        labels = labels.to(device)
        
        # 前向传播
        outputs = model(features)
        loss = loss_fn(outputs, labels.long(),params=model.parameters())
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * features.size(0)
        
        if (i+1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, i+1, len(train_dataset)//batch_size, loss.item()))
    
    epoch_loss = running_loss / len(train_dataset)
    loss_list.append(epoch_loss)

plt.plot(loss_list)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.show()

torch.save(model.state_dict(),path+'model_filter.pt')
# 在测试集上评估模型
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for features, labels in test_loader:
        # 将数据移动到GPU
        features = features.to(device)
        labels = labels.to(device)

        # 前向传播
        outputs = model(features)
        _, predicted = torch.max(outputs.data, dim=1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the test features: {} %'.format(100 * correct / total))


