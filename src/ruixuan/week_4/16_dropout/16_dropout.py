from torch import nn
import torch


class MyNet(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size, num_classes)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        out = self.dropout(x)
        print("out is ", out)
        out = self.fc1(x)
        print("after fc is ", out)
        return out


input_size = 10
num_classes = 5
model = MyNet(input_size, num_classes)
x = torch.arange(0, 10).reshape(-1).float()
print("输入向量", x)
model.train()
print("训练模式下:", model(x))
model.eval()
print("测试模式下:", model(x))
