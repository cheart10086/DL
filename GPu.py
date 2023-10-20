import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# 如果 CUDA 可用，则使用 GPU；否则使用 CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

x=torch.unsqueeze(torch.linspace(-1,1,100),dim=1).to(device)
#torch.unsqueeze函数用于增加张量维数，即当前在
#linspace用来均匀创建数值，结构为linspace(start,stop,num)
y=x.pow(2)+0.2*torch.rand(x.size()).to(device)

plt.scatter(x.cpu().numpy(),y.cpu().numpy())
plt.show()

class Net(torch.nn.Module):
  def __init__(self,n_features,n_hidden,n_output):
    super(Net, self).__init__()
    self.hidden=torch.nn.Linear(n_features,n_hidden)
    self.predict = torch.nn.Linear(n_hidden,n_output)

  def forward(self,x):
    x=F.relu(self.hidden(x))
    x=self.predict(x)
    return x

net=Net(1,10,1).to(device)
print(net)

optimizer=torch.optim.SGD(net.parameters(),lr=0.2)
loss_func=torch.nn.MSELoss()
plt.ion()
plt.show()

for t in range(10):
  prediction=net(x)

  loss=loss_func(prediction,y)

  optimizer.zero_grad()
  loss.backward()
  optimizer.step()

  if t % 5 == 0:
    plt.cla()
    plt.scatter(x.cpu().data.numpy(), y.cpu().data.numpy())
    plt.plot(x.cpu().data.numpy(), prediction.cpu().data.numpy(), 'r-', lw=5)
    plt.text(0.5, 0, 'loss=%.4f' % loss.cpu().data.numpy(), fontdict={'size': 20, 'color': 'red'})
    plt.pause(0.1)
