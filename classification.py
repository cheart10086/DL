import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

n_data=torch.ones(100,2)
x0=torch.normal(2*n_data,1)
y0=torch.zeros(100)
x1=torch.normal(-2*n_data,1)
y1=torch.ones(100)
x=torch.cat((x0,x1),0).type(torch.FloatTensor)
y=torch.cat((y0,y1),).type(torch.LongTensor)

class Net(torch.nn.Module):
    def __init__(self,n_features,n_hidden,n_output):
        super(Net,self).__init__()
        self.hidden=torch.nn.Linear(n_features,n_hidden)
        self.out= torch.nn.Linear(n_hidden,n_output)
    def forward(self,x):
        x=F.relu(self.hidden(x))
        x=self.out(x)
        return x

net=Net(n_features=2,n_hidden=10,n_output=2)
net1=torch.nn.Sequential(
    torch.nn.Linear(2,10),
    torch.nn.ReLU(),
    torch.nn.Linear(10,2)
)

print(net)
print(net1)


optimizer=torch.optim.SGD(net.parameters(),lr=2)
loss_func=torch.nn.CrossEntropyLoss()

plt.ion()

for t in range(10):
    out=net(x)

    loss=loss_func(out,y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if t%2==0:
        plt.cla()
        prediction=torch.max(out,1)[1]
        pred_y=y.data.numpy()
        target_y=y.data.numpy()
        plt.scatter(x.data.numpy()[:,0],x.data.numpy()[:,1],c=pred_y,s=100,lw=0,cmap='RdYlGn')
        accuray=float((pred_y==target_y).astype(int).sum())/float(target_y.size)
        plt.text(1.5,-4,'Accuacy=%.2f'%accuray,fontdict={'size':20,'color':'red'})
        plt.pause(0.1)

plt.ioff()
plt.show()
