def test(model, device, test_loader):
    model.eval()
#将模型设置为评估模式
    test_loss = 0
#计算损失
    correct = 0
#正确预测的数量
    with torch.no_grad():
#这个with不是很懂？？
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
#计算损失
            pred = output.max(1, keepdim=True)[1]
#这一句不是很懂？？
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
#计算平均损失
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

for epoch in range(1, EPOCHS + 1):
    train(model, DEVICE, train_loader, optimizer, epoch)
    test(model, DEVICE, test_loader)