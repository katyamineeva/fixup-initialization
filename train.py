def train_epoch(model, optimizer, train_loader):
    loss_log, acc_log = [], []
    model.train()
    for _, (x_batch, y_batch) in zip(trange(len(train_loader)), train_loader):
        data = x_batch.cuda()
        target = y_batch.cuda()
        y_batch = y_batch.cuda()
        optimizer.zero_grad()
        output = model(data).cuda()   
        
        pred = torch.max(output, 1)[1].cuda()
        acc = torch.eq(pred, y_batch).float().mean()
        acc_log.append(acc)
        
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        loss = loss.item()
        loss_log.append(loss)

        # clear_output()
        # print("\n Current loss: {:.2}".format(loss))
    return loss_log, acc_log

def test(model, test_loader):
    loss_log, acc_log = [], []
    model.eval()
    for x_batch, y_batch in test_loader:
        y_batch = y_batch.cuda()
        data = x_batch.cuda()
        target = y_batch.cuda()
        output = model(data).cuda()
        loss = F.cross_entropy(output, target)
        
        pred = torch.max(output, 1)[1].cuda()
        acc = float(torch.eq(pred, y_batch).float().mean())
        acc_log.append(acc)
        
        loss = loss.item()
        loss_log.append(loss)
    return loss_log, acc_log

def plot_history(train_history, val_history, title='loss'):
    plt.figure(figsize=(9, 6))
    plt.title('{}'.format(title))
    plt.plot(train_history, label='train', zorder=1)
    
    points = np.array(val_history)
    
    plt.scatter(points[:, 0], points[:, 1], marker='+', s=180, c='orange', label='val', zorder=2)
    plt.xlabel('train steps')
    
    plt.legend(loc='best')
    plt.grid()

    plt.show()
    
    
def train(model, opt, sch, n_epochs, train_loader, test_loader):
    train_log, train_acc_log = [], []
    val_log, val_acc_log = [], []
    filename = get_filename(model, train_loader, n_epochs)

    for epoch in range(n_epochs):
        train_loss, train_acc = train_epoch(model, opt, train_loader)

        val_loss, val_acc = test(model, test_loader)
        sch.step(val_loss[-1])

        train_log.extend(train_loss)
        train_acc_log.extend(train_acc)

        steps = len(train_log)
        val_log.append((steps, np.mean(val_loss)))
        val_acc_log.append((steps, np.mean(val_acc)))
        
        # i'm not adding this funcion in rep, but you might want to implement your own
        # save_logs(train_log, val_log, train_acc_log, val_acc_log, filename)
        
        clear_output()
        plot_history(train_log, val_log)    
        plot_history(train_acc_log, val_acc_log, title='accuracy')   
            
    print("Final error: {:.2%}".format(1 - val_acc_log[-1][1]))
    print("Logs have saves to " + filename + " file.")
    return train_log, val_log, train_acc_log, val_acc_log
