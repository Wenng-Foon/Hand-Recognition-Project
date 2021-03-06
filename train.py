import os
import time
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

criterion = torch.nn.CrossEntropyLoss()

def train(args, model, optimizer, dataloaders):
    trainloader, testloader = dataloaders

    best_testing_accuracy = 0.0

    # training
    for epoch in range(args.epochs):
        model.train()

        batch_time = time.time(); iter_time = time.time()
        for i, data in enumerate(trainloader):

            imgs, labels = data
            imgs, labels = imgs.to(device), labels.to(device)

            cls_scores = model(imgs)
            loss = criterion(cls_scores, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 50 == 0 and i != 0:
                print('epoch:{}, iter:{}, time:{:.2f}, loss:{:.5f}'.format(epoch, i,
                    time.time()-iter_time, loss.item()))
                iter_time = time.time()
        batch_time = time.time() - batch_time
        print('[epoch {} | time:{:.2f} | loss:{:.5f}]'.format(epoch, batch_time, loss.item()))
        print('-------------------------------------------------')

        if epoch % 1 == 0:
            testing_accuracy = evaluate(args, model, testloader)
            print('testing accuracy: {:.3f}'.format(testing_accuracy))

            if testing_accuracy > best_testing_accuracy:
                ### compare the previous best testing accuracy and the new testing accuracy
                ### save the model and the optimizer --------------------------------
                #
                torch.save({'epoch': epoch,
                            'accuracy' : testing_accuracy,
                            'loss' : loss.item(),
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict()
                            }, './{}_{}_checkpoint.pth'.format(args.exp_id, args.mode))
                            
                best_testing_accuracy = testing_accuracy
                #
                #
                ### -----------------------------------------------------------------
                print('new best model saved at epoch: {}'.format(epoch))
    print('-------------------------------------------------')
    print('best testing accuracy achieved: {:.3f}'.format(best_testing_accuracy))

def evaluate(args, model, testloader):
    total_count = torch.tensor([0.0],device=device); correct_count = torch.tensor([0.0],device=device)
    for i, data in enumerate(testloader):
        imgs, labels = data
        imgs, labels = imgs.to(device), labels.to(device)

        total_count += labels.size(0)

        with torch.no_grad():
            cls_scores = model(imgs)

            predict = torch.argmax(cls_scores, dim=1)
            correct_count += (predict == labels).sum()
    testing_accuracy = correct_count / total_count
    return testing_accuracy.item()


def resume(args, model, optimizer):
    checkpoint_path = './{}_{}_checkpoint.pth'.format(args.exp_id, args.mode)
    assert os.path.exists(checkpoint_path), ('checkpoint do not exits for %s' % checkpoint_path)

    ### load the model and the optimizer --------------------------------

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    testing_accuracy = checkpoint['accuracy']
    loss = checkpoint['loss']

    print("resuming from epoch {}, testing_accuracy: {}, loss: {} ".format(epoch, testing_accuracy, loss))
    
    model.to(device)
    ### -----------------------------------------------------------------

    print('Resume completed for the model\n')

    return model, optimizer
