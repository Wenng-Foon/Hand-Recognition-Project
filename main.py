import argparse
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader

from model import Attention_ResNet50
from train import train, resume, evaluate
from recognition_dataloader import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-id", "--exp_id", type=str, default='res0')
    parser.add_argument("-m", "--mode", choices=["normal", "se", "cbam" ], default="normal", help="To train a normal resnet or with SE/CBAM")
    parser.add_argument("-b","--batch_size", type=int, default=16, help='number of training epochs')
    parser.add_argument("-e","--epochs", type=int, default=50, help='number of training epochs')
    parser.add_argument("-lr", "--learning_rate", type=float, default=1e-3, help='learning rate')
    
    parser.add_argument("-pre", "--pretrain", type=int, default=1, help='specify if pretraining is needed')
    parser.add_argument("-red", "--reduction_ratio", type=int, default=16, help='reduction ratio for SE module')
    parser.add_argument("-pool", "--pooling", choices=["avg", "max", "both" ], default="avg", help='specify pooling used by SE module')

    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--resume', type=int, default=0, help='resume the trained model')
    parser.add_argument('--test', type=int, default=0, help='test with trained model')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # dataloaders
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(32),
        transforms.RandomRotation(20),
        transforms.Resize((180, 160)),
        transforms.RandomAffine(20, translate=(0.2, 0.2)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


    trainvalset = RecognitionImageFolder(root='./NUS', subdir='RecognitionData', transform=transform)

    trainset, testset = torch.utils.data.random_split(trainvalset, [2250, 500])

    trainloader = DataLoader(trainset, batch_size=args.batch_size,
                                              shuffle=True, num_workers=2)
    testloader = DataLoader(testset, batch_size=args.batch_size,
                                              shuffle=False, num_workers=2)
    dataloaders = (trainloader, testloader)

    classes = ('a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j')

    # network
    if args.pretrain == "0":
        pretrained = False
    else:
        pretrained = True

    if args.mode == "cbam":
        model = Attention_ResNet50(pretrained=pretrained, reduction_ratio=args.reduction_ratio, pool_type=args.pooling, use_Spatial= True).to(device)
    elif args.mode == "se":
        model = Attention_ResNet50(pretrained=pretrained, reduction_ratio=args.reduction_ratio, pool_type=args.pooling, use_Spatial= False).to(device)
    else:
        model = models.resnet50(pretrained=pretrained)
        model.fc = nn.Linear(2048,10)
        model = model.to(device)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9,0.999))

    # resume the trained model
    if args.resume:
        model, optimizer = resume(args, model, optimizer)

    if args.test == 1: # test mode, resume the trained model and test
        testing_accuracy = evaluate(args, model, testloader)
        print('testing finished, accuracy: {:.3f}'.format(testing_accuracy))
    else: # train mode, train the network from scratch
        train(args, model, optimizer, dataloaders)
        print('training finished')
