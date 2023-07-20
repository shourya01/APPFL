from .cnn import CNN
from .resnet import resnet18
from .resnet_fda import ResnetMultiTaskNet

def get_model(args):
    ## User-defined model
    if args.model == "CNN":
        model = CNN(args.num_channel, args.num_classes, args.num_pixel)
    if args.model == "resnet18":
        model = resnet18(num_channel=args.num_channel, num_classes=args.num_classes, pretrained=args.pretrained)   
    # if args.model == "resnet_fda":
    #     model = ResnetMultiTaskNet(pretrained=args.pretrained, frozen_feature_layers=args.frozen_feature_layers,
    #                                 resnet=args.resnet, hidden_size=args.hidden_size, num_classes=args.resnet_classes)
    #     # model = ResnetMultiTaskNet(pretrained=True, frozen_feature_layers=False, 
    #     #                            resnet='resnet152', hidden_size=512, num_classes=[2,3,2,10])
    return model
