import os
import argparse, random
import math
import numpy as np
import torch,json
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, datasets
import torch.optim.lr_scheduler as lr_scheduler
from my_dataset import MyDataSet
from all_models.swin_transformer_original import cswin4_tiny_v2_patch4_window7_224 as create_model
from utils import read_split_data, train_one_epoch, evaluate, collate_fn, plot_data_loader_image

from thop import profile
def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
seed_torch()




def calculate_gflops(model, input_size):
    input = torch.randn(input_size)
    flops, params = profile(model, inputs=(input,))
    gflops = flops / 1e9
    return gflops

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if os.path.exists("./all_weights/{}".format(args.model_name)) is False:
        os.makedirs("./all_weights/{}".format(args.model_name))

    tb_writer = SummaryWriter(log_dir="./log/{}".format(args.model_name))

    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(args.data_path)

    best_Acc = 0.0
    data_transform = {
        "train": transforms.Compose([

            transforms.RandomHorizontalFlip(),  # 以0.5的概率对图像进行水平翻转
            transforms.RandomVerticalFlip(),  # 以0.5的概率对图像进行垂直翻转
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    # 实例化训练数据集
    train_dataset = MyDataSet(images_path=train_images_path, images_class=train_images_label, transform=data_transform["train"])
    # 实例化验证数据集
    val_dataset = MyDataSet(images_path=val_images_path, images_class=val_images_label, transform=data_transform["val"])


    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=val_dataset.collate_fn)


    model = create_model(num_classes=args.num_classes)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("模型参数大小为：{}, 学习率为：{}, batch_size为：{}".format(n_parameters, args.lr, args.batch_size))
    input_tensor = (1, 3, 224, 224)
    gflops = calculate_gflops(model,input_tensor)

    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = torch.load(args.weights, map_location=device)["model"]
        # 删除有关分类类别的权重
        for k in list(weights_dict.keys()):
            if "head" in k:
                del weights_dict[k]
        print(model.load_state_dict(weights_dict, strict=False))
    
    if args.freeze_layers:
        for name, para in model.named_parameters():
            # 除head外，其他权重全部冻结
            if "head" not in name:
                para.requires_grad_(False)
            else:
                print("training {}".format(name))
    
    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(pg, lr=args.lr, weight_decay=0.05)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)
    
    
    
    for epoch in range(args.epochs):
        # train
        train_loss, train_acc = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch)
        scheduler.step()
    
        # validate
        val_loss, val_acc = evaluate(model=model,
                                     data_loader=val_loader,
                                     device=device,
                                     epoch=epoch)
    
        tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_acc, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)
    
        if epoch == 199:
            torch.save(model.state_dict(), "./all_weights/{}/model-{}.pth".format(args.model_name, epoch))
        if best_Acc < val_acc:
            best_Acc = val_acc
            torch.save(model.state_dict(), "./final_weights/{}_best.path".format(args.model_name))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lrf', type=float, default=1e-6)
    parser.add_argument('--model-name', type=str, default="cswin")
    # 数据集所在根目录
    # http://download.tensorflow.org/example_images/flower_photos.tgz
    parser.add_argument('--data-path', type=str,default="F:\dataset_large_re_aug_test_no_Aug_v4")
    # 预训练权重路径，如果不想载入就设置为空字符
    parser.add_argument('--weights', type=str, default='',
                        help='initial weights path')
    # 是否冻结权重
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()
    main(opt)
