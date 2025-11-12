import torch
import torch.nn as nn
from torch.autograd import Variable
import sys, argparse, os, copy, glob, datetime
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve, roc_auc_score, classification_report  # classification_report,auc
from numpy import interp
from itertools import cycle
# import BRTY2.model_DMR.networks as mil
import TRANS_NET as MIL_model
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
import random

from pdb import set_trace as st

def set_seed(seed=42):
    random.seed(seed)                          # Python随机性
    np.random.seed(seed)                       # Numpy随机性
    torch.manual_seed(seed)                    # CPU随机性
    torch.cuda.manual_seed(seed)               # 当前GPU随机性
    torch.cuda.manual_seed_all(seed)           # 多GPU随机性
    torch.backends.cudnn.deterministic = True  # 让CUDNN使用确定性算法
    torch.backends.cudnn.benchmark = False     # 关闭自动调优以避免非确定性


def all_label(txt_path = r"/home2/lsy/BRCA_data/BRCA_ONLY_CLASS/datatxt/TCGA_YY_LABEL.txt"):
    IHC_LABEL = ['Negative', 'Positive']
    txt = open(txt_path,"r")
    svs_basename = []
    er =[]
    pr=[]
    her2=[]
    for line_path in txt:
        line_path.rstrip("\n")
        line_path.lstrip("\n")
        path = line_path.split()
        if "TCGA" in path[0]:
            svs_basename.append(str(os.path.basename(path[0])).split(".h5")[0])
            er.append(IHC_LABEL.index(path[2]))
            pr.append(IHC_LABEL.index(path[3]))
            her2.append(IHC_LABEL.index(path[4]))
        else:
            svs_basename.append(str(os.path.basename(path[0]+" "+path[1]+" "+path[2]+" "+path[3])).split(".h5")[0])
            er.append(IHC_LABEL.index(path[5]))
            pr.append(IHC_LABEL.index(path[6]))
            her2.append(IHC_LABEL.index(path[7]))

    return [svs_basename, er, pr, her2]


'''
def roc_show(Y_valid, Y_pred, nb_classes):
    Y_valid = label_binarize(Y_valid, classes=[i for i in range(nb_classes)])
    Y_pred = label_binarize(Y_pred, classes=[i for i in range(nb_classes)])
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(nb_classes):
        fpr[i], tpr[i], _ = roc_curve(Y_valid[:, i], Y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(Y_valid.ravel(), Y_pred.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Compute macro-average ROC curve and ROC area
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(nb_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(nb_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= nb_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    lw = 2
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red'])
    CLASS_LABEL = ['Luminal A', 'Luminal B', 'Her2', 'Basal']
    for i, color in zip(range(nb_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of {0} (area = {1:0.2f})'
                       ''.format(CLASS_LABEL[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.show()
'''

def get_bag_feats(csv_file_df, args, all_svs_label):

    feats_csv_path = csv_file_df.iloc[0]
    df = pd.read_csv(feats_csv_path)
    # feats = shuffle(df).reset_index(drop=True)
    feats = df.reset_index(drop=True)
    feats = feats.to_numpy()
    label = np.zeros(args.num_classes)
    if args.class_name == 'IHC':
        CLASS_LABEL = ['LumA', 'LumB', 'Her2', 'Basal']
    else:
        CLASS_LABEL = ['Negative', 'Positive']
    if args.num_classes == 1:
        label[0] = CLASS_LABEL.index(csv_file_df.iloc[1])
    else:
        if CLASS_LABEL.index(csv_file_df.iloc[1]) <= (len(label) - 1):
            label[CLASS_LABEL.index(csv_file_df.iloc[1])] = 1

    svs_basename = str(os.path.basename(feats_csv_path)).split('.csv')[0]
    id = all_svs_label[0].index(svs_basename)
    if all_svs_label[1][id]== 0:
        er_label =[1,0]
    else:
        er_label=[0,1]

    if all_svs_label[2][id]==0:
        pr_label=[1,0]
    else:
        pr_label=[0,1]

    if all_svs_label[3][id]==0:
        her2_label =[1,0]
    else:
        her2_label =[0,1]

    return label, feats, er_label, pr_label, her2_label

def train(train_df, milnet, criterion, optimizer, args, all_svs_label):
    milnet.train()
    total_loss = 0
    Tensor = torch.cuda.FloatTensor

    K = 10
    print(f" ============ K={K} ============ ")

    for i in range(len(train_df)):
        optimizer.zero_grad()
        label, feats, er_label, pr_label, her2_label = get_bag_feats(train_df.iloc[i], args, all_svs_label)
        feats = dropout_patches(feats, args.dropout_patch)
        bag_label = Variable(Tensor([label]))
        er_label = Variable(Tensor([er_label]))
        pr_label = Variable(Tensor([pr_label]))
        her2_label = Variable(Tensor([her2_label]))
        bag_feats = Variable(Tensor([feats]))
        bag_feats = bag_feats.view(-1, args.feats_size)
        C, InstancePred = milnet(bag_feats)
        max_prediction, _ = torch.max(InstancePred[3], 0)
        er_max_prediction, _ = torch.max(InstancePred[0], 0)
        pr_max_prediction, _ = torch.max(InstancePred[1], 0)
        her2_max_prediction, _ = torch.max(InstancePred[2], 0)

        # ---- 取Top-K平均代替max ----
        def topk_mean(pred, k=K):
            # pred: [num_patches, num_classes]
            values, _ = torch.topk(pred, k=min(k, pred.shape[0]), dim=0)
            return values.mean(dim=0)

        IHC_topk = topk_mean(InstancePred[3])
        ER_topk  = topk_mean(InstancePred[0])
        PR_topk  = topk_mean(InstancePred[1])
        HER2_topk = topk_mean(InstancePred[2])

        bag_loss = criterion(C.view(1, -1), bag_label.view(1, -1))
        # max_loss = criterion(max_prediction.view(1, -1), bag_label.view(1, -1)) + criterion(er_max_prediction.view(1, -1), er_label.view(1, -1)) + criterion(pr_max_prediction.view(1, -1), pr_label.view(1, -1))+ criterion(her2_max_prediction.view(1, -1), her2_label.view(1, -1))
        topk_loss = (
            criterion(IHC_topk.view(1, -1), bag_label.view(1, -1)) +
            criterion(ER_topk.view(1, -1), er_label.view(1, -1)) +
            criterion(PR_topk.view(1, -1), pr_label.view(1, -1)) +
            criterion(HER2_topk.view(1, -1), her2_label.view(1, -1))
        )

        loss = 0.5 * bag_loss + 0.5 * topk_loss#max_loss
        
        loss.backward()
        optimizer.step()
        total_loss = total_loss + loss.item()
        sys.stdout.write('\r Training bag [%d/%d] bag loss: %.4f' % (i, len(train_df), loss.item()))
    return total_loss / len(train_df)


def dropout_patches(feats, p):
    idx = np.random.choice(np.arange(feats.shape[0]), int(feats.shape[0] * (1 - p)), replace=False)
    sampled_feats = np.take(feats, idx, axis=0)
    pad_idx = np.random.choice(np.arange(sampled_feats.shape[0]), int(feats.shape[0] * p), replace=False)
    pad_feats = np.take(sampled_feats, pad_idx, axis=0)
    sampled_feats = np.concatenate((sampled_feats, pad_feats), axis=0)
    return sampled_feats


def test(test_df, milnet, criterion, optimizer, args, all_svs_label):
    milnet.eval()
    csvs = shuffle(test_df).reset_index(drop=True)
    total_loss = 0
    test_labels = []
    test_predictions = []
    Tensor = torch.cuda.FloatTensor

    K = 10

    with torch.no_grad():
        for i in range(len(test_df)):
            label, feats, er_label, pr_label, her2_label = get_bag_feats(test_df.iloc[i], args, all_svs_label)
            bag_label = Variable(Tensor([label]))
            er_label = Variable(Tensor([er_label]))
            pr_label = Variable(Tensor([pr_label]))
            her2_label = Variable(Tensor([her2_label]))
            bag_feats = Variable(Tensor([feats]))
            bag_feats = bag_feats.view(-1, args.feats_size)
            C, InstancePred = milnet(bag_feats)
            max_prediction, _ = torch.max(InstancePred[3], 0)
            er_max_prediction, _ = torch.max(InstancePred[0], 0)
            pr_max_prediction, _ = torch.max(InstancePred[1], 0)
            her2_max_prediction, _ = torch.max(InstancePred[2], 0)

            # ---- 取Top-K平均代替max ----
            def topk_mean(pred, k=K):
                # pred: [num_patches, num_classes]
                values, _ = torch.topk(pred, k=min(k, pred.shape[0]), dim=0)
                return values.mean(dim=0)

            IHC_topk = topk_mean(InstancePred[3])
            ER_topk  = topk_mean(InstancePred[0])
            PR_topk  = topk_mean(InstancePred[1])
            HER2_topk = topk_mean(InstancePred[2])

            bag_loss = criterion(C.view(1, -1), bag_label.view(1, -1))
            # max_loss = criterion(max_prediction.view(1, -1), bag_label.view(1, -1)) + criterion(er_max_prediction.view(1, -1), er_label.view(1, -1)) + criterion(pr_max_prediction.view(1, -1), pr_label.view(1, -1))+ criterion(her2_max_prediction.view(1, -1), her2_label.view(1, -1))
            topk_loss = (
                criterion(IHC_topk.view(1, -1), bag_label.view(1, -1)) +
                criterion(ER_topk.view(1, -1), er_label.view(1, -1)) +
                criterion(PR_topk.view(1, -1), pr_label.view(1, -1)) +
                criterion(HER2_topk.view(1, -1), her2_label.view(1, -1))
            )
            loss = 0.5 * bag_loss + 0.5 * topk_loss#max_loss

            total_loss = total_loss + loss.item()
            sys.stdout.write('\r Testing bag [%d/%d] bag loss: %.4f' % (i, len(test_df), loss.item()))
            test_labels.extend([label])
            if args.average:
                test_predictions.extend([(0.5 * torch.sigmoid(IHC_topk) + 0.5 * torch.sigmoid(
                    C)).squeeze().cpu().numpy()])
            else:
                test_predictions.extend([(0.0 * torch.sigmoid(max_prediction) + 1.0 * torch.sigmoid(
                    C)).squeeze().cpu().numpy()])
    test_labels = np.array(test_labels)
    test_predictions = np.array(test_predictions)
    if args.class_name == 'IHC':
        class_name = ['LumA', 'LumB', 'Her2', 'Basal']
    else:
        class_name = ['Negative', 'Positive']
    auc_value, _, thresholds_optimal = multi_label_roc(test_labels, test_predictions, args.num_classes, class_name)
    if args.num_classes == 1:
        class_prediction_bag = copy.deepcopy(test_predictions)
        class_prediction_bag[test_predictions >= thresholds_optimal[0]] = 1
        class_prediction_bag[test_predictions < thresholds_optimal[0]] = 0
        test_predictions = class_prediction_bag
        test_labels = np.squeeze(test_labels)
    else:
        for i in range(args.num_classes):
            class_prediction_bag = copy.deepcopy(test_predictions[:, i])
            class_prediction_bag[test_predictions[:, i] >= thresholds_optimal[i]] = 1
            class_prediction_bag[test_predictions[:, i] < thresholds_optimal[i]] = 0
            test_predictions[:, i] = class_prediction_bag
    bag_score = 0
    for i in range(0, len(test_df)):
        bag_score = np.array_equal(test_labels[i], test_predictions[i]) + bag_score
    avg_score = bag_score / len(test_df)

    return total_loss / len(test_df), avg_score, auc_value, thresholds_optimal

def multi_label_roc(labels, predictions, num_classes, class_name):
    fprs = []
    tprs = []
    thresholds = []
    thresholds_optimal = []
    aucs = []
    if len(predictions.shape) == 1:
        predictions = predictions[:, None]

    _, ll = torch.max(torch.tensor(labels), 1)
    _, pp = torch.max(torch.tensor(predictions), 1)
    # print(classification_report(ll, pp, target_names=class_name))

    for c in range(0, num_classes):
        label = labels[:, c]
        prediction = predictions[:, c]
        fpr, tpr, threshold = roc_curve(label, prediction, pos_label=1)
        fpr_optimal, tpr_optimal, threshold_optimal = optimal_thresh(fpr, tpr, threshold)
        c_auc = roc_auc_score(label, prediction)
        aucs.append(c_auc)
        thresholds.append(threshold)
        thresholds_optimal.append(threshold_optimal)
    return aucs, thresholds, thresholds_optimal

def optimal_thresh(fpr, tpr, thresholds, p=0):
    loss = (fpr - tpr) - p * tpr / (fpr + tpr + 1)
    idx = np.argmin(loss, axis=0)
    return fpr[idx], tpr[idx], thresholds[idx]

def get_config():
    """Returns the ViT-B/16 configuration."""
    config = argparse.ArgumentParser(description='patch features learned by SimCLR')
    config.add_argument('--hidden_size', default=512, type=int)
    config.add_argument('--mlp_dim', default=1024, type=int)
    config.add_argument('--num_heads', default=2, type=int)
    config.add_argument('--num_layers', default=2, type=int)
    config.add_argument('--attention_dropout_rate', default=0.0, type=float)
    config.add_argument('--dropout_rate', default=0.1, type=float)
    config.add_argument('--classifier', default='token', type=str)
    config.add_argument('--representation_size', default=None, type=int)
    args = config.parse_args()
    return args

def main():
    parser = argparse.ArgumentParser(description='Train DSMIL on 20x patch features learned by SimCLR')
    parser.add_argument('--num_classes', default=4, type=int, help='Number of output classes [2]')
    parser.add_argument('--feats_size', default=512, type=int, help='Dimension of the feature size [512]')
    parser.add_argument('--lr', default=0.0002, type=float, help='Initial learning rate [0.0002]')
    parser.add_argument('--num_epochs', default=400, type=int, help='Number of total training epochs')
    parser.add_argument('--gpu_index', type=int, nargs='+', default=(0, 1), help='GPU ID(s) [0]')
    parser.add_argument('--weight_decay', default=5e-3, type=float, help='Weight decay [5e-3]')
    parser.add_argument('--split', default=0.2, type=float, help='Training/Validation split [0.2]')
    parser.add_argument('--dropout_patch', default=0, type=float, help='Patch dropout rate [0]')
    parser.add_argument('--dropout_node', default=0, type=float, help='Bag classifier dropout rate [0]')
    parser.add_argument('--non_linearity', default=1, type=float, help='Additional nonlinear operation [0]')
    parser.add_argument('--average', type=bool, default=True,help='Average the score of max-pooling and bag aggregating')
    parser.add_argument('--class_name', default='IHC', type=str)
    parser.add_argument('--seed', default=42, type=int, help='random seed')  # 添加参数

    args = parser.parse_args()
    
    set_seed(args.seed)


    gpu_ids = tuple(args.gpu_index)
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in gpu_ids)
    all_svs_label = all_label(txt_path=r"/home2/lsy/BRCA_data/BRCA_ONLY_CLASS/datatxt/TCGA_YY_LABEL.txt")
    config = get_config()
    ER_classifier = MIL_model.InstanceFC(in_size=args.feats_size, out_size=2).cuda()
    PR_classifier = MIL_model.InstanceFC(in_size=args.feats_size, out_size=2).cuda()
    Her2_classifier = MIL_model.InstanceFC(in_size=args.feats_size, out_size=2).cuda()
    IHC_classifier = MIL_model.InstanceFC(in_size=args.feats_size, out_size=4).cuda()
    FClist = [ER_classifier, PR_classifier, Her2_classifier, IHC_classifier]
    milnet = MIL_model.BagMIL(config, FClist, input_size=args.feats_size, output_class=4).cuda()

    for name, param in milnet.named_parameters():
        if param.requires_grad:
            print(f'*** requires_grad: {name}') 

    # state_dict_weights = torch.load('/home2/lsy/BRCA_data/BRCA_ONLY_CLASS/src/weights/10202025/top10.pth')
    # milnet.load_state_dict(state_dict_weights, strict=False)
    '''
    try:
        milnet.load_state_dict(state_dict_weights, strict=False)
    except:
        del state_dict_weights['b_classifier.v.1.weight']
        del state_dict_weights['b_classifier.v.1.bias']
        milnet.load_state_dict(state_dict_weights, strict=False)
    '''

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(milnet.parameters(), lr=args.lr, betas=(0.5, 0.9), weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epochs, 0.000005)
    bags_csv = os.path.join('/home4/lsy/BRCA_data/patch_features/BRCA.csv')

    bags_path = pd.read_csv(bags_csv)
    train_path = bags_path.iloc[0:int(len(bags_path) * (1 - args.split-0.1)), :]  # (0, 0.7)
    val_path = bags_path.iloc[int(len(bags_path) * (1 - args.split-0.1)):int(len(bags_path) * (1 - args.split)), :]  # (0.7, 0.8)
    test_path = bags_path.iloc[int(len(bags_path) * (1 - args.split)):, :]  # (0.8, 1)
    best_score = 0
    save_path = os.path.join('weights', datetime.date.today().strftime("%m%d%Y"))
    os.makedirs(save_path, exist_ok=True)
    run = len(glob.glob(os.path.join(save_path, '*.pth')))

    for epoch in range(1, args.num_epochs):
        train_path = shuffle(train_path).reset_index(drop=True)
        test_path = shuffle(test_path).reset_index(drop=True)
        train_loss_bag = train(train_path, milnet, criterion, optimizer, args,all_svs_label)  # iterate all bags
        test_loss_bag, avg_score, aucs, thresholds_optimal = test(val_path, milnet, criterion, optimizer, args, all_svs_label)
        print('\r Epoch [%d/%d] train loss: %.4f test loss: %.4f, average score: %.4f, AUC: ' %
                  (epoch, args.num_epochs, train_loss_bag, test_loss_bag, avg_score) + '|'.join(
                'class-{}>>{}'.format(*k) for k in enumerate(aucs)))
        scheduler.step()
        current_score = sum(aucs) # (sum(aucs) + avg_score) / 2
        if current_score >= best_score:
            best_score = current_score
            save_name = os.path.join(save_path, 'top10.pth')
            # torch.save(milnet.FClist[0].state_dict(), str(save_name).replace(".pth","fclist0.pth"))
            # torch.save(milnet.FClist[1].state_dict(), str(save_name).replace(".pth", "fclist1.pth"))
            # torch.save(milnet.FClist[2].state_dict(), str(save_name).replace(".pth", "fclist2.pth"))
            # torch.save(milnet.FClist[3].state_dict(), str(save_name).replace(".pth", "fclist3.pth"))
            torch.save(milnet.state_dict(), save_name)
            print('Best model saved at: ' + save_name)
            print('Best thresholds ===>>> ' + '|'.join(
                    'class-{}>>{}'.format(*k) for k in enumerate(thresholds_optimal)))

if __name__ == '__main__':
    main()