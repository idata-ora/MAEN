import h5py
from matplotlib import pyplot as plt

import TRANS_NET as MIL_model
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.models as models
import torchvision.transforms.functional as VF
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve, roc_auc_score, classification_report
from torchvision import transforms
import sys, argparse, os, glob
import pandas as pd
import numpy as np
from PIL import Image
from collections import OrderedDict
from skimage import exposure, io, img_as_ubyte, transform
import warnings
from sklearn.utils import shuffle
from torch.autograd import Variable

class InstanceFC(nn.Module):
    def __init__(self, in_size, out_size=2):
        super(InstanceFC, self).__init__()
        self.fc = nn.Linear(in_size, out_size)
    def forward(self, feats):
        x = self.fc(feats)
        return x

def get_bag_feats(csv_file_df, args, all_svs_label):

    feats_csv_path = csv_file_df.iloc[0]
    df = pd.read_csv(feats_csv_path)
    # feats = shuffle(df).reset_index(drop=True)
    feats = df.to_numpy()

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
    # print(svs_basename, feats)
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

    return label, feats, er_label, pr_label, her2_label, svs_basename

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

def test(args, bags_list, milnet, all_svs_label):
    milnet.eval()
    num_bags = len(bags_list)
    print(num_bags)
    Tensor = torch.FloatTensor
    pred_id = 0
    test_labels = []
    test_predictions = []
    svs_name=[]
    la=[]
    lb=[]
    HE=[]
    BS=[]
    lab=[]
    pre=[]
    for i in range(0, num_bags):
        with torch.no_grad():
            label, feats, er_label, pr_label, her2_label, svs_basename = get_bag_feats(bags_list.iloc[i], args, all_svs_label)
            bag_feats = Variable(Tensor([feats])).cuda()
            bag_feats = bag_feats.view(-1, args.feats_size)
            C, InstancePred = milnet(bag_feats)
            # from pdb import set_trace as st; st()

            '''
            csv_file = "/home4/lsy/BRCA_data/patch_coords"+"/"+svs_basename+".csv"
            coord = pd.read_csv(csv_file)
            # print(coord)
            # for kkk in range(len(InstancePred)):
            _, m_indices = torch.sort(InstancePred[3], 0,
                                  descending=True)  # sort class scores along the instance dimension, m_indices in shape N x C
            index = m_indices[0, :]
            print(svs_basename, index, coord.loc[int(index[0].item())], coord.loc[int(index[1].item())],
                  coord.loc[int(index[2].item())], coord.loc[int(index[3].item())]) # 0 n ; 1 p; ['LumA', 'LumB', 'Her2', 'Basal']
            '''

            max_prediction, _ = torch.max(InstancePred[3], 0)
            # bag_prediction = (0.5 * torch.sigmoid(max_prediction) + 0.5 * torch.sigmoid(C)).squeeze().cpu().numpy()
            bag_prediction = (0.5 * torch.sigmoid(C)).squeeze().cpu().numpy()
            test_labels.extend([label])
            test_predictions.extend([bag_prediction])
            CLASS = ['Luminal A', 'Luminal B', 'HER2-enriched', 'TNBC']
            if "TCGA" in svs_basename:
                if np.argmax(label) != np.argmax(bag_prediction):
                    print(svs_basename, bag_prediction, np.argmax(label), np.argmax(bag_prediction))
                    svs_name.append(svs_basename)
                    la.append(bag_prediction[0])
                    lb.append(bag_prediction[1])
                    HE.append(bag_prediction[2])
                    BS.append(bag_prediction[3])
                    lab.append(CLASS[np.argmax(label)])
                    pre.append(CLASS[np.argmax(bag_prediction)])

            # print(svs_basename, bag_prediction)
            # print(svs_basename,C, max_prediction, bag_prediction)
            '''
            if bag_prediction[0] >= args.thres_BRCA[0]:
                pred = 'LumA'
                print(bags_list.iloc[i][1], 'LumA')

            elif bag_prediction[1] >= args.thres_BRCA[1]:
                pred = 'LumB'
                print(bags_list.iloc[i][1], 'LumB')

            elif bag_prediction[2] >= args.thres_BRCA[2]:
                pred = 'Her2'
                print(bags_list.iloc[i][1], 'Her2')

            elif bag_prediction[3] >= args.thres_BRCA[3]:
                pred = 'Basal'
                print(bags_list.iloc[i][1], 'Basal')

            else:
                pred = 'erro'
                print(bags_list.iloc[i][1], 'erro')
            
            if pred == bags_list.iloc[i][1]:
                pred_id =pred_id +1
            '''

            '''
            color_map = np.zeros((np.amax(pos_arr, 0)[0]+1, np.amax(pos_arr, 0)[1]+1, 3))
            attentions = attentions.cpu().numpy()
            attentions = exposure.rescale_intensity(attentions, out_range=(0, 1))
            for k, pos in enumerate(pos_arr):
                tile_color = np.asarray(color) * attentions[k]
                color_map[pos[0], pos[1]] = tile_color
            slide_name = os.path.basename(all_path)
            color_map = transform.resize(color_map, (color_map.shape[0]*32, color_map.shape[1]*32), order=0)
            io.imsave(os.path.join('/home4/lsy/BRCA_data/test', 'output', slide_name+'.png'), img_as_ubyte(color_map))
            '''

    data2yy = {
        "ID": svs_name,
        "Luminal A": la,
        "Luminal B": lb,
        "HER2-enriched": HE,
        "TNBC": BS,
        "Prediction":pre,
        "Label":lab
    }
    data2yy = pd.DataFrame(data2yy)
    data2yy.to_csv("/home2/lsy/BRCA_data/yy_Probability.csv")

    test_labels = np.array(test_labels)
    test_predictions = np.array(test_predictions)
    auc_value, _, thresholds_optimal = multi_label_roc(test_labels, test_predictions, args.num_classes,class_name=['LumA', 'LumB', 'Her2', 'Basal'])
    print(auc_value)
    print(pred_id)

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
    print(classification_report(ll, pp, target_names=class_name, digits=9))
    fig, ax = plt.subplots(figsize=(12, 11))
    CLASS_LABEL = ['Luminal A', 'Luminal B', 'HER2-enriched', 'Basal-like']
    colors=['darkorange','limegreen','steelblue','pink']
    data_csv = [[] for i in range(8)]
    for c in range(0, num_classes):
        label = labels[:, c]
        prediction = predictions[:, c]
        data_csv[2 * c] = label
        data_csv[2 * c + 1] = prediction
        fpr, tpr, threshold = roc_curve(label, prediction, pos_label=1)
        fpr_optimal, tpr_optimal, threshold_optimal = optimal_thresh(fpr, tpr, threshold)
        c_auc = roc_auc_score(label, prediction)
        aucs.append(c_auc)
        thresholds.append(threshold)
        thresholds_optimal.append(threshold_optimal)

        plt.plot(fpr, tpr, color=colors[c],
                 lw=4,label="%s (AUC = %0.2f)" % (CLASS_LABEL[c],c_auc))  ###假正率为横坐标，真正率为纵坐标做曲线plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

    data_pd = {
        'Luminal_A_label': data_csv[0],
        'Luminal_A_pred': data_csv[1],
        'Luminal_B_label': data_csv[2],
        'Luminal_B_pred': data_csv[3],
        'HER2-enriched_label': data_csv[4],
        'HER2-enriched_pred': data_csv[5],
        'TNBC_label': data_csv[6],
        'TNBC_pred': data_csv[7]
    }
    data_pd = pd.DataFrame(data_pd)
    data_pd.to_csv("/home2/lsy/BRCA_data/shiyan_outcome_csv/MAEN.csv")

    plt.plot([0, 1], [0, 1], color=[0.593, 0.668, 0.535], lw=4, linestyle="--")
    plt.xlim([0, 1.0])
    plt.ylim([0, 1.0])
    plt.xlabel("False Positive Rate", {'family': 'Calibri', 'weight': 'normal', 'size': 40})
    plt.ylabel("True Positive Rate", {'family': 'Calibri', 'weight': 'normal', 'size': 40})
    # plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right", prop={'family': 'Calibri', 'size': 24})
    plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], [0, 0.2, 0.4, 0.6, 0.8, 1.0], fontproperties='Calibri', size=40)
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], [0, 0.2, 0.4, 0.6, 0.8, 1.0], fontproperties='Calibri', size=40)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_aspect(1)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    plt.show()
    print(np.mean(aucs))
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
    config.add_argument('--attention_dropout_rate', default=False, type=float)
    config.add_argument('--dropout_rate', default=False, type=float)
    config.add_argument('--classifier', default='token', type=str)
    config.add_argument('--representation_size', default=None, type=int)
    args = config.parse_args()
    return args

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Testing workflow includes attention computing and color map production')
    parser.add_argument('--num_classes', type=int, default=4, help='Number of output classes')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size of feeding patches')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--feats_size', type=int, default=512)
    parser.add_argument('--class_name', type=str, default='IHC')
    parser.add_argument('--thres_BRCA', type=float, default=[0.48853808641433716, 0.39991703629493713, 0.39052528142929077, 0.5166370868682861])
    # parser.add_argument('--thres_lusc', type=float, default=0.2752) AUC: class-0>>0.8346298801700811|class-1>>0.6340403782264247|class-2>>0.8425067672873432|class-3>>0.8437543133195308
    args = parser.parse_args()
    config = get_config()

    ER_classifier = InstanceFC(in_size=args.feats_size, out_size=2).cuda()
    PR_classifier = InstanceFC(in_size=args.feats_size, out_size=2).cuda()
    Her2_classifier = InstanceFC(in_size=args.feats_size, out_size=2).cuda()
    IHC_classifier = InstanceFC(in_size=args.feats_size, out_size=4).cuda()
    # er_state_dict = torch.load("/home2/lsy/BRCA_data/BRCA_ONLY_CLASS/src/weights/03082023/1fclist0.pth")
    # pr_state_dict = torch.load("/home2/lsy/BRCA_data/BRCA_ONLY_CLASS/src/weights/03082023/1fclist1.pth")
    # her2_state_dict = torch.load("/home2/lsy/BRCA_data/BRCA_ONLY_CLASS/src/weights/03082023/1fclist2.pth")
    # ihc_state_dict = torch.load("/home2/lsy/BRCA_data/BRCA_ONLY_CLASS/src/weights/03082023/1fclist3.pth")

    # er_state_dict = torch.load("/home2/lsy/BRCA_data/BRCA_ONLY_CLASS/src/weights/03082023/1fclist0.pth")
    # pr_state_dict = torch.load("/home2/lsy/BRCA_data/BRCA_ONLY_CLASS/src/weights/03082023/1fclist1.pth")
    # her2_state_dict = torch.load("/home2/lsy/BRCA_data/BRCA_ONLY_CLASS/src/weights/03082023/1fclist2.pth")
    # ihc_state_dict = torch.load("/home2/lsy/BRCA_data/BRCA_ONLY_CLASS/src/weights/03082023/1fclist3.pth")
    # er_state_dict["fc.weight"] = er_state_dict["fc.0.weight"]
    # er_state_dict["fc.bias"] = er_state_dict["fc.0.bias"]
    # pr_state_dict["fc.weight"] = pr_state_dict["fc.0.weight"]
    # pr_state_dict["fc.bias"] = pr_state_dict["fc.0.bias"]
    # her2_state_dict["fc.weight"] = her2_state_dict["fc.0.weight"]
    # her2_state_dict["fc.bias"] = her2_state_dict["fc.0.bias"]
    # ihc_state_dict["fc.weight"] = ihc_state_dict["fc.0.weight"]
    # ihc_state_dict["fc.bias"] = ihc_state_dict["fc.0.bias"]
    # ER_classifier.load_state_dict(er_state_dict, strict=False)
    # PR_classifier.load_state_dict(pr_state_dict, strict=False)
    # Her2_classifier.load_state_dict(her2_state_dict, strict=False)
    # IHC_classifier.load_state_dict(ihc_state_dict, strict=False)

    FClist = [ER_classifier, PR_classifier, Her2_classifier, IHC_classifier]
    milnet = MIL_model.BagMIL(config, FClist, input_size=args.feats_size, output_class=4).cuda()
    # state_dict_weights = torch.load(os.path.join('/home2/lsy/BRCA_data/BRCA_ONLY_CLASS/src/weights/03082023/1.pth'))
    state_dict_weights = torch.load(os.path.join('/home2/lsy/BRCA_data/BRCA_ONLY_CLASS/src/weights/10212025/top50.pth'))
    milnet.load_state_dict(state_dict_weights, strict=False)
    print(">>> Loaded keys from state_dict:")
    for k in state_dict_weights.keys():
        print(k)

    all_svs_label = all_label(txt_path=r"/home2/lsy/BRCA_data/BRCA_ONLY_CLASS/datatxt/TCGA_YY_LABEL.txt")
    bags_path = pd.read_csv('/home4/lsy/BRCA_data/patch_features/BRCA.csv')
    test_path = bags_path.iloc[int(len(bags_path) * (1 - 0.2)):, :]
    # print(test_path)
    # test_path = shuffle(test_path).reset_index(drop=True)
    # os.makedirs(os.path.join('/home4/lsy/BRCA_data/test', 'output'), exist_ok=True)
    test(args, test_path, milnet, all_svs_label)