import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

import hydra
import logging
from copy import deepcopy
from omegaconf import DictConfig, OmegaConf
from utils import load_data, product_split, sub_hypergraph, fix_iso_v

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from dhg.nn import MLP
from dhg import Hypergraph
from dhg.random import set_seed
from dhg.utils import split_by_num
from dhg.models import HGNNP, HGNN, HNHN, UniGCN, UniGAT, UniSAGE, UniGIN
from dhg.metrics import HypergraphVertexClassificationEvaluator as Evaluator
from kan_layer import NaiveFourierKANLayer as KANLayer
from kan import MultKAN
from my_model import MyKAN, MyHGNN, MyMLPs
from tqdm import tqdm

# =========================================================================
# train teacher
def train(net, X, G, lbls, train_mask, optimizer):
    net.train()
    optimizer.zero_grad()
    outs = net(X, G)
    loss = F.nll_loss(F.log_softmax(outs[train_mask], dim=1), lbls[train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def valid(net, X, G, lbls, mask, evaluator):
    net.eval()
    outs = net(X, G)
    res = evaluator.validate(lbls[mask], outs[mask])
    return res


@torch.no_grad()
def test(net, X_t, G_t, lbls_t, mask_t, X, G, lbls, mask, prod_mask, evaluator):
    net.eval()
    # transductive
    outs_t = net(X_t, G_t)
    res_t = evaluator.test(lbls_t[mask_t], outs_t[mask_t])
    # # inductive
    outs = net(X, G)
    res_i = evaluator.test(lbls[mask], outs[mask])
    # product
    outs = net(X, G)
    res_p = evaluator.test(lbls[prod_mask], outs[prod_mask])
    res = {}
    for k, v in res_p.items():
        res[f"prod_{k}"] = v
    for k, v in res_i.items():
        res[f"ind_{k}"] = v
    for k, v in res_t.items():
        res[f"trans_{k}"] = v
    return res


# =========================================================================
# train student
class HighOrderConstraint(nn.Module):
    def __init__(self, model, X, G, noise_level=1.0, tau=1.0):
        super().__init__()
        model.eval()
        self.tau = tau
        pred = model(X, G).softmax(dim=-1).detach()
        entropy_x = -(pred * pred.log()).sum(1, keepdim=True)
        entropy_x[entropy_x.isnan()] = 0
        entropy_e = G.v2e(entropy_x, aggr="mean")

        X_noise = X.clone() * (torch.randn_like(X) + 1) * noise_level
        pred_ = model(X_noise, G).softmax(dim=-1).detach()
        entropy_x_ = -(pred_ * pred_.log()).sum(1, keepdim=True)
        entropy_x_[entropy_x_.isnan()] = 0
        entropy_e_ = G.v2e(entropy_x_, aggr="mean")

        self.delta_e_ = (entropy_e_ - entropy_e).abs()
        self.delta_e_ = 1 - self.delta_e_ / self.delta_e_.max()
        self.delta_e_ = self.delta_e_.squeeze()

    def forward(self, pred_s, pred_t, G):
        pred_s, pred_t = F.softmax(pred_s, dim=1), F.softmax(pred_t, dim=1)
        e_mask = torch.bernoulli(self.delta_e_).bool()
        pred_s_e = G.v2e(pred_s, aggr="mean")
        pred_s_e = pred_s_e[e_mask]
        pred_t_e = G.v2e(pred_t, aggr="mean")
        pred_t_e = pred_t_e[e_mask]
        loss = F.kl_div(torch.log(pred_s_e / self.tau), pred_t_e / self.tau, reduction="batchmean", log_target=True)
        return loss


# 高可靠节点判断
class HighOrderVerticeConstraint(nn.Module):
    def __init__(self, model, X, G, noise_level=1.0, tau=1.0):
        super().__init__()
        model.eval()
        self.tau = tau
        pred = model(X, G).softmax(dim=-1).detach()
        entropy_x = -(pred * pred.log()).sum(1, keepdim=True)
        entropy_x[entropy_x.isnan()] = 0
        # entropy_e = G.v2e(entropy_x, aggr="mean")

        X_noise = X.clone() * (torch.randn_like(X) + 1) * noise_level
        pred_ = model(X_noise, G).softmax(dim=-1).detach()
        entropy_x_ = -(pred_ * pred_.log()).sum(1, keepdim=True)
        entropy_x_[entropy_x_.isnan()] = 0
        # entropy_e_ = G.v2e(entropy_x_, aggr="mean")

        self.delta_x_ = (entropy_x_ - entropy_x).abs()
        self.delta_x_ = 1 - self.delta_x_ / self.delta_x_.max()
        self.delta_x_ = self.delta_x_.squeeze()

    def forward(self, pred_s, pred_t, G):
        pred_s, pred_t = F.softmax(pred_s, dim=1), F.softmax(pred_t, dim=1)
        v_mask = torch.bernoulli(self.delta_x_).bool()
        # pred_s_e = G.v2e(pred_s, aggr="mean")
        pred_s_v = pred_s[v_mask]
        # pred_t_e = G.v2e(pred_t, aggr="mean")
        pred_t_v = pred_t[v_mask]
        loss = F.kl_div(torch.log(pred_s_v / self.tau), pred_t_v / self.tau, reduction="batchmean", log_target=True)
        return loss
    

def train_stu(net, X, G, lbls, out_t, train_mask, optimizer, hec=None, hvc=None, lamb=0):
    net.train()
    optimizer.zero_grad()
    outs = net(X)
    loss_x = F.nll_loss(F.log_softmax(outs[train_mask], dim=1), lbls[train_mask])
    loss_k = F.kl_div(F.log_softmax(outs, dim=1), F.softmax(out_t, dim=1), reduction="batchmean", log_target=True)
    if hec is not None:
        loss_h = hec(outs, out_t, G)
        loss_k = loss_h + loss_k
    if hvc is not None: # 节点增强
        loss_h = hvc(outs, out_t, G)
        loss_k = loss_h + loss_k
    loss = loss_x * lamb + loss_k * (1 - lamb)
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def valid_stu(net, X, lbls, mask, evaluator):
    net.eval()
    outs = net(X)
    res = evaluator.validate(lbls[mask], outs[mask])
    return res


@torch.no_grad()
def test_stu(net, X_t, lbls_t, mask_t, X, lbls, mask_i, prod_mask, evaluator):
    net.eval()
    # transductive
    outs_t = net(X_t)
    res_t = evaluator.test(lbls_t[mask_t], outs_t[mask_t])
    # inductive
    outs = net(X)
    res_i = evaluator.test(lbls[mask_i], outs[mask_i])
    # product
    outs = net(X)
    res_p = evaluator.test(lbls[prod_mask], outs[prod_mask])
    res = {}
    for k, v in res_p.items():
        res[f"prod_{k}"] = v
    for k, v in res_i.items():
        res[f"ind_{k}"] = v
    for k, v in res_t.items():
        res[f"trans_{k}"] = v
    return res


def exp(seed, cfg: DictConfig):
   set_seed(seed)
   device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
   evaluator = Evaluator(["accuracy", "f1_score", {"f1_score": {"average": "micro"}}])

   data, edge_list = load_data(cfg.data.name)

   G = Hypergraph(data["num_vertices"], edge_list)
   G = fix_iso_v(G)
   train_mask, val_mask, test_mask = split_by_num(data["num_vertices"], data["labels"], cfg.data.num_train, cfg.data.num_val)
   obs_idx, obs_train_mask, obs_val_mask, obs_test_mask, test_ind_mask = product_split(train_mask, val_mask, test_mask, cfg.data.test_ind_ratio)
   G_t = sub_hypergraph(G, obs_idx)
   G_t = fix_iso_v(G_t)
   X, lbl = data["features"], data["labels"]
   X_t, lbl_t = X[obs_idx], lbl[obs_idx]

   if cfg.model.teacher == "hgnn":
       net = HGNN(X.shape[1], 32, data["num_classes"], use_bn=False)
   elif cfg.model.teacher == "hgnnp":
       net = HGNNP(X.shape[1], 32, data["num_classes"], use_bn=False)
   elif cfg.model.teacher == "hnhn":
       net = HNHN(X.shape[1], 32, data["num_classes"], use_bn=False)
   elif cfg.model.teacher == "unigcn":
       net = UniGCN(X.shape[1], 32, data["num_classes"], use_bn=False)
   elif cfg.model.teacher == "unigat":
       net = UniGAT(X.shape[1], 8, data["num_classes"], 4, use_bn=False)
   elif cfg.model.teacher == "unisage":
       net = UniSAGE(X.shape[1], 32, data["num_classes"], use_bn=False)
   elif cfg.model.teacher == "unigin":
       net = UniGIN(X.shape[1], 32, data["num_classes"], use_bn=False)
   else:
       raise NotImplementedError

   # train teacher
   optimizer = optim.Adam(net.parameters(), lr=0.01, weight_decay=5e-4)
   X, lbl, G = X.to(device), lbl.to(device), G.to(device)
   X_t, lbl_t, G_t = X_t.to(device), lbl_t.to(device), G_t.to(device)
   net = net.to(device)

   best_state = None
   best_epoch, best_val = 0, 0
   for epoch in range(200):
       # train
       train(net, X_t, G_t, lbl_t, obs_train_mask, optimizer)
       # validation
       if epoch % 1 == 0:
           with torch.no_grad():
               val_res = valid(net, X_t, G_t, lbl_t, obs_val_mask, evaluator)
           if val_res > best_val:
               best_epoch = epoch
               best_val = val_res
               best_state = deepcopy(net.state_dict())
   # test
   if cfg.data.ft_noise_level > 0:
       X_noise = (1 - cfg.data.ft_noise_level) * X + cfg.data.ft_noise_level * torch.randn_like(X)
       X_t_noise = (1 - cfg.data.ft_noise_level) * X_t + cfg.data.ft_noise_level * torch.randn_like(X_t)
   net.load_state_dict(best_state)
   res_t = test(net, X_t, G_t, lbl_t, obs_test_mask, X_noise, G, lbl, test_ind_mask, test_mask, evaluator)
   # logging.info(f"teacher test best epoch: {best_epoch}, res: {res_t}")

   # -------------------------------------------------------------------------------------
   # train student
   out_t = net(X_t, G_t).detach()
   if cfg.model.new_stu_e == "light_hgnnp":
       hec = HighOrderConstraint(net, X_t, G_t, noise_level=cfg.data.hc_noise_level, tau=cfg.loss.tau)
   else:
       hec = None

   if cfg.model.new_stu_v == "hgnn2kanp":
       hvc = HighOrderVerticeConstraint(net, X_t, G_t, noise_level=cfg.data.hc_noise_level, tau=cfg.loss.tau)
   else:
       hvc = None

   if cfg.model.student == "MLP":
       net_s = nn.Sequential(MLP([X.shape[1], cfg.model.hid]), nn.Linear(cfg.model.hid, data["num_classes"]))
   else:
       # net_s = nn.Sequential(KANLayer(X.shape[1], cfg.model.hid), nn.Linear(cfg.model.hid, data["num_classes"]))
       net_s = MyKAN(X.shape[1], cfg.model.hid, data["num_classes"])
   optimizer = optim.Adam(net_s.parameters(), lr=0.01, weight_decay=5e-4)
   net_s = net_s.to(device)

   best_state = None
   best_epoch, best_val = 0, 0
   for epoch in range(200):
       # train
       # train_stu(net_s, X_t, G_t, lbl_t, out_t, obs_train_mask, optimizer, hc=hc, lamb=cfg.loss.lamb)
       train_stu(net_s, X_t, G_t, lbl_t, out_t, obs_train_mask, optimizer, hec=hec, hvc=hvc, lamb=cfg.loss.lamb)

       # validation
       if epoch % 1 == 0:
           with torch.no_grad():
               val_res = valid_stu(net_s, X_t, lbl_t, obs_val_mask, evaluator)
           if val_res > best_val:
               best_epoch = epoch
               best_val = val_res
               best_state = deepcopy(net_s.state_dict())
   # test
   net_s.load_state_dict(best_state)
   res_s = test_stu(net_s, X_t, lbl_t, obs_test_mask, X_noise, lbl, test_ind_mask, test_mask, evaluator)
   # logging.info(f"student test best epoch: {best_epoch}, res: {res_s}\\n")
   return {"t": res_t, "s": res_s}

@hydra.main(config_path=".", config_name="prod_config", version_base="1.1")
def save_model(cfg: DictConfig):
    set_seed(2023)
    # 确定设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 初始化评估器
    metrics = ["accuracy", "f1_score", {"f1_score": {"average": "micro"}}]
    evaluator = Evaluator(metrics)

    # 加载数据
    data, edge_list = load_data(cfg.data.name)

    # 创建超图并修复孤立节点
    G = fix_iso_v(Hypergraph(data["num_vertices"], edge_list))

    # 划分训练、验证和测试集
    train_mask, val_mask, test_mask = split_by_num(data["num_vertices"], data["labels"], cfg.data.num_train, cfg.data.num_val)

    # 生成观察索引和掩码
    obs_idx, obs_train_mask, obs_val_mask, obs_test_mask, test_ind_mask = product_split(train_mask, val_mask, test_mask, cfg.data.test_ind_ratio)

    # 创建子超图并修复孤立节点
    G_t = fix_iso_v(sub_hypergraph(G, obs_idx))

    # 提取特征和标签
    X, lbl = data["features"], data["labels"]
    X_t, lbl_t = X[obs_idx], lbl[obs_idx]

    # 选择HGNN教师模型
    net = MyHGNN(X.shape[1], 32, data["num_classes"], use_bn=False)

    # 训练教师模型
    optimizer = optim.Adam(net.parameters(), lr=0.01, weight_decay=5e-4)
    X, lbl, G = X.to(device), lbl.to(device), G.to(device)
    X_t, lbl_t, G_t = X_t.to(device), lbl_t.to(device), G_t.to(device)
    net = net.to(device)

    best_state, best_val = None, 0
    for epoch in range(200):
        # 训练模型
        train(net, X_t, G_t, lbl_t, obs_train_mask, optimizer)

        # 每个epoch进行验证
        if epoch % 1 == 0:
            with torch.no_grad():
                val_res = valid(net, X_t, G_t, lbl_t, obs_val_mask, evaluator)
            
            # 更新最佳验证结果和模型状态
            if val_res > best_val:
                best_val = val_res
                best_state = deepcopy(net.state_dict())
    # 测试教师模型
    net.load_state_dict(best_state)
    res_t = test(net, X_t, G_t, lbl_t, obs_test_mask, X, G, lbl, test_ind_mask, test_mask, evaluator)
    # 保存最佳模型
    model_save_path = f'/media/shared/panyonghao/project/hgnn2kan/ckpts/{cfg.data.name}_HGNN_2_32.pth'
    torch.save(best_state, model_save_path)

    # 训练学生模型
    out_t = net(X_t, G_t).detach()

    # 初始化高阶约束
    hec = HighOrderConstraint(net, X_t, G_t, noise_level=cfg.data.hc_noise_level, tau=cfg.loss.tau)

    # 构建学生模型
    hvc = None
    student_model_name = ""
    if cfg.model.student == "MLP":
        net_s = nn.Sequential(MLP([X.shape[1], cfg.model.hid]), nn.Linear(cfg.model.hid, data["num_classes"]))
        student_model_name = "LightHGNNs"
    else:
        net_s = MyKAN(X.shape[1], cfg.model.hid, data["num_classes"])
        hvc = HighOrderVerticeConstraint(net, X_t, G_t, noise_level=cfg.data.hc_noise_level, tau=cfg.loss.tau) 
        student_model_name = "HGNN2KANs"

    # 设置优化器
    optimizer = optim.Adam(net_s.parameters(), lr=0.01, weight_decay=5e-4)
    net_s = net_s.to(device)

    # 初始化最佳状态
    best_state, best_val = None, 0

    # 训练过程
    for epoch in range(200):
        train_stu(net_s, X_t, G_t, lbl_t, out_t, obs_train_mask, optimizer, hec=hec, hvc=hvc, lamb=cfg.loss.lamb)

        # 验证过程
        if epoch % 1 == 0:
            with torch.no_grad():
                val_res = valid_stu(net_s, X_t, lbl_t, obs_val_mask, evaluator)
            if val_res > best_val:
                best_val = val_res
                best_state = deepcopy(net_s.state_dict())

    # 测试学生模型
    net_s.load_state_dict(best_state)
    res_s = test_stu(net_s, X_t, lbl_t, obs_test_mask, X, lbl, test_ind_mask, test_mask, evaluator)
    # 保存学生模型
    model_save_path = f'/media/shared/panyonghao/project/hgnn2kan/ckpts/{cfg.data.name}_{student_model_name}_2_128.pth'
    torch.save(net_s.state_dict(), model_save_path)
    print(f"Teacher: {res_t},\n Student: {res_s}")
    return {"t": res_t, "s": res_s}

@hydra.main(config_path=".", config_name="prod_config", version_base="1.1")
def test_model_by_noise(cfg: DictConfig):
    """
    测试不同噪声比例对模型效果的影响
    """
    set_seed(2023)
    # 确定设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 初始化评估器
    metrics = ["accuracy", "f1_score", {"f1_score": {"average": "micro"}}]
    evaluator = Evaluator(metrics)

    # 加载数据
    data, edge_list = load_data(cfg.data.name)

    # 创建超图并修复孤立节点
    G = fix_iso_v(Hypergraph(data["num_vertices"], edge_list))

    # 划分训练、验证和测试集
    train_mask, val_mask, test_mask = split_by_num(data["num_vertices"], data["labels"], cfg.data.num_train, cfg.data.num_val)

    # 生成观察索引和掩码
    obs_idx, obs_train_mask, obs_val_mask, obs_test_mask, test_ind_mask = product_split(train_mask, val_mask, test_mask, cfg.data.test_ind_ratio)

    # 创建子超图并修复孤立节点
    G_t = fix_iso_v(sub_hypergraph(G, obs_idx))

    # 提取特征和标签
    X, lbl = data["features"], data["labels"]
    X_t, lbl_t = X[obs_idx], lbl[obs_idx]
    X, lbl, G = X.to(device), lbl.to(device), G.to(device)
    X_t, lbl_t, G_t = X_t.to(device), lbl_t.to(device), G_t.to(device)

    # 从文件加载模型
    ckpt_path = f'/media/shared/panyonghao/project/hgnn2kan/ckpts/'
    # tea_model_name = "dblp4k_paper_HGNN_2_32.pth" # "paper_hgnn_2_32.pth"
    # stu_mlp_name = "dblp4k_paper_LightHGNNs_2_128.pth" # "paer_lighthgnnp_2_128.pth"
    # stu_kan_name = "dblp4k_paper_HGNN2KANs_2_128.pth" # "paper_hgnn2kanp_2_128.pth"
    model_name = "paper_hgnn2kanp_2_128.pth"
    model_path = os.path.join(ckpt_path, model_name)
    net = MyHGNN(X.shape[1], 32, data["num_classes"], use_bn=False)
    # net = nn.Sequential(MLP([X.shape[1], 128]), nn.Linear(128, data["num_classes"]))
    # net = MyKAN(X.shape[1], cfg.model.hid, data["num_classes"])
    # net.load_state_dict(torch.load(model_path))
    net = torch.load(model_path)
    net = net.to(device)
    print(f"加载预训练模型成功: {model_name}")

    # 测试不同噪声比例对模型的影响
    result_list = []
    for noise_level in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
        X_noise = (1 - noise_level) * X + noise_level * torch.normal(0,0.1,X.shape).to(device)
        X_t_noise = (1 - noise_level) * X_t + noise_level * torch.normal(0,0.1,X_t.shape).to(device)

        # 判断模型类型
        if "32" in model_name:
            res = test(net, X_t_noise, G_t, lbl_t, obs_test_mask, X_noise, G, lbl, test_ind_mask, test_mask, evaluator)
        else:
            res = test_stu(net, X_t_noise, lbl_t, obs_test_mask, X_noise, lbl, test_ind_mask, test_mask, evaluator)
        result_list.append(res)

    return result_list


# 处理实验结果数据
def process_results(result_list):
    # 提取指定键的值
    values = [
        {
            'prod_accuracy': item['prod_accuracy'],
            'ind_accuracy': item['ind_accuracy'],
            'trans_accuracy': item['trans_accuracy']
        }
        for item in result_list
    ]

    # 输出结果
    prod_list = []
    ind_list = []
    trans_list = []
    for v in values:
        prod_list.append(round(v['prod_accuracy']*100, 2))
        ind_list.append(round(v['ind_accuracy']*100, 2))
        trans_list.append(round(v['trans_accuracy']*100, 2 ))
    print(f"prod_list: {prod_list}")
    print(f"ind_list: {ind_list}")
    print(f"trans_list: {trans_list}")
    return [prod_list, ind_list, trans_list]


@hydra.main(config_path=".", config_name="prod_config", version_base="1.1")
def main(cfg: DictConfig):
    # res = exp(2023, cfg)
    # logging.info(OmegaConf.to_yaml(cfg))
    # logging.info(f"teacher: {res['t']}")
    # logging.info(f"student: {res['s']}")
    result = test_model_by_noise(cfg)
    process_results(result)


if __name__ == "__main__":
    main()
    # save_model()

