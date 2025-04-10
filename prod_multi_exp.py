import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

import hydra
import logging
from omegaconf import DictConfig, OmegaConf
import numpy as np
from prod_train import exp
from utils import MultiExpMetric
import torch


@hydra.main(config_path=".", config_name="prod_config", version_base="1.1")
def main(cfg: DictConfig):
    logging.info(OmegaConf.to_yaml(cfg))
    noise_ratio_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    hyper_dataset_list = ["dblp4k_paper"] # , "news20", "ca_cora", "cc_cora", "cc_citeseer", "dblp4k_conf", "dblp4k_paper", "dblp4k_term", "imdb_aw"
    teacher_name_list = ["hgnn", "hgnnp"] # , "hnhn", "unigcn", "unigat", "unisage", "unigin"
    student_name_list = ["MLP", "KAN" ] # 
    plus_e_or_not_list = ["light_hgnn", "light_hgnnp"] # 
    plus_v_or_not_list = ["hgnn2kan", "hgnn2kanp"]

    with open("/media/shared/panyonghao/project/hgnn2kan/cache/pro/prod_multi_exp_v2.log", "w") as f:
        for noise_ratio in noise_ratio_list:
            cfg.data.ft_noise_level = noise_ratio
            for dataset in hyper_dataset_list:
                cfg.data.name = dataset
                for teacher_name in teacher_name_list:
                    cfg.model.teacher = teacher_name # 选择教师模型
                    for student_name in student_name_list:
                        cfg.model.student = student_name # 选择学生模型
                        for plus_e_or_not in plus_e_or_not_list:
                            for plus_v_or_not in plus_v_or_not_list:
                                cfg.model.new_stu_e = plus_e_or_not # 选择是否使用可靠超边拓扑蒸馏
                                cfg.model.new_stu_v = plus_v_or_not
                                res_all = MultiExpMetric() # 存放结果
                                for seed in range(5):
                                    res = exp(seed, cfg)
                                    res_all.update(res)
                                # logging.info(OmegaConf.to_yaml(cfg))
                                print("**"*30)
                                logging.info(f"cfg.data.ft_noise_level: {cfg.data.ft_noise_level}, tea:{cfg.model.teacher}, stu:{cfg.model.student}, new_stu_e: {cfg.model.new_stu_e}, new_stu_v: {cfg.model.new_stu_v}\n")
                                logging.info(res_all)
                                str_tmp = "*****************************************************\n"
                                f.write(str_tmp)
                                f.write(f"cfg.data.ft_noise_level: {cfg.data.ft_noise_level}, tea:{cfg.model.teacher}, stu:{cfg.model.student}, new_stu_e: {cfg.model.new_stu_e}, new_stu_v: {cfg.model.new_stu_v}\n")
                                f.write(str(res_all)+"\n")
                                t_prod_acc = res_all.t["prod_accuracy"]
                                t_ind_acc = res_all.t["ind_accuracy"]
                                t_trans_acc = res_all.t["trans_accuracy"]
                                s_prod_acc = res_all.s["prod_accuracy"]
                                s_ind_acc = res_all.s["ind_accuracy"]
                                s_trans_acc = res_all.s["trans_accuracy"]

@hydra.main(config_path=".", config_name="prod_config", version_base="1.1")
def main2(cfg: DictConfig):
    logging.info(OmegaConf.to_yaml(cfg))
    noise_ratio_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    student_name_list = ["MLP", "KAN"] # 

    hgnn_acc_list = []
    hgnn_acc_list_copy = []
    lighthgnnp_acc_list = []
    hgnn2kanp_acc_list = []
    with open("/media/shared/panyonghao/project/hgnn2kan/cache/pro/prod_multi_exp_v3.log", "w") as f:
        for noise_ratio in noise_ratio_list:
            # 释放GPU未被引用的缓存显存
            for i in range(5):
                torch.cuda.empty_cache()

            cfg.data.ft_noise_level = noise_ratio
            cfg.data.name = "dblp4k_paper"
            cfg.model.teacher = "hgnn"  # 选择教师模型

            for student_name in student_name_list:
                cfg.model.student = student_name
                # 根据学生模型选择相应的超边拓扑蒸馏
                if student_name == "MLP":
                    cfg.model.new_stu_e = "light_hgnnp"
                    cfg.model.new_stu_v = "hgnn2kan"
                elif student_name == "KAN":
                    cfg.model.new_stu_e = "light_hgnnp"
                    cfg.model.new_stu_v = "hgnn2kanp"

                res_all = MultiExpMetric()  # 存放结果
                for seed in range(5):
                    res = exp(seed, cfg)
                    res_all.update(res)

                # 日志记录
                log_message = (
                    f"cfg.data.ft_noise_level: {cfg.data.ft_noise_level}, "
                    f"tea: {cfg.model.teacher}, stu: {cfg.model.student}, "
                    f"new_stu_e: {cfg.model.new_stu_e}, new_stu_v: {cfg.model.new_stu_v}\\n"
                )
                logging.info(log_message)
                logging.info(res_all)

                # 写入文件
                f.write("*****************************************************\\n")
                f.write(log_message)
                f.write(f"{res_all}\\n")

                # 记录准确率
                t_prod_acc = res_all.t["prod_accuracy"]
                s_prod_acc = res_all.s["prod_accuracy"]
                
                if student_name == "MLP":
                    lighthgnnp_acc_list.append(s_prod_acc)
                    hgnn_acc_list.append(t_prod_acc)
                elif student_name == "KAN":
                    hgnn2kanp_acc_list.append(s_prod_acc)
                    hgnn_acc_list_copy.append(t_prod_acc)
    return [hgnn_acc_list, hgnn_acc_list_copy, lighthgnnp_acc_list, hgnn2kanp_acc_list]


@hydra.main(config_path=".", config_name="prod_config", version_base="1.1")
def main3(cfg: DictConfig):
    logging.info(OmegaConf.to_yaml(cfg))
    noise_ratio_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    student_name_list = ["MLP", "KAN" ] # 

    hgnn_acc_list = []
    hgnn_acc_list_copy = []
    lighthgnnp_acc_list = []
    hgnn2kanp_acc_list = []
    with open("/media/shared/panyonghao/project/hgnn2kan/cache/pro/prod_multi_exp_train_split.log", "w") as f:
        for noise_ratio in noise_ratio_list:
            cfg.data.ft_noise_level = noise_ratio
            cfg.data.name = "dblp4k_paper"
            cfg.model.teacher = "hgnn"  # 选择教师模型

            for student_name in student_name_list:
                # 根据学生模型选择相应的超边拓扑蒸馏
                if student_name == "MLP":
                    cfg.model.new_stu_e = "light_hgnnp"
                    cfg.model.new_stu_v = "hgnn2kan"
                elif student_name == "KAN":
                    cfg.model.new_stu_e = "light_hgnnp"
                    cfg.model.new_stu_v = "hgnn2kanp"

                res_all = MultiExpMetric()  # 存放结果
                for seed in range(5):
                    res = exp(seed, cfg)
                    res_all.update(res)

                # 日志记录
                log_message = (
                    f"cfg.data.ft_noise_level: {cfg.data.ft_noise_level}, "
                    f"tea: {cfg.model.teacher}, stu: {cfg.model.student}, "
                    f"new_stu_e: {cfg.model.new_stu_e}, new_stu_v: {cfg.model.new_stu_v}\\n"
                )
                logging.info(log_message)
                logging.info(res_all)

                # 写入文件
                f.write("*****************************************************\\n")
                f.write(log_message)
                f.write(f"{res_all}\\n")

                # 记录准确率
                t_prod_acc = res_all.t["prod_accuracy"]
                s_prod_acc = res_all.s["prod_accuracy"]
                
                if student_name == "MLP":
                    lighthgnnp_acc_list.append(s_prod_acc)
                    hgnn_acc_list.append(t_prod_acc)
                elif student_name == "KAN":
                    hgnn2kanp_acc_list.append(s_prod_acc)
                    hgnn_acc_list_copy.append(t_prod_acc)
    return [hgnn_acc_list, hgnn_acc_list_copy, lighthgnnp_acc_list, hgnn2kanp_acc_list]


if __name__ == "__main__":
    # main()
    result = main2()
    print(result)
