import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import hydra
import logging
from omegaconf import DictConfig, OmegaConf

from trans_train import exp
from utils import MultiExpMetric


@hydra.main(config_path=".", config_name="trans_config", version_base="1.1")
def main(cfg: DictConfig):
    logging.info(OmegaConf.to_yaml(cfg))
    # , "news20", "ca_cora", "cc_cora", "cc_citeseer", "dblp4k_conf", "dblp4k_paper", "dblp4k_term", "imdb_aw"
    # dataset_list = ["cd_1000", "cd_2000", "cd_4000", "cd_10000", "cd_20000", "cd_25000", "cd_30000", "cd_35000", "cd_40000", "cd_45000", "cd_50000", "cd_55000"] # "cd_2000", "cd_4000", "cd_10000", "cd_20000"
    dataset_list = ["cc_citeseer"]
    student_name_list = ["MLP", "KAN"] #  "MLP", , "KAN"
    plus_or_not_list = ["light_hgnn", "light_hgnnp"] # , "light_hgnnp"
    for dataset in dataset_list:
        cfg.data.name = dataset
        for student_name in student_name_list:
            cfg.model.student = student_name # 选择学生模型
            for plus_or_not in plus_or_not_list:
                cfg.model.new_stu = plus_or_not # 选择是否使用可靠超边拓扑蒸馏
                cfg.model.new_v_stu = "hgnn2kan"
                res_all = MultiExpMetric()
                tos_t = []  # 存放老师超图模型拓扑得分
                tos_s = []  # 存放学生超图模型拓扑得分
                tea_spend_time_list = []
                stu_spend_time_list = []
                tea_memory_used_list = []
                stu_memory_used_list = []
                for seed in range(5):
                    res, tos, tea_spend_time, stu_spend_time, tea_memory_used, stu_memory_used = exp(seed, cfg)
                    res_all.update(res)
                    tos_t.append(tos['tos_t'])
                    tos_s.append(tos['tos_s'])
                    tea_spend_time_list.append(tea_spend_time)
                    stu_spend_time_list.append(stu_spend_time)
                    tea_memory_used_list.append(tea_memory_used)
                    stu_memory_used_list.append(stu_memory_used)
                # logging.info(OmegaConf.to_yaml(cfg))
                # 计算res中'tos_t'和'tos_s'的平均值
                # logging.info(f"老师超图模型表现: {res_all.t}")
                
                # 计算tos_t和tos_s的平均值
                logging.info(f"dataset:{cfg.data.name}, teacher: {cfg.model.teacher}, student: {cfg.model.student}, new_stu: {cfg.model.new_stu}, new_v_stu: {cfg.model.new_v_stu}\n")
                logging.info(f"\n{res_all}")
                logging.info(f"老师超图模型平均拓扑得分: {sum(tos_t) / len(tos_t)}")
                logging.info(f"学生超图模型平均拓扑得分: {sum(tos_s) / len(tos_s)}")
                logging.info(f"教师模型平均运行时间: {sum(tea_spend_time_list) / len(tea_spend_time_list)}")
                logging.info(f"学生模型平均运行时间: {sum(stu_spend_time_list) / len(stu_spend_time_list)}")
                logging.info(f"教师模型GPU内存占用: {sum(tea_memory_used_list) / len(tea_memory_used_list)}")
                logging.info(f"学生模型GPU内存占用: {sum(stu_memory_used_list) / len(stu_memory_used_list)}")
                print('**' * 60)


if __name__ == "__main__":
    main()
