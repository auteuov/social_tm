import os
import pandas as pd
import artm
from tm_utils import calc_hier, save_hier_model, get_df_scores, get_transform_params, calc_single, \
    create_artm_summary, timeit
from config import logger
logger.info("artm version", artm.version())


def make_experiment(expr, init_dir, need_transform, num_topics_list,
                    reg_dict_list, num_collection_passes,
                    need_save_model, data_path, odir, num_topics_list_str,
                    use_hier=True):
    df_info = pd.read_csv(r"{}/info/groups_info_names.csv".format(init_dir), sep='\t', encoding='utf-8')

    # transform
    if need_transform:
        transform_dict = get_transform_params(r"{}/vw/clients_posts.vw".format(init_dir), init_dir)
    else:
        transform_dict = None

    os.makedirs(odir, exist_ok=True)
    logger.info(odir)
    logger.info(f"num_topics_list {num_topics_list}")
    logger.info(f"experiment name {expr}")
    if use_hier:
        total_scores_list, hier, level_list = calc_hier(num_topics_list=num_topics_list,
                                                        expr=expr, odir=odir,
                                                        data_path=data_path,
                                                        reg_dict_list=reg_dict_list,
                                                        transform_dict=transform_dict,
                                                        topwords_count=60,
                                                        df_info=df_info,
                                                        num_collection_passes=num_collection_passes)
    else:
        if len(num_topics_list) == len(reg_dict_list):
            for num_topics, reg_dict in zip(num_topics_list, reg_dict_list):
                total_scores_list, hier, level_list = calc_single(num_topics=num_topics,
                                                                  expr=expr, odir=odir,
                                                                  data_path=data_path,
                                                                  reg_dict=reg_dict,
                                                                  transform_dict=transform_dict,
                                                                  topwords_count=60,
                                                                  df_info=df_info,
                                                                  num_collection_passes=num_collection_passes)
        else:
            raise Exception("num_topics_list and reg_dict_list have different length")

    # save all model
    if need_save_model:
        model_dir_path = f"{odir}{expr}_{num_topics_list_str}_hier.model"
        save_hier_model(hier, model_dir_path)

    df_scores = get_df_scores(total_scores_list)
    expr_desc = "{}_{}_{}".format(num_topics_list[0], num_topics_list[-1], bool(reg_dict_list[0]))
    df_scores.to_csv("{}../{}_levels_{}_scores.csv".format(odir, expr, expr_desc, index=False, sep='\t'))
    ofile_summary_path = r"{}summary_hier_{}.xlsx".format(odir, expr_desc)
    create_artm_summary(odir, ofile_summary_path, expr=expr)


@timeit
def main_hier():
    expr = 'clients_posts_sample_one'
    init_dir = r"../datasets"
    expr_list = [expr]
    need_save_model = True
    need_transform = True
    use_hier = False
    # reg_dict = dict(spars_theta_tau=-100)
    reg_dict_list = (None, None)
    # reg_dict = False
    num_topics_list = (32, 64)
    num_collection_passes = 20

    for expr in expr_list:
        # odir params
        data_path = r'{}/vw/{}.vw'.format(init_dir, expr)
        num_topics_list_str = "_".join(map(str, num_topics_list))
        odir = r"{}/output_summary_hier_tf/{}_{}_reg_{}/".format(init_dir,
                                                                 expr, num_topics_list_str,
                                                                 bool(reg_dict_list[0]))
        make_experiment(expr, init_dir, need_transform, num_topics_list,
                        reg_dict_list, num_collection_passes, need_save_model,
                        data_path, odir, num_topics_list_str, use_hier=use_hier)
    print("total elapsed")
