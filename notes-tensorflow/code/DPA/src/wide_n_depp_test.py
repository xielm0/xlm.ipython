# -*- coding:utf-8 -*-
import pandas as pd
import tensorflow as tf
import tempfile
import numpy as np

import argparse
import sys

LABEL = ['label']
FEATURE_COLUM = ["sex", "age", "carrer", "marriage", "haschild", "province", "city", "jd_lev", "sku_jd_prc",
                 "sku_jd_prc_after", "sku_jd_prc_rate", "cate_id_1st", "cate_id_2nd", "cate_id_3rd", "brand_id",
                 "dim_item_fin_cate_name", "data_type", "len", "width", "height", "calc_volume", "wt", "sku_comment_nums",
                 "sku_comment_score", "sku_comment_good_nums", "sku_comment_good_rate", "sku_comment_bad_nums",
                 "sku_comment_bad_rate"]

CATEGORICAL_COLUMNS = ["sex", "carrer", "marriage", "haschild", "province", "city", "jd_lev", "cate_id_1st", "cate_id_2nd", "cate_id_3rd",
                       "brand_id", "dim_item_fin_cate_name", "data_type"]

CONTINUOUS_COLUMNS = ["age", "sku_jd_prc", "sku_jd_prc_after", "sku_jd_prc_rate", "len", "width", "height",
                      "calc_volume", "wt", "sku_comment_nums", "sku_comment_score",
                      "sku_comment_good_nums", "sku_comment_good_rate",
                      "sku_comment_bad_nums", "sku_comment_bad_rate"]

TRAIN_PATH = "type=basic_train/part-00000-146b62e9-069c-4124-9197-474937c16382.csv"
TEST_PATH = "type=basic_test/part-00000-25db03b1-98c9-4152-b9c7-b60cbd174ac0.csv"




def build_estimator(model_dir, model_type):
    # Sparse base columns.
    sex = tf.contrib.layers.sparse_column_with_keys(column_name="sex", keys=['-1', '2', '0', '1'])
    carrer = tf.contrib.layers.sparse_column_with_hash_bucket(column_name="carrer", hash_bucket_size=15)
    marriage = tf.contrib.layers.sparse_column_with_hash_bucket(column_name="marriage", hash_bucket_size=3)
    haschild = tf.contrib.layers.sparse_column_with_hash_bucket(column_name="haschild", hash_bucket_size=3)
    province = tf.contrib.layers.sparse_column_with_hash_bucket(column_name="province", hash_bucket_size=35)
    city = tf.contrib.layers.sparse_column_with_hash_bucket(column_name="city", hash_bucket_size=400)
    jd_lev = tf.contrib.layers.sparse_column_with_hash_bucket(column_name="jd_lev", hash_bucket_size=8)
    cate_id_1st = tf.contrib.layers.sparse_column_with_hash_bucket(column_name="cate_id_1st", hash_bucket_size=100)
    cate_id_2nd = tf.contrib.layers.sparse_column_with_hash_bucket(column_name="cate_id_2nd", hash_bucket_size=500)
    cate_id_3rd = tf.contrib.layers.sparse_column_with_hash_bucket(column_name="cate_id_3rd", hash_bucket_size=1000)
    brand_id = tf.contrib.layers.sparse_column_with_hash_bucket(column_name="brand_id", hash_bucket_size=18000)
    dim_item_fin_cate_name = tf.contrib.layers.sparse_column_with_hash_bucket(column_name="dim_item_fin_cate_name", hash_bucket_size=10)
    data_type = tf.contrib.layers.sparse_column_with_hash_bucket(column_name="data_type", hash_bucket_size=10)

    # Continuous base columns.
    age = tf.contrib.layers.real_valued_column("age")
    sku_jd_prc = tf.contrib.layers.real_valued_column("sku_jd_prc")
    sku_jd_prc_after = tf.contrib.layers.real_valued_column("sku_jd_prc_after")
    sku_jd_prc_rate = tf.contrib.layers.real_valued_column("sku_jd_prc_rate")
    len = tf.contrib.layers.real_valued_column("len")
    width = tf.contrib.layers.real_valued_column("width")
    height = tf.contrib.layers.real_valued_column("height")
    calc_volume = tf.contrib.layers.real_valued_column("calc_volume")
    wt = tf.contrib.layers.real_valued_column("wt")
    sku_comment_nums = tf.contrib.layers.real_valued_column("sku_comment_nums")
    sku_comment_score = tf.contrib.layers.real_valued_column("sku_comment_score")
    sku_comment_good_nums = tf.contrib.layers.real_valued_column("sku_comment_good_nums")
    sku_comment_good_rate = tf.contrib.layers.real_valued_column("sku_comment_good_rate")
    sku_comment_bad_nums = tf.contrib.layers.real_valued_column("sku_comment_bad_nums")
    sku_comment_bad_rate = tf.contrib.layers.real_valued_column("sku_comment_bad_rate")

    # Transformations.
    age_buckets = tf.contrib.layers.bucketized_column(age,
                                                      boundaries=[
                                                          0, 18, 25, 30, 35, 40, 45,
                                                          50, 55, 60, 65
                                                      ]) # hash_bucket_size=12

    # Wide columns and deep columns.
    wide_columns = [sex, carrer, age_buckets, marriage, haschild, province, city, jd_lev, cate_id_1st, cate_id_2nd,
                    cate_id_3rd, brand_id, dim_item_fin_cate_name, data_type,
                    tf.contrib.layers.crossed_column([carrer, haschild],
                                                     hash_bucket_size=60),
                    tf.contrib.layers.crossed_column([sex, carrer],
                                                     hash_bucket_size=60),
                    tf.contrib.layers.crossed_column([sex, age_buckets],
                                                     hash_bucket_size=50),
                    tf.contrib.layers.crossed_column([age_buckets, marriage],
                                                     hash_bucket_size=50),
                    tf.contrib.layers.crossed_column([age_buckets, haschild],
                                                     hash_bucket_size=40),
                    tf.contrib.layers.crossed_column([sex, province],
                                                     hash_bucket_size=140),
                    tf.contrib.layers.crossed_column([sex, jd_lev],
                                                     hash_bucket_size=35),
                    tf.contrib.layers.crossed_column([province, haschild],
                                                     hash_bucket_size=110),
                    tf.contrib.layers.crossed_column([sex, cate_id_1st],
                                                     hash_bucket_size=400),
                    tf.contrib.layers.crossed_column([sex, cate_id_3rd],
                                                     hash_bucket_size=4000),
                    tf.contrib.layers.crossed_column([sex, brand_id],
                                                     hash_bucket_size=70000),
                    tf.contrib.layers.crossed_column([sex, dim_item_fin_cate_name],
                                                     hash_bucket_size=40),
                    tf.contrib.layers.crossed_column([age_buckets, dim_item_fin_cate_name],
                                                     hash_bucket_size=120),
                    ]

    deep_columns = [sex, carrer, age_buckets, marriage, haschild, province, jd_lev,
                    dim_item_fin_cate_name, data_type, sku_jd_prc, sku_jd_prc_after, sku_jd_prc_rate,
                    len, width, height, calc_volume, wt, sku_comment_nums, sku_comment_score, sku_comment_good_nums,
                    sku_comment_good_rate, sku_comment_bad_nums, sku_comment_bad_rate,
                    tf.contrib.layers.embedding_column(city, dimension=32),
                    tf.contrib.layers.embedding_column(cate_id_1st, dimension=32),
                    tf.contrib.layers.embedding_column(cate_id_2nd, dimension=32),
                    tf.contrib.layers.embedding_column(cate_id_3rd, dimension=32),
                    tf.contrib.layers.embedding_column(brand_id, dimension=64),
                    ]

    if model_type == "wide":
        m = tf.contrib.learn.LinearClassifier(model_dir=model_dir,
                                              feature_columns=wide_columns)

    elif model_type == "deep":
        m = tf.contrib.learn.DNNClassifier(model_dir=model_dir,
                                           feature_columns=deep_columns,
                                           hidden_units=[100, 50])
    else:
        m = tf.contrib.learn.DNNLinearCombinedClassifier(
            n_classes=2,
            model_dir=model_dir,
            linear_feature_columns=wide_columns,
            linear_optimizer=tf.train.FtrlOptimizer(0.01),
            dnn_feature_columns=deep_columns,
            dnn_hidden_units=[512, 256, 128],
            dnn_activation_fn=tf.nn.relu,
            dnn_dropout=0.8,
            dnn_optimizer=tf.train.AdagradOptimizer(0.01),
            fix_global_step_increment_bug=True,)
    return m


def input_fn(df):
    continuous_cols = {k: tf.constant(df[k].values) for k in CONTINUOUS_COLUMNS}

    categorical_cols = {
        k: tf.SparseTensor(
            indices=[[i, 0] for i in range(df[k].size)],
            values=df[k].values,
            dense_shape=[df[k].size, 1])
        for k in CATEGORICAL_COLUMNS}
    feature_cols = dict(continuous_cols)
    feature_cols.update(categorical_cols)
    # Converts the label column into a constant Tensor.
    label = tf.constant(df['label'].values)
    # Returns the feature columns and the label.
    return feature_cols, label


def train_and_eval(model_dir, model_type, train_steps, train_data='', test_data=''):
    df_train = pd.read_csv(
        tf.gfile.Open(TRAIN_PATH),
        names=LABEL + FEATURE_COLUM,
        skipinitialspace=True,
        engine="python")
    df_train['sex'] = df_train['sex'].astype(np.str)

    df_test = pd.read_csv(
        tf.gfile.Open(TEST_PATH),
        names=LABEL + FEATURE_COLUM,
        skipinitialspace=True,
        # skiprows=1,
        engine="python")
    df_test['sex'] = df_test['sex'].astype(np.str)


    # remove NaN elements
    df_train = df_train.dropna(how='any', axis=0)
    df_test = df_test.dropna(how='any', axis=0)

    model_dir = tempfile.mkdtemp() if not model_dir else model_dir
    print("model directory = %s" % model_dir)

    m = build_estimator(model_dir, model_type)
    m.fit(input_fn=lambda: input_fn(df_train), steps=train_steps)
    results = m.evaluate(input_fn=lambda: input_fn(df_test), steps=1)

    for key in sorted(results):
        print("%s: %s" % (key, results[key]))


FLAGS = None


def main(_):
    train_and_eval(FLAGS.model_dir, FLAGS.model_type, FLAGS.train_steps,
                   FLAGS.train_data, FLAGS.test_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="",
        help="Base directory for output models."
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="wide_n_deep",
        help="Valid model types: {'wide', 'deep', 'wide_n_deep'}."
    )
    parser.add_argument(
        "--train_steps",
        type=int,
        default=200,
        help="Number of training steps."
    )
    parser.add_argument(
        "--train_data",
        type=str,
        default="",
        help="Path to the training data."
    )
    parser.add_argument(
        "--test_data",
        type=str,
        default="",
        help="Path to the test data."
    )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)