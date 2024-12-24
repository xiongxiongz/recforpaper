"""
Created on Nov 20, 2021
Updated on Apr 23, 2022
train SASRec demo
@author: Ziyao Geng(zggzy1996@163.com)
"""
import os

import pandas as pd
from absl import flags, app
from time import time
from tensorflow.keras.optimizers import Adam

from reclearn.models.matching import SASRec
from reclearn.data.datasets import movielens as ml
from reclearn.evaluator import eval_pos_neg
from data.utils.data_loader import DataGenerator

import datetime

FLAGS = flags.FLAGS

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Setting training parameters
flags.DEFINE_string("file_path", "data/ml-1m/ratings.dat", "file path.")
flags.DEFINE_string("train_path", "data/ml-1m/ml_seq_train.txt", "train path. If set to None, the program will split the dataset.")
flags.DEFINE_string("val_path", "data/ml-1m/ml_seq_val.txt", "val path.")
flags.DEFINE_string("test_path", "data/ml-1m/ml_seq_test.txt", "test path.")
flags.DEFINE_string("meta_path", "data/ml-1m/ml_seq_meta.txt", "meta path.")
flags.DEFINE_integer("item_dim", 64, "The size of item embedding dimension.")
flags.DEFINE_integer("user_dim", 50, "The size of user embedding dimension.")
flags.DEFINE_float("embed_reg", 0.0, "The value of embedding regularization.")
flags.DEFINE_integer("blocks", 2, "The Number of blocks.")
flags.DEFINE_integer("num_heads", 2, "The Number of attention heads.")
flags.DEFINE_integer("ffn_hidden_unit", 64, "Number of hidden unit in FFN.")
flags.DEFINE_float("dnn_dropout", 0.2, "Float between 0 and 1. Dropout of user and item MLP layer.")
flags.DEFINE_float("layer_norm_eps", 1e-6, "Small float added to variance to avoid dividing by zero.")
flags.DEFINE_boolean("use_l2norm", False, "Whether user embedding, item embedding should be normalized or not.")
flags.DEFINE_string("loss_name", "binary_cross_entropy_loss", "Loss Name.")
flags.DEFINE_float("gamma", 0.5, "If hinge_loss is selected as the loss function, you can specify the margin.")
flags.DEFINE_float("learning_rate", 0.001, "Learning rate.")
flags.DEFINE_integer("neg_num", 4, "The number of negative sample for each positive sample.")
flags.DEFINE_integer("seq_len", 100, "The length of user's behavior sequence.")
flags.DEFINE_integer("epochs", 20, "train steps.")
flags.DEFINE_integer("batch_size", 512, "Batch Size.")
flags.DEFINE_integer("test_neg_num", 100, "The number of test negative samples.")
flags.DEFINE_integer("k", 10, "recall k items at test stage.")
flags.DEFINE_integer("seed", None, "random seed.")

def main(argv):
    # TODO: 1. Split Data
    if FLAGS.train_path == "None":
        train_path, val_path, test_path, meta_path = ml.split_seq_data(file_path=FLAGS.file_path)
    else:
        train_path, val_path, test_path, meta_path = FLAGS.train_path, FLAGS.val_path, FLAGS.test_path, FLAGS.meta_path
    # with open(meta_path) as f:
    with open("data/ml1m/ml_seq_meta.txt") as f:
        max_user_num, max_item_num = [int(x) for x in f.readline().strip('\n').split('\t')]
    # TODO: 2. Load Sequence Data
    # train_data = ml.load_seq_data(train_path, "train", FLAGS.seq_len, FLAGS.neg_num, max_item_num)
    train_data = ml.load_txt_data("data/ml1m/ml1m.txt", "train", FLAGS.seq_len, FLAGS.neg_num, max_item_num)
    train_generator = DataGenerator(train_data, FLAGS.batch_size)
    # val_data = ml.load_seq_data(val_path, "val", FLAGS.seq_len, FLAGS.neg_num, max_item_num)
    val_data = ml.load_txt_data("data/ml1m/ml1m.txt", "val", FLAGS.seq_len, FLAGS.neg_num, max_item_num)
    val_generator = DataGenerator(val_data, FLAGS.batch_size)
    # test_data = ml.load_seq_data(test_path, "test", FLAGS.seq_len, FLAGS.test_neg_num, max_item_num)
    test_data = ml.load_txt_data("data/ml1m/ml1m.txt", "test", FLAGS.seq_len, FLAGS.test_neg_num, max_item_num)
    # TODO: 3. Set Model Hyper Parameters.
    model_params = {
        'item_num': max_item_num + 1,
        'user_num': max_user_num + 1,
        'item_dim': FLAGS.item_dim,
        'user_dim': FLAGS.user_dim,
        'seq_len': FLAGS.seq_len,
        'blocks': FLAGS.blocks,
        'num_heads': FLAGS.num_heads,
        'ffn_hidden_unit': FLAGS.ffn_hidden_unit,
        'dnn_dropout': FLAGS.dnn_dropout,
        'use_l2norm': FLAGS.use_l2norm,
        'loss_name': FLAGS.loss_name,
        'gamma': FLAGS.gamma,
        'embed_reg': FLAGS.embed_reg,
        'seed': FLAGS.seed
    }
    # 获取当前时间作为模型文件名后缀
    start_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")  # 格式：20241130_123456
    model_name = f"sasrec_{start_time}"
    # TODO: 4. Build Model
    model = SASRec(**model_params)
    model.compile(optimizer=Adam(learning_rate=FLAGS.learning_rate))
    # TODO: 5. Fit Model
    try:
        results = []
        for epoch in range(1, FLAGS.epochs + 1):
            t1 = time()
            model.fit(
                x=train_generator,
                epochs=1,
                validation_data=val_generator,
                use_multiprocessing=True,
                workers=4,
                # batch_size=FLAGS.batch_size
            )
            t2 = time()
            eval_dict = eval_pos_neg(model, test_data, ['hr', 'mrr', 'ndcg'], FLAGS.k)
            '''
            print('Iteration %d Fit [%.1f s], Evaluate [%.1f s]: HR = %.4f, MRR = %.4f, NDCG = %.4f'
                  % (epoch, t2 - t1, time() - t2, eval_dict['hr'], eval_dict['mrr'], eval_dict['ndcg']))
            '''
            # @10, @20, @40
            print('Iteration %d Fit [%.1f s], Evaluate [%.1f s]: HR_10 = %.4f, MRR@10 = %.4f, NDCG@10 = %.4f,'
                  ' HR@20 = %.4f, MRR@20 = %.4f, NDCG@20 = %.4f, HR@40 = %.4f, MRR@40 = %.4f, NDCG@40 = %.4f'
                  % (epoch, t2 - t1, time() - t2, eval_dict['hr'], eval_dict['mrr'], eval_dict['ndcg'], eval_dict['hr_20'], eval_dict['mrr_20'], eval_dict['ndcg_20'], eval_dict['hr_40'], eval_dict['mrr_40'], eval_dict['ndcg_40']))
            results.append([epoch, t2 - t1, time() - t2, eval_dict['hr'], eval_dict['mrr'], eval_dict['ndcg'], eval_dict['hr_20'], eval_dict['mrr_20'], eval_dict['ndcg_20'], eval_dict['hr_40'], eval_dict['mrr_40'], eval_dict['ndcg_40']])
        # write logs
        pd.DataFrame(results, columns=['Iteration', 'fit_time', 'evaluate_time', 'hr@10', 'mrr@10', 'ndcg@10', 'hr@20', 'mrr@20', 'ndcg@20', 'hr@40', 'mrr@40', 'ndcg@40']).\
            to_csv("logs/SASRec_log_{}_maxlen_{}_dim_{}_blocks_{}_heads_{}.csv".format(start_time, FLAGS.seq_len, FLAGS.embed_dim, FLAGS.blocks, FLAGS.num_heads), index=False)
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        model.save(model_name, save_format='tf')


if __name__ == '__main__':
    app.run(main)