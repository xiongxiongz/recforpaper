import numpy as np
from tensorflow.keras.optimizers import Adam
import os
from time import time
import datetime
import pandas as pd
from tqdm import tqdm
from reclearn.models.matching import SASRec
from reclearn.data.datasets import movielens as ml
from reclearn.evaluator import eval_pos_neg
from reclearn.models.filling.dqn import DQNAgent
from data.utils.data_loader import DataGenerator


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

class SASRecDQNEnv:
    def __init__(self, sasrec_model, dataset, batch_size=512):
        self.sasrec_model = sasrec_model  # SASRec model
        self.dataset = dataset  # Training dataset
        self.current_sequence = None
        self.current_data_index = None
        self.action_map = self._build_action_map(len(dataset['click_seq'][0]))
        self.batch_size = batch_size

    def _build_action_map(self, sequence_length):
        """Construct a map of action indices to (i, j) swaps."""
        action_map = []
        for i in range(sequence_length - 1):
            action_map.append((i, i+1))
        return action_map

    def reset(self):
        self.current_data_index = 100
        self.current_sequence = self.dataset['click_seq'][self.current_data_index]
        return self.current_sequence

    def step(self, action):
        original_data_seqs, original_pos, original_neg= [], [], []
        new_data_seqs = []
        seq = self.current_sequence.copy()
        i, j = self.action_map[action]
        seq[i], seq[j] = seq[j], seq[i]  # Swap two movies
        original_data_seqs.append(self.current_sequence)
        new_data_seqs.append(seq)
        original_pos.append(self.dataset['pos_item'][self.current_data_index])
        original_neg.append(self.dataset['neg_item'][self.current_data_index])
        original_data = {'click_seq': np.array(original_data_seqs), 'pos_item': np.array(original_pos), 'neg_item': np.array(original_neg)}
        new_data = {'click_seq': np.array(new_data_seqs), 'pos_item': np.array(original_pos), 'neg_item': np.array(original_neg)}
        original_metrics = self.evaluate(original_data)
        new_metrics = self.evaluate(new_data)

        reward = (new_metrics["HR@k"] - original_metrics["HR@k"]) + \
                 (new_metrics["NDCG@k"] - original_metrics["NDCG@k"])

        self.current_sequence = seq
        return seq, reward * 10, new_data_seqs, original_pos, original_neg

    def evaluate(self, data, batch_size=1):
        pred_y = - self.sasrec_model.predict(data, batch_size)
        rank = pred_y.argsort().argsort()[:, 0]
        hr = self.compute_hr(rank)
        ndcg = self.compute_ndcg(rank)
        return {"HR@k": hr, "NDCG@k": ndcg}

    def compute_hr(self, rank, k=10):
        res = 0.0
        for r in rank:
            if r < k:
                res += 1
        return res / len(rank)

    def compute_ndcg(self, rank, k=10):
        res = 0.0
        for r in rank:
            if r < k:
                res += 1 / np.log2(r + 2)
        return res / len(rank)


if __name__ == "__main__":

    # tran super data
    k = 10
    learning_rate = 0.001
    batch_size = 32
    neg_num = 100
    test_neg_num = 100
    fine_tune_epochs = 20
    train_path = "/home/cqj/zzh/recforpaper/data/ml-1m/ml_seq_train.txt"
    val_path = "/home/cqj/zzh/recforpaper/data/ml-1m/ml_seq_val.txt"
    test_path = "/home/cqj/zzh/recforpaper/data/ml-1m/ml_seq_test.txt"
    meta_path = "/home/cqj/zzh/recforpaper/data/ml-1m/ml_seq_meta.txt"
    train_count_path = "/home/cqj/zzh/recforpaper/data/ml-1m/movie_view_count.txt"
    test_count_path = "/home/cqj/zzh/recforpaper/data/ml-1m/movie_view_count_test.txt"
    with open(meta_path) as f:
        max_user_num, max_item_num = [int(x) for x in f.readline().strip('\n').split('\t')]
    model_params = {
        'item_num': max_item_num + 1,
        'embed_dim': 64,
        'seq_len': 100,
        'blocks': 2,
        'num_heads': 2,
        'ffn_hidden_unit': 64,
        'dnn_dropout': 0.2,
        'use_l2norm': False,
        'loss_name': "binary_cross_entropy_loss",
        'gamma': 0.5
    }
    # Load Sequence Data
    train_data = ml.load_seq_data(train_path, train_count_path, "train", model_params['seq_len'], neg_num, max_item_num)
    train_generator = DataGenerator(train_data, batch_size)
    val_data = ml.load_seq_data(val_path, train_count_path, "val", model_params['seq_len'], neg_num, max_item_num)
    val_generator = DataGenerator(val_data, batch_size)
    test_data = ml.load_seq_data(test_path, test_count_path, "test", model_params['seq_len'], test_neg_num, max_item_num)
    # setup model
    fine_tune_model = SASRec(**model_params)
    fine_tune_model.load_weights('/home/cqj/zzh/recforpaper/sasrec_20241209_003918/variables/variables')
    fine_tune_model.compile(optimizer=Adam(learning_rate=learning_rate))

    env = SASRecDQNEnv(fine_tune_model, train_data, batch_size)
    state_size = model_params['seq_len']  # Length of the movie sequence
    action_size = state_size - 1  # All possible swaps
    agent = DQNAgent(state_size, action_size)

    episodes = 100   # 1000
    range_len = 50
    for e in range(episodes):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        for num in range(range_len):   #  500
            action = agent.act(state)
            next_state, reward, _, _, _ = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            done = num == (range_len-1)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                print(f"Episode: {e}/{episodes}, Reward: {reward}, Epsilon: {agent.epsilon}")
        if len(agent.memory) > 32:
            agent.replay(32)
    print("replay over")
    # Fine-tune SASRec with enhanced dataset
    fine_tune_seqs, fine_tune_pos, fine_tune_neg = [], [], []
    for _ in tqdm(range(16)):
        state = env.reset()
        state = np.reshape(state, [1, model_params['seq_len']])
        action = agent.act_greedy(state)  # 使用 DQN 选择动作
        _, _, current_seqs, current_pos, current_neg = env.step(action)
        fine_tune_seqs += current_seqs
        fine_tune_pos += current_pos
        fine_tune_neg += current_neg
    fine_tune_batch = 16
    fine_tune_train_data = {'click_seq': np.array(fine_tune_seqs), 'pos_item': np.array(fine_tune_pos), 'neg_item': np.array(fine_tune_neg)}
    fine_tune_train_generator = DataGenerator(fine_tune_train_data, fine_tune_batch)
    fine_tune_val_generator = DataGenerator(val_data, fine_tune_batch)
    # 获取当前时间作为模型文件名后缀
    start_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")  # 格式：20241130_123456
    model_name = f"sasrec_fine_tune_{start_time}"
    try:
        results = []
        for epoch in range(1, fine_tune_epochs + 1):
            t1 = time()
            fine_tune_model.fit(
                x=fine_tune_train_data,
                epochs=1,
                validation_data=fine_tune_val_generator,
                use_multiprocessing=True,
                workers=4
            )
            t2 = time()
            eval_dict = eval_pos_neg(fine_tune_model, test_data, ['hr', 'mrr', 'ndcg'], k, batch_size)
            # @10, @20, @40
            print('Iteration %d Fit [%.1f s], Evaluate [%.1f s]: HR_10 = %.4f, MRR@10 = %.4f, NDCG@10 = %.4f,'
                  ' HR@20 = %.4f, MRR@20 = %.4f, NDCG@20 = %.4f, HR@40 = %.4f, MRR@40 = %.4f, NDCG@40 = %.4f'
                  % (epoch, t2 - t1, time() - t2, eval_dict['hr'], eval_dict['mrr'], eval_dict['ndcg'], eval_dict['hr_20'], eval_dict['mrr_20'], eval_dict['ndcg_20'], eval_dict['hr_40'], eval_dict['mrr_40'], eval_dict['ndcg_40']))
            results.append([epoch, t2 - t1, time() - t2, eval_dict['hr'], eval_dict['mrr'], eval_dict['ndcg'], eval_dict['hr_20'], eval_dict['mrr_20'], eval_dict['ndcg_20'], eval_dict['hr_40'], eval_dict['mrr_40'], eval_dict['ndcg_40']])
        # write logs
        pd.DataFrame(results, columns=['Iteration', 'fit_time', 'evaluate_time', 'hr@10', 'mrr@10', 'ndcg@10', 'hr@20', 'mrr@20', 'ndcg@20', 'hr@40', 'mrr@40', 'ndcg@40']).\
            to_csv("logs/SASRec_fine_tune_log_{}_maxlen_{}_dim_{}_blocks_{}_heads_{}.csv".format(start_time, model_params['seq_len'], model_params['embed_dim'], model_params['blocks'], model_params['num_heads']), index=False)
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        fine_tune_model.save(model_name, save_format='tf')
