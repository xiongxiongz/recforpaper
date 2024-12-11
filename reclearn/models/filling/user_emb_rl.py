from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Dense
from tensorflow.keras.optimizers import Adam
from reclearn.evaluator import eval_rank


def cal_rl_loss(pred, metric_names, k=10):
    pred_y = - pred
    return eval_rank(pred_y, metric_names, k)


def rl_loss(logits):
    res_dict = cal_rl_loss(logits, ['hr', 'mrr', 'ndcg'], 10)
    return res_dict['ndcg']


class UserEmbeddingRL:
    def __init__(self, embedding_dim):
        self.embedding_dim = embedding_dim
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(InputLayer(input_shape=self.embedding_dim))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(self.embedding_dim, activation='linear'))
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        return model

    def adjust_embedding(self, user_embeddings):
        # 调整嵌入矩阵
        delta_u = self.model.predict(user_embeddings)
        return user_embeddings + delta_u

