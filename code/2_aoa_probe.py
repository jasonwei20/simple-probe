import config, utils_common
import numpy as np
np.random.seed(42)
from tqdm import tqdm
import gc
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from statistics import mean

def get_x_y(word_to_embedding, word_to_aoa):
    n_total = len(word_to_embedding)
    x = []
    y = []
    for word, embedding in word_to_embedding.items():
        assert word in word_to_aoa
        aoa = word_to_aoa[word]
        x.append(embedding)
        y.append(aoa)
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=42)
    return x_train, x_test, y_train, y_test

def eval_linear_regression(x_train, x_test, y_train, y_test):
    real_probe = LinearRegression().fit(x_train, y_train)
    y_train_random = np.array(y_train, copy=True); np.random.shuffle(y_train_random)
    control_probe = LinearRegression().fit(x_train, y_train_random)
    y_mean_control = np.mean(y_train)
    y_test_mean_control = np.zeros(y_test.shape)
    y_test_mean_control.fill(y_mean_control)

    def compute_acc(y_test, y_test_predict, margin_1=1.0, margin_2=2.0):
        
        diff_list = []
        within_margin_1_list = []
        within_margin_2_list = []
        for i in range(y_test.shape[0]):
            diff = abs(y_test[i] - y_test_predict[i])
            diff_list.append(diff)
            if diff < margin_1:
                within_margin_1_list.append(1)
            else:
                within_margin_1_list.append(0)
            if diff < margin_2:
                within_margin_2_list.append(1)
            else:
                within_margin_2_list.append(0)
        mean_diff = mean(diff_list)
        mean_correct_1 = mean(within_margin_1_list)
        mean_correct_2 = mean(within_margin_2_list)
        return mean_diff, mean_correct_1, mean_correct_2

    def evaluate_acc(probe, probe_name, x_train, x_test, y_train, y_test):

        score = probe.score(x_train, y_train)
        y_test_predict = probe.predict(x_test)
        mean_diff, mean_correct_1, mean_correct_2 = compute_acc(y_test, y_test_predict)
        print(f"{probe_name}\t score: {score:.3f} \tmean_diff: {mean_diff:.3f} \tmean_correct_1: {mean_correct_1:.3f} \tmean_correct_2: {mean_correct_2:.3f}")

    mean_diff, mean_correct_1, mean_correct_2 = compute_acc(y_test, y_test_mean_control)
    print(f"Mean control\t\t\tmean_diff: {mean_diff:.3f} \tmean_correct_1: {mean_correct_1:.3f} \tmean_correct_2: {mean_correct_2:.3f}")

    evaluate_acc(control_probe, "Control Probe", x_train, x_test, y_train, y_test)
    evaluate_acc(real_probe, "Real Probe", x_train, x_test, y_train, y_test)


if __name__ == "__main__":
    word_to_aoa = utils_common.load_pickle(config.aoa_dict_path)
    word_to_embedding = utils_common.load_pickle(config.aoa_embedding_path)
    print(f"AoA ranges from {min(word_to_aoa.values())} to {max(word_to_aoa.values())} with mean {mean(word_to_aoa.values()):.2f}")

    x_train, x_test, y_train, y_test = get_x_y(word_to_embedding, word_to_aoa)
    eval_linear_regression(x_train, x_test, y_train, y_test)