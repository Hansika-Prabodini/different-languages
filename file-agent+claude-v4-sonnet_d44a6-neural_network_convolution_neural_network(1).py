"""
 - - - - - -- - - - - - - - - - - - - - - - - - - - - - -
Name - - CNN - Convolution Neural Network For Photo Recognizing
Goal - - Recognize Handwriting Word Photo
Detail: Total 5 layers neural network
        * Convolution layer
        * Pooling layer
        * Input layer layer of BP
        * Hidden layer of BP
        * Output layer of BP
Author: Stephen Lee
Github: 245885195@qq.com
Date: 2017.9.20
- - - - - -- - - - - - - - - - - - - - - - - - - - - - -
"""

import pickle

import numpy as np
from matplotlib import pyplot as plt


class CNN:
    def __init__(
        self, conv1_get, size_p1, bp_num1, bp_num2, bp_num3, rate_w=0.2, rate_t=0.2
    ):
        """
        :param conv1_get: [a,c,d], size, number, step of convolution kernel
        :param size_p1: pooling size
        :param bp_num1: units number of flatten layer
        :param bp_num2: units number of hidden layer
        :param bp_num3: units number of output layer
        :param rate_w: rate of weight learning
        :param rate_t: rate of threshold learning
        """
        self.num_bp1 = bp_num1
        self.num_bp2 = bp_num2
        self.num_bp3 = bp_num3
        self.conv1 = conv1_get[:2]
        self.step_conv1 = conv1_get[2]
        self.size_pooling1 = size_p1
        self.rate_weight = rate_w
        self.rate_thre = rate_t
        rng = np.random.default_rng()
        # Use numpy arrays instead of matrices for better performance
        self.w_conv1 = rng.random((self.conv1[1], self.conv1[0], self.conv1[0])) - 0.5
        self.wkj = rng.random((self.num_bp3, self.num_bp2)) - 0.5
        self.vji = rng.random((self.num_bp2, self.num_bp1)) - 0.5
        self.thre_conv1 = 2 * rng.random(self.conv1[1]) - 1
        self.thre_bp2 = 2 * rng.random(self.num_bp2) - 1
        self.thre_bp3 = 2 * rng.random(self.num_bp3) - 1

    def save_model(self, save_path):
        # save model dict with pickle
        model_dic = {
            "num_bp1": self.num_bp1,
            "num_bp2": self.num_bp2,
            "num_bp3": self.num_bp3,
            "conv1": self.conv1,
            "step_conv1": self.step_conv1,
            "size_pooling1": self.size_pooling1,
            "rate_weight": self.rate_weight,
            "rate_thre": self.rate_thre,
            "w_conv1": self.w_conv1,
            "wkj": self.wkj,
            "vji": self.vji,
            "thre_conv1": self.thre_conv1,
            "thre_bp2": self.thre_bp2,
            "thre_bp3": self.thre_bp3,
        }
        with open(save_path, "wb") as f:
            pickle.dump(model_dic, f)

        print(f"Model saved: {save_path}")

    @classmethod
    def read_model(cls, model_path):
        # read saved model
        with open(model_path, "rb") as f:
            model_dic = pickle.load(f)  # noqa: S301

        conv_get = model_dic.get("conv1")
        conv_get.append(model_dic.get("step_conv1"))
        size_p1 = model_dic.get("size_pooling1")
        bp1 = model_dic.get("num_bp1")
        bp2 = model_dic.get("num_bp2")
        bp3 = model_dic.get("num_bp3")
        r_w = model_dic.get("rate_weight")
        r_t = model_dic.get("rate_thre")
        # create model instance
        conv_ins = CNN(conv_get, size_p1, bp1, bp2, bp3, r_w, r_t)
        # modify model parameter - ensure compatibility with array format
        w_conv1_loaded = model_dic.get("w_conv1")
        if isinstance(w_conv1_loaded, list):
            # Convert old matrix format to new array format
            conv_ins.w_conv1 = np.array([np.array(w) for w in w_conv1_loaded])
        else:
            conv_ins.w_conv1 = w_conv1_loaded
            
        conv_ins.wkj = np.array(model_dic.get("wkj"))
        conv_ins.vji = np.array(model_dic.get("vji"))
        conv_ins.thre_conv1 = np.array(model_dic.get("thre_conv1"))
        conv_ins.thre_bp2 = np.array(model_dic.get("thre_bp2"))
        conv_ins.thre_bp3 = np.array(model_dic.get("thre_bp3"))
        return conv_ins

    def sig(self, x):
        return 1 / (1 + np.exp(-1 * x))

    def do_round(self, x):
        return round(x, 3)

    def convolute(self, data, convs, w_convs, thre_convs, conv_step):
        # convolution process
        size_conv = convs[0]
        num_conv = convs[1]
        size_data = data.shape[0]
        size_feature_map = (size_data - size_conv) // conv_step + 1
        
        # Pre-allocate arrays for better memory efficiency
        data_featuremap = np.zeros((num_conv, size_feature_map, size_feature_map))
        
        # Vectorized convolution calculation
        for i_map in range(num_conv):
            for i in range(size_feature_map):
                for j in range(size_feature_map):
                    i_start = i * conv_step
                    j_start = j * conv_step
                    patch = data[i_start:i_start + size_conv, j_start:j_start + size_conv]
                    net_focus = np.sum(patch * w_convs[i_map]) - thre_convs[i_map]
                    data_featuremap[i_map, i, j] = self.sig(net_focus)
        
        # Create focus_list more efficiently
        focus_patches = []
        for i in range(0, size_data - size_conv + 1, conv_step):
            for j in range(0, size_data - size_conv + 1, conv_step):
                patch = data[i:i + size_conv, j:j + size_conv]
                focus_patches.append(patch.flatten())
        
        focus_list = np.array(focus_patches)
        return focus_list, [data_featuremap[i] for i in range(num_conv)]

    def pooling(self, featuremaps, size_pooling, pooling_type="average_pool"):
        # pooling process
        size_map = featuremaps[0].shape[0]
        size_pooled = size_map // size_pooling
        featuremap_pooled = []
        
        for feature_map in featuremaps:
            # Pre-allocate result array
            map_pooled = np.zeros((size_pooled, size_pooled))
            
            for i in range(size_pooled):
                for j in range(size_pooled):
                    i_start = i * size_pooling
                    j_start = j * size_pooling
                    pool_region = feature_map[i_start:i_start + size_pooling, 
                                            j_start:j_start + size_pooling]
                    
                    if pooling_type == "average_pool":
                        map_pooled[i, j] = np.mean(pool_region)
                    elif pooling_type == "max_pooling":
                        map_pooled[i, j] = np.max(pool_region)
            
            featuremap_pooled.append(map_pooled)
        return featuremap_pooled

    def _expand(self, data):
        # expanding three dimension data to one dimension array - more efficient
        if isinstance(data, list):
            # Convert list of arrays to a single flattened array
            return np.concatenate([arr.flatten() for arr in data])
        else:
            return data.flatten()

    def _expand_mat(self, data_mat):
        # expanding matrix to one dimension array - more efficient
        return data_mat.flatten().reshape(1, -1)

    def _calculate_gradient_from_pool(
        self, out_map, pd_pool, num_map, size_map, size_pooling
    ):
        """
        calculate the gradient from the data slice of pool layer
        pd_pool: list of matrix
        out_map: the shape of data slice(size_map*size_map)
        return: pd_all: list of matrix, [num, size_map, size_map]
        """
        pd_all = []
        pd_pool = np.array(pd_pool)
        pooled_size = size_map // size_pooling
        
        i_pool = 0
        for i_map in range(num_map):
            # More efficient gradient calculation
            pd_conv1 = np.zeros((size_map, size_map))
            
            for i in range(pooled_size):
                for j in range(pooled_size):
                    i_start = i * size_pooling
                    j_start = j * size_pooling
                    pd_conv1[i_start:i_start + size_pooling, 
                            j_start:j_start + size_pooling] = pd_pool[i_pool]
                    i_pool += 1
            
            # Vectorized multiplication
            out_map_i = out_map[i_map]
            pd_conv2 = pd_conv1 * out_map_i * (1 - out_map_i)
            pd_all.append(pd_conv2)
        return pd_all

    def train(
        self, patterns, datas_train, datas_teach, n_repeat, error_accuracy, draw_e=bool
    ):
        # model training
        print("----------------------Start Training-------------------------")
        print((" - - Shape: Train_Data  ", np.shape(datas_train)))
        print((" - - Shape: Teach_Data  ", np.shape(datas_teach)))
        rp = 0
        all_mse = []
        mse = 10000
        while rp < n_repeat and mse >= error_accuracy:
            error_count = 0
            print(f"-------------Learning Time {rp}--------------")
            for p in range(len(datas_train)):
                # print('------------Learning Image: %d--------------'%p)
                data_train = np.array(datas_train[p])
                data_teach = np.array(datas_teach[p])
                data_focus1, data_conved1 = self.convolute(
                    data_train,
                    self.conv1,
                    self.w_conv1,
                    self.thre_conv1,
                    conv_step=self.step_conv1,
                )
                data_pooled1 = self.pooling(data_conved1, self.size_pooling1)
                num_feature_maps = len(data_conved1)
                feature_map_size = data_conved1[0].shape[0]
                
                data_bp_input = self._expand(data_pooled1)
                bp_out1 = data_bp_input

                bp_net_j = np.dot(bp_out1, self.vji.T) - self.thre_bp2
                bp_out2 = self.sig(bp_net_j)
                bp_net_k = np.dot(bp_out2, self.wkj.T) - self.thre_bp3
                bp_out3 = self.sig(bp_net_k)

                # --------------Model Leaning ------------------------
                # calculate error and gradient---------------
                error_diff = data_teach - bp_out3
                pd_k_all = error_diff * bp_out3 * (1 - bp_out3)
                pd_j_all = np.dot(pd_k_all, self.wkj) * bp_out2 * (1 - bp_out2)
                pd_i_all = np.dot(pd_j_all, self.vji)

                pd_conv1_pooled = pd_i_all / (self.size_pooling1 * self.size_pooling1)
                pd_conv1_all = self._calculate_gradient_from_pool(
                    data_conved1,
                    pd_conv1_pooled.flatten().tolist(),
                    num_feature_maps,
                    feature_map_size,
                    self.size_pooling1,
                )
                # weight and threshold learning process---------
                # convolution layer - more efficient weight updates
                for k_conv in range(self.conv1[1]):
                    pd_conv_flat = pd_conv1_all[k_conv].flatten()
                    delta_w = self.rate_weight * np.outer(pd_conv_flat, data_focus1.flatten())
                    delta_w = delta_w.sum(axis=1).reshape(self.conv1[0], self.conv1[0])
                    
                    self.w_conv1[k_conv] += delta_w
                    self.thre_conv1[k_conv] -= np.sum(pd_conv1_all[k_conv]) * self.rate_thre
                
                # all connected layer - use vectorized operations
                self.wkj += np.outer(pd_k_all, bp_out2) * self.rate_weight
                self.vji += np.outer(pd_j_all, bp_out1) * self.rate_weight
                self.thre_bp3 -= pd_k_all * self.rate_thre
                self.thre_bp2 -= pd_j_all * self.rate_thre
                
                # calculate the sum error of all single image
                errors = np.sum(np.abs(error_diff))
                error_count += errors
                # print('   ----Teach      ',data_teach)
                # print('   ----BP_output  ',bp_out3)
            rp = rp + 1
            mse = error_count / patterns
            all_mse.append(mse)

        def draw_error():
            yplot = [error_accuracy for i in range(int(n_repeat * 1.2))]
            plt.plot(all_mse, "+-")
            plt.plot(yplot, "r--")
            plt.xlabel("Learning Times")
            plt.ylabel("All_mse")
            plt.grid(True, alpha=0.5)
            plt.show()

        print("------------------Training Complete---------------------")
        print((" - - Training epoch: ", rp, f"     - - Mse: {mse:.6f}"))
        if draw_e:
            draw_error()
        return mse

    def predict(self, datas_test):
        # model predict
        print("-------------------Start Testing-------------------------")
        print((" - - Shape: Test_Data  ", np.shape(datas_test)))
        
        # Pre-allocate output array for better performance
        produce_out = np.zeros((len(datas_test), self.num_bp3))
        
        for p in range(len(datas_test)):
            data_test = np.array(datas_test[p])
            data_focus1, data_conved1 = self.convolute(
                data_test,
                self.conv1,
                self.w_conv1,
                self.thre_conv1,
                conv_step=self.step_conv1,
            )
            data_pooled1 = self.pooling(data_conved1, self.size_pooling1)
            data_bp_input = self._expand(data_pooled1)

            bp_out1 = data_bp_input
            bp_net_j = np.dot(bp_out1, self.vji.T) - self.thre_bp2
            bp_out2 = self.sig(bp_net_j)
            bp_net_k = np.dot(bp_out2, self.wkj.T) - self.thre_bp3
            bp_out3 = self.sig(bp_net_k)
            produce_out[p] = bp_out3
            
        # Vectorized rounding operation
        return np.round(produce_out, 3)

    def convolution(self, data):
        # return the data of image after convoluting process so we can check it out
        data_test = np.array(data)
        data_focus1, data_conved1 = self.convolute(
            data_test,
            self.conv1,
            self.w_conv1,
            self.thre_conv1,
            conv_step=self.step_conv1,
        )
        data_pooled1 = self.pooling(data_conved1, self.size_pooling1)

        return data_conved1, data_pooled1


if __name__ == "__main__":
    """
    I will put the example in another file
    """
