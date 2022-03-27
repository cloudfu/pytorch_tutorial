import torch
from torch.utils import data
from enum import Enum

class ExampleDataType(Enum):
    linear = 1
    sin = 2

class base_model(object):

    def __init__(self,name="base_model"):
        self.nn_name = name
        self.nn_model = ""
        self.loss_function = ""
        self.optimizer = ""
        self.learning_rate = 0.03

        self.example_data = ""
        self.data_iter = ""

        self.__init_nn__()
        self.__init_loss_function__()
        self.__init_optimizer__()

    def __init_nn__(self):
        pass

    def __init_loss_function__(self):
        pass

    def __init_optimizer__(self):
        pass

    def __generate_data__(self):
        pass        

    def get_nn_model(self):
        return self.nn_model

    def get_loss_function(self):
        return self.loss_function

    def get_optimizer(self):
        return self.optimizer

    def train():
        pass

    def load_batch_data(self,data_arrays,batch_size,is_shuffle=True):
        dataset = data.TensorDataset(*data_arrays)
        return data.DataLoader(dataset,batch_size,is_shuffle)

    def save(self):
        torch.save(self.nn_model, "./save_model/"+self.nn_name+".pkl")

    def load(self):
        self.nn_model = torch.load("./save_model/"+self.nn_name+".pkl")

    def train():
        pass


