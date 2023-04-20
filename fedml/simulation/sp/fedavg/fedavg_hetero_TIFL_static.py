import copy
import logging
import random

import numpy as np
import torch
import torch.nn as nn
from torchsummary import summary
import wandb

from fedml import mlops
from fedml.ml.trainer.trainer_creator import create_model_trainer
from .client import Client

import os,sys
import subprocess
import glob
from os import path

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


class CNN(torch.nn.Module):

    def __init__(self,perf_coeff, only_digits=True):

        super(CNN, self).__init__()
        self.conv2d_1 = torch.nn.Conv2d(1,int(32*perf_coeff), kernel_size=3)
        self.max_pooling = nn.MaxPool2d(2, stride=2)
        self.conv2d_2 = torch.nn.Conv2d(int(perf_coeff*32),int( perf_coeff*64), kernel_size=3)
        self.dropout_1 = nn.Dropout(0.25)
        self.flatten = nn.Flatten()
        self.linear_1 = nn.Linear(int(perf_coeff*9216), int(perf_coeff*128))
        self.dropout_2 = nn.Dropout(0.5)
        self.linear_2 = nn.Linear(int(perf_coeff*128), 10 if only_digits else 62)
        self.relu = nn.ReLU()
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = torch.unsqueeze(x, 1)
        x = self.conv2d_1(x)
        x = self.relu(x)
        x = self.conv2d_2(x)
        x = self.relu(x)
        x = self.max_pooling(x)
        x = self.dropout_1(x)
        x = self.flatten(x)
        x = self.linear_1(x)
        x = self.relu(x)
        x = self.dropout_2(x)
        x = self.linear_2(x)
        # x = self.softmax(self.linear_2(x))
        return x




class FedAvgAPI(object):
    def __init__(self, args, device, dataset, model):
        self.device = device
        self.args = args
        [
            train_data_num,
            test_data_num,
            train_data_global,
            test_data_global,
            train_data_local_num_dict,
            train_data_local_dict,
            test_data_local_dict,
            class_num,
        ] = dataset
        self.train_global = train_data_global
        self.test_global = test_data_global
        self.val_global = None
        self.train_data_num_in_total = train_data_num
        self.test_data_num_in_total = test_data_num

        self.client_type_list=[]

        self.client_list = []
        self.train_data_local_num_dict = train_data_local_num_dict
        self.train_data_local_dict = train_data_local_dict
        self.test_data_local_dict = test_data_local_dict

        self.global_rounds=101
        self.client_total=100

        self.init=0
        self.counter=0

        self.client_select_count ={}
        self.loss_dict={}
        self.acc_dict={}
        self.round_loss={}
        self.round_acc={}
        for i in range(self.client_total):
            self.loss_dict[i] = []
            self.acc_dict[i] = []
            self.client_select_count[i]=0
        for i in range(self.global_rounds):
            self.round_loss[i] = []
            self.round_acc[i] = []

        self.tier_accuracy=[0,0,0]
        self.tier_accuracy_prev=[0,0,0]
        self.new_prob=[0.33,0.33,0.33]
        self.update_prob=5
        self.tier=[0,1,2]
        self.selected_tier=0

        #model = CNN(0.5, False)
        logging.info("model = {}".format(model))
        logging.info("Orignal model")
        self.model_trainer = create_model_trainer(model, args)
        self.model = model
        logging.info("self.model_trainer = {}".format(self.model_trainer))

        '''self._setup_clients(
            train_data_local_num_dict, train_data_local_dict, test_data_local_dict, self.model_trainer,
        )'''

    def _setup_clients(
        self, perf_coeff,client_list,train_data_local_num_dict, train_data_local_dict, test_data_local_dict, #model_trainer,
    ):
        model=CNN(perf_coeff,False)
        self.model=model
        #logging.info("model = {}".format(model))
        self.model_trainer = create_model_trainer(model, self.args)
        #logging.info("############setup_clients (START)#############")
        client_idx=client_list
        #for client_idx in client_list:#range(self.args.client_num_per_round): #needs to be modified for number of clients in a round
        c = Client(
            client_idx,
            train_data_local_dict[client_idx],
            test_data_local_dict[client_idx],
            train_data_local_num_dict[client_idx],
            self.args,
            self.device,
            self.model_trainer,
        )
        self.client_list.append(c)

        #logging.info("############setup_clients (END)#############")

    # modified by akshat
    def train(self):
        logging.info("self.model_trainer = {}".format(self.model_trainer))
        w_global = self.model_trainer.get_model_params()
        mlops.log_training_status(mlops.ClientConstants.MSG_MLOPS_CLIENT_STATUS_TRAINING)
        mlops.log_aggregation_status(mlops.ServerConstants.MSG_MLOPS_SERVER_STATUS_RUNNING)
        mlops.log_round_info(self.args.comm_round, -1)
        for round_idx in range(self.args.comm_round):

            logging.info("################Communication round : {}".format(round_idx))

            w_locals = []

            """
            for scalability: following the original FedAvg algorithm, we uniformly sample a fraction of clients in each round.
            Instead of changing the 'Client' instances, our implementation keeps the 'Client' instances and then updates their local dataset 
            """

            num_tier=3

            '''if round_idx%self.update_prob==0 and round_idx>=self.update_prob:
                logging.info("Previous tierwise accuracies = %s" % str(self.tier_accuracy_prev))
                logging.info("Current tierwise accuracies = %s" % str(self.tier_accuracy))
                if self.tier_accuracy[self.selected_tier]<=self.tier_accuracy_prev[self.selected_tier]:
                    logging.info("Tier probabilities updating")
                    tier_acc_copy=[]
                    tier_acc_copy=self.tier_accuracy
                    self.change_probs(tier_acc_copy)
                else:
                    logging.info("Tier probabilities do not need update")
                self.tier_accuracy_prev = self.tier_accuracy
                self.tier_accuracy=[0,0,0]'''


            client_indexes = self._client_sampling(
                round_idx,self.client_total
                , self.args.client_num_per_round   #needs to be modified for total and round client numbers
            )



            #modifications for clustering

            device_type_info=['High','Medium','Low']

            dev_epoch={0:5,1:3,2:1}
            client=self.client_list
            idx=0
            cluster_sample_num=0
            w_cluster_global=[]
            client_temp=[]
            l=0

            logging.info("Device type under training: %s" % str(device_type_info[self.selected_tier]))
            for client_idx in client_indexes:
                if round_idx != self.global_rounds - 1:
                    self.client_select_count[client_idx] += 1
                #client_temp.append(client_idx)
                for i in range(dev_epoch[self.selected_tier]):
                    client[client_idx].update_local_dataset(
                        client_idx,
                        self.train_data_local_dict[client_idx],
                        self.test_data_local_dict[client_idx],
                        self.train_data_local_num_dict[client_idx],
                    )
                    # train on new dataset
                    mlops.event("train", event_started=True,
                                event_value="{}_{}".format(str(round_idx), str(idx)))
                    # print((list(w_global)))

                    w, l = client[client_idx].train(copy.deepcopy(w_global))
                    # self.loss_dict[client_idx].append(l)
                    # logging.info(l)
                    mlops.event("train", event_started=False,
                                event_value="{}_{}".format(str(round_idx), str(idx)))
                    # self.logging.info("local weights = " + str(w))
                    # logging.info(client[idx].get_sample_number())
                    cluster_sample_num += client[client_idx].get_sample_number()
                    w_locals.append((client[client_idx].get_sample_number(), copy.deepcopy(w)))
                    idx += 1
                self.loss_dict[client_idx].append(l)


            cluster_sample_num = 0


                # update dataset
            #logging.info(client_temp)



            # update global weights
            mlops.event("agg", event_started=True, event_value=str(round_idx))
            #logging.info(np.shape(w_locals))
            w_global = self._aggregate(w_locals)
            self.model_trainer.set_model_params(w_global)
            mlops.event("agg", event_started=False, event_value=str(round_idx))

            sum_data_point=0
            sum_correct=0
            sum_loss=0

            for client_idx in self.loss_dict.keys():
                metrics = client[client_idx].local_test(False)
                sum_data_point+=metrics["test_total"]
                sum_correct+=metrics["test_correct"]
                sum_loss+=metrics["test_loss"]
                self.loss_dict[client_idx].append(metrics["test_loss"]/metrics["test_total"])
                self.acc_dict[client_idx].append(metrics["test_correct"] / metrics["test_total"])
                self.round_loss[round_idx].append(metrics["test_loss"]/metrics["test_total"])
                self.round_acc[round_idx].append(metrics["test_correct"] / metrics["test_total"])
                '''if round_idx%(self.update_prob-1)==0 and round_idx>=(self.update_prob-1):
                    if self.client_type_list[client_idx] == 'H':
                        self.tier_accuracy[0] += (metrics["test_correct"] / metrics["test_total"])
                    elif self.client_type_list[client_idx] == 'M':
                        self.tier_accuracy[1] += (metrics["test_correct"] / metrics["test_total"])
                    elif self.client_type_list[client_idx] == 'L':
                        self.tier_accuracy[2] += (metrics["test_correct"] / metrics["test_total"])

            if round_idx % (self.update_prob - 1) == 0 and round_idx >= (self.update_prob - 1):
                self.tier_accuracy[0] /= int(0.2 * client_num_in_total)
                self.tier_accuracy[1] /= int(0.5 * client_num_in_total)
                self.tier_accuracy[2] /= int(0.3 * client_num_in_total)'''

            train_loss=sum_loss/sum_data_point
            train_acc=sum_correct/sum_data_point
            logging.info("Training Loss = %s" %str(train_loss))
            logging.info("Training Accuracy = %s" % str(train_acc))





            # test results
            # at last round
            '''if round_idx == self.args.comm_round - 1:
                self._local_test_on_all_clients(round_idx)
            # per {frequency_of_the_test} round
            elif round_idx % self.args.frequency_of_the_test == 0:
                if self.args.dataset.startswith("stackoverflow"):
                    self._local_test_on_validation_set(round_idx)
                else:
                    self._local_test_on_all_clients(round_idx)

            mlops.log_round_info(self.args.comm_round, round_idx)'''
        #logging.info(self.loss_dict)
        filename1='C:\\Users\\Akshat\Desktop\\fedml loss hetero.txt'
        filename2 = 'C:\\Users\\Akshat\Desktop\\fedml acc hetero.txt'

        f = open(filename1, 'w')
        f.write(str(self.round_loss))
        f.close()
        f = open(filename2, 'w')
        f.write(str(self.round_acc))
        f.close()
        self.plot(self.round_loss, self.round_acc, self.client_select_count)

        mlops.log_training_finished_status()
        mlops.log_aggregation_finished_status()
    # modified by akshat

    def _client_sampling(self, round_idx, client_num_in_total, client_num_per_round):
        client_high_num = int(0.2 * client_num_in_total)
        client_medium_num = int(0.5 * client_num_in_total)
        client_low_num = int(0.3 * client_num_in_total)






        if self.init==0:
            self.init=1
            for i in range(client_high_num):
                self.client_type_list.append('H')
            for i in range(client_medium_num):
                self.client_type_list.append('M')
            for i in range(client_low_num):
                self.client_type_list.append('L')
            np.random.seed(0)
            np.random.shuffle(self.client_type_list)

            c=0
            logging.info("Setting up clients")
            for i in self.client_type_list:
                if i=='H':
                    self._setup_clients(1, c, self.train_data_local_num_dict, self.train_data_local_dict,
                                        self.test_data_local_dict)
                elif i=='M':
                    self._setup_clients(1, c, self.train_data_local_num_dict,
                                        self.train_data_local_dict,
                                        self.test_data_local_dict)
                elif i=='L':
                    self._setup_clients(1, c, self.train_data_local_num_dict,
                                        self.train_data_local_dict,
                                        self.test_data_local_dict)
                c+=1



            logging.info("Client setup completed")

        logging.info("client_configurations= %s" % str(self.client_type_list))
        client_high = [i for i, x in enumerate(self.client_type_list) if x == 'H']
        client_medium = [i for i,x in enumerate(self.client_type_list) if x=='M']
        client_low = [i for i, x in enumerate(self.client_type_list) if x == 'L']

        if client_num_in_total == client_num_per_round:
            client_indexes = [client_index for client_index in range(client_num_in_total)]
        else:
            num_clients = min(client_num_per_round, client_num_in_total)
            tier_dict={0:client_high,1:client_medium,2:client_low}
            tier_name={0:"High",1:"Medium",2:"Low"}
            #random.seed(round_idx)  # make sure for each comparison, we are selecting the same clients each round
            #client_indexes = [random.sample(client_high,int(0.2*num_clients)),random.sample(client_medium,int(0.5*num_clients)),random.sample(client_low,int(0.3*num_clients))]
            '''c_h = (self.counter*int(0.2*num_clients))% client_high_num
            c_m = (self.counter*int(0.5*num_clients)) % client_medium_num
            c_l = (self.counter*int(0.3*num_clients)) % client_low_num
            client_indexes=[client_high[c_h:c_h+int(0.2*num_clients)],client_medium[c_m:c_m+int(0.5*num_clients)],client_low[c_l:c_l+int(0.3*num_clients)]]
            self.counter+=1
            #client_indexes=random.sample(range(50), num_clients)'''
            self.selected_tier = random.choices([0, 1, 2], weights=self.new_prob, k=1)[0]
            client_indexes=random.sample(tier_dict[self.selected_tier],client_num_per_round)



        logging.info("Tier selected= %s" % str(tier_name[self.selected_tier]))
        logging.info("client_indexes = %s" % str(client_indexes))
        '''client_indexes=[i for j in client_indexes for i in j]
        idx = client_indexes
        client_type_list_selected = np.array(client_type_list)[idx].tolist()
        logging.info(client_type_list_selected)
        logging.info("Client type count:")
        logging.info("High = %d" % int(client_type_list_selected.count('H')))
        logging.info("Medium = %d" % int(client_type_list_selected.count('M')))
        logging.info("Low = %d" % int(client_type_list_selected.count('L')))
        # return high,medium and low indexes seperately or as list of list
        id_high = [i for i, x in enumerate(client_type_list_selected) if x == 'H']
        id_medium = [i for i, x in enumerate(client_type_list_selected) if x == 'M']
        id_low = [i for i, x in enumerate(client_type_list_selected) if x == 'L']
        client_high = np.array(client_indexes)[id_high].tolist()
        client_medium = np.array(client_indexes)[id_medium].tolist()
        client_low = np.array(client_indexes)[id_low].tolist()
        logging.info("Client ids by type:")
        logging.info("High devices = %s" % str(client_high))
        logging.info("Medium devices = %s" % str(client_medium))
        logging.info("Low devices = %s" % str(client_low))
        client_indexes_by_type = [client_high, client_medium, client_low]
        # logging.info(client_indexes_by_type)'''
        return client_indexes

    def plot(self, loss_list, acc_list,count_list):
        client_count=[]
        for values in count_list.values():
            client_count.append(values)

        binwidth=1
        plt.hist(client_count, bins=range(0,21, binwidth),edgecolor='black')
        plt.xlabel('Frequency')
        plt.ylabel('Client count')
        plt.xticks(np.arange(0,21,1))
        plt.title("Frequency of client selection TIFL")
        plt.grid()
        plt.show()

        data = pd.DataFrame(loss_list)
        lst = np.arange(0, self.global_rounds, ((self.global_rounds-1)/10))
        median_data = []

        plot_data = data[lst].copy()

        for i in lst:
            median_data.append(plot_data[i].median())

        # print(plot_data.head())
        sns.violinplot(data=plot_data)
        sns.lineplot(data=median_data, color="r")
        plt.xlabel("Global Round id")
        plt.ylabel("Loss")
        plt.title("Global round vs Loss across devices")
        plt.legend(
            title="Total clients={}\nClients per round=10\nMode=Clustering (5:3:1)\nMethod=TIFL(Adaptive 20:30:50)".format(self.client_total))
        plt.grid()
        plt.show()

        data = acc_list

        data = pd.DataFrame(data)

        mean_data = []

        plot_data = data[lst].copy()
        plot_data = plot_data[plot_data.select_dtypes(include=['number']).columns] * 100

        for i in lst:
            mean_data.append(plot_data[i].mean())

        # print(plot_data.head())
        sns.violinplot(data=plot_data, cut=0)
        sns.lineplot(data=mean_data, color="r")
        plt.xlabel("Global Round id")
        plt.ylabel("Accuracy")
        plt.title("Global round vs Accuracy across devices")
        plt.legend(
            title="Total clients={}\nClients per round=10\nMode=Clustering (5:3:1)\nMethod=TIFL(Adaptive 20:30:50)".format(self.client_total))
        plt.grid()
        plt.show()


    def argsort(self,seq):
        return [x for x, y in sorted(enumerate(seq), key=lambda x: x[1])]

    def change_probs(self,tier_acc):

        num_tier=3-self.credits.count(0)
        A=self.argsort(tier_acc)
        D=num_tier*(num_tier-1)/2
        for i in range(1,num_tier+1):
            self.new_prob[A[i-1]]=max(0,(num_tier-i)/D)
        logging.info("Updated tier probabilities = %s" % str(self.new_prob))



    def _generate_validation_set(self, num_samples=10000):
        test_data_num = len(self.test_global.dataset)
        sample_indices = random.sample(range(test_data_num), min(num_samples, test_data_num))
        subset = torch.utils.data.Subset(self.test_global.dataset, sample_indices)
        sample_testset = torch.utils.data.DataLoader(subset, batch_size=self.args.batch_size)
        self.val_global = sample_testset

    def _aggregate(self, w_locals):
        training_num = 0
        for idx in range(len(w_locals)):
            (sample_num, averaged_params) = w_locals[idx]
            training_num += sample_num

        (sample_num, averaged_params) = w_locals[0]
        for k in averaged_params.keys():
            for i in range(0, len(w_locals)):
                local_sample_number, local_model_params = w_locals[i]
                w = local_sample_number / training_num
                if i == 0:
                    averaged_params[k] = local_model_params[k] * w
                else:
                    averaged_params[k] += local_model_params[k] * w
        return averaged_params

    def _aggregate_noniid_avg(self, w_locals):
        """
        The old aggregate method will impact the model performance when it comes to Non-IID setting
        Args:
            w_locals:
        Returns:
        """
        (_, averaged_params) = w_locals[0]
        for k in averaged_params.keys():
            temp_w = []
            for (_, local_w) in w_locals:
                temp_w.append(local_w[k])
            averaged_params[k] = sum(temp_w) / len(temp_w)
        return averaged_params

    def _local_test_on_all_clients(self, round_idx):

        logging.info("################local_test_on_all_clients : {}".format(round_idx))

        train_metrics = {"num_samples": [], "num_correct": [], "losses": []}

        test_metrics = {"num_samples": [], "num_correct": [], "losses": []}

        client = self.client_list[0]

        for client_idx in range(self.args.client_num_in_total):
            """
            Note: for datasets like "fed_CIFAR100" and "fed_shakespheare",
            the training client number is larger than the testing client number
            """
            if self.test_data_local_dict[client_idx] is None:
                continue
            client.update_local_dataset(
                0,
                self.train_data_local_dict[client_idx],
                self.test_data_local_dict[client_idx],
                self.train_data_local_num_dict[client_idx],
            )
            # train data
            train_local_metrics = client.local_test(False)
            train_metrics["num_samples"].append(copy.deepcopy(train_local_metrics["test_total"]))
            train_metrics["num_correct"].append(copy.deepcopy(train_local_metrics["test_correct"]))
            train_metrics["losses"].append(copy.deepcopy(train_local_metrics["test_loss"]))

            # test data
            test_local_metrics = client.local_test(True)
            test_metrics["num_samples"].append(copy.deepcopy(test_local_metrics["test_total"]))
            test_metrics["num_correct"].append(copy.deepcopy(test_local_metrics["test_correct"]))
            test_metrics["losses"].append(copy.deepcopy(test_local_metrics["test_loss"]))

        # test on training dataset
        train_acc = sum(train_metrics["num_correct"]) / sum(train_metrics["num_samples"])
        train_loss = sum(train_metrics["losses"]) / sum(train_metrics["num_samples"])

        # test on test dataset
        test_acc = sum(test_metrics["num_correct"]) / sum(test_metrics["num_samples"])
        test_loss = sum(test_metrics["losses"]) / sum(test_metrics["num_samples"])

        stats = {"training_acc": train_acc, "training_loss": train_loss}
        if self.args.enable_wandb:
            wandb.log({"Train/Acc": train_acc, "round": round_idx})
            wandb.log({"Train/Loss": train_loss, "round": round_idx})

        mlops.log({"Train/Acc": train_acc, "round": round_idx})
        mlops.log({"Train/Loss": train_loss, "round": round_idx})
        logging.info(stats)

        stats = {"test_acc": test_acc, "test_loss": test_loss}
        if self.args.enable_wandb:
            wandb.log({"Test/Acc": test_acc, "round": round_idx})
            wandb.log({"Test/Loss": test_loss, "round": round_idx})

        mlops.log({"Test/Acc": test_acc, "round": round_idx})
        mlops.log({"Test/Loss": test_loss, "round": round_idx})
        logging.info(stats)


    def _local_test_on_validation_set(self, round_idx):

        logging.info("################local_test_on_validation_set : {}".format(round_idx))

        if self.val_global is None:
            self._generate_validation_set()

        client = self.client_list[0]
        client.update_local_dataset(0, None, self.val_global, None)
        # test data
        test_metrics = client.local_test(True)

        if self.args.dataset == "stackoverflow_nwp":
            test_acc = test_metrics["test_correct"] / test_metrics["test_total"]
            test_loss = test_metrics["test_loss"] / test_metrics["test_total"]
            stats = {"test_acc": test_acc, "test_loss": test_loss}
            if self.args.enable_wandb:
                wandb.log({"Test/Acc": test_acc, "round": round_idx})
                wandb.log({"Test/Loss": test_loss, "round": round_idx})

            mlops.log({"Test/Acc": test_acc, "round": round_idx})
            mlops.log({"Test/Loss": test_loss, "round": round_idx})

        elif self.args.dataset == "stackoverflow_lr":
            test_acc = test_metrics["test_correct"] / test_metrics["test_total"]
            test_pre = test_metrics["test_precision"] / test_metrics["test_total"]
            test_rec = test_metrics["test_recall"] / test_metrics["test_total"]
            test_loss = test_metrics["test_loss"] / test_metrics["test_total"]
            stats = {
                "test_acc": test_acc,
                "test_pre": test_pre,
                "test_rec": test_rec,
                "test_loss": test_loss,
            }
            if self.args.enable_wandb:
                wandb.log({"Test/Acc": test_acc, "round": round_idx})
                wandb.log({"Test/Pre": test_pre, "round": round_idx})
                wandb.log({"Test/Rec": test_rec, "round": round_idx})
                wandb.log({"Test/Loss": test_loss, "round": round_idx})

            mlops.log({"Test/Acc": test_acc, "round": round_idx})
            mlops.log({"Test/Pre": test_pre, "round": round_idx})
            mlops.log({"Test/Rec": test_rec, "round": round_idx})
            mlops.log({"Test/Loss": test_loss, "round": round_idx})
        else:
            raise Exception("Unknown format to log metrics for dataset {}!" % self.args.dataset)

        logging.info(stats)
