import copy
import logging
import random

import numpy as np
from Pyfhel import Pyfhel
import torch
import wandb
import json

from fedml import mlops
from fedml.ml.trainer.trainer_creator import create_model_trainer
from .client import Client

import os,sys
import subprocess
import glob
import math
import time
from os import path
from functools import reduce

# import pandas as pd
# import seaborn as sns
# from matplotlib import pyplot as plt


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

        # The below one stores client ids based their dataset size specified in threshold
        self.client_tier_dict = {'H':[], 'M':[], 'L':[] }

        # For each client, its tier is mentioned below
        self.client_type_list=[]

        self.client_list = []
        self.train_data_local_num_dict = train_data_local_num_dict
        self.train_data_local_dict = train_data_local_dict
        self.test_data_local_dict = test_data_local_dict

        self.global_rounds = self.args.comm_round
        self.client_total = self.args.client_num_in_total

        self.init = 0
        self.counter=[0,0,0]

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

        self.num_tier = 3

        self.tier_accuracy= [0, 0, 0]
        self.tier_accuracy_prev= [0, 0, 0]
        self.new_prob = [1,0.01,0.01]
        
        # Specify ratio for credits of each tier below
        # The first 2 parameters are used. Remaining credits from comm_round are assgined to last
        self.credits_ratio = [0.5, 0.3, 0.2]

        self.credits = list(map(lambda x: int(x*self.global_rounds), self.credits_ratio))
        
        self.credits[-1] = self.global_rounds - sum(self.credits[:-1])
        
        # The below parameter is to check for update prob based on config file value
        self.update_prob = self.args.frequency_of_the_test
        self.tier = [0,1,2]
        self.selected_tier=0

        # The below parameter is to divide the clients into tiers based on their data size
        # If a client has data size between threshold[1] (exclusive) and threshold[2] (invlusive), it is medium
        if self.args.model=="cnn": self.threshold = [0, 15, 45]
        elif self.args.model=="lr": self.threshold = [0, 12, 50]

        for client_idx in range(self.client_total):
            if (self.train_data_local_num_dict[client_idx] <= self.threshold[1]):
                self.client_type_list.append('H')
            elif (self.train_data_local_num_dict[client_idx] > self.threshold[2]):
                self.client_type_list.append('L')
            else:
                self.client_type_list.append('M')   
            self.client_tier_dict[self.client_type_list[client_idx]].append(client_idx)

        logging.info("model = {}".format(model))

        self.model_trainer = create_model_trainer(model, args)
        self.model = model
        logging.info("self.model_trainer = {}".format(self.model_trainer))

        self._setup_clients(
            train_data_local_num_dict, train_data_local_dict, test_data_local_dict, self.model_trainer,
        )
        
        self.HE = Pyfhel(key_gen=True, context_params={
            'scheme': 'CKKS', 
            'n': 2**14, # For CKKS, n/2 values can be encoded in a single ciphertext.
            'scale': 2**30,
            'qi_sizes': [60, 30, 30, 30, 60]  # Number of bits of each prime in the chain.
        })
        self.HE.relinKeyGen()
        

    def _setup_clients(
        self, train_data_local_num_dict, train_data_local_dict, test_data_local_dict, model_trainer,
    ):
        logging.info("############setup_clients (START)#############")
        for client_idx in range(self.args.client_num_per_round):
            c = Client(
                client_idx,
                train_data_local_dict[client_idx],
                test_data_local_dict[client_idx],
                train_data_local_num_dict[client_idx],
                self.args,
                self.device,
                model_trainer,
            )
            self.client_list.append(c)
        logging.info("############setup_clients (END)#############")
        
    def add_noise(self, data, seed):
        np.random.seed(seed)
        diffs = np.abs(np.diff(data))
        sensitivity = np.mean(diffs)
        noise = np.random.laplace(scale=sensitivity/10, size=data.shape)
        return data + noise

    def encrypt_arr(self, weights,idx):
        if self.args.encryption_scheme=="Homomorphic":
            print("Homomorhic encryption privacy mechanism is used")
            new_dict={}
            num_params=0
            for k,v in weights.items():
                num_params+=math.prod(list(v.size()))
                if k=="linear_2.weight" or k=="linear_2.bias" or k=="linear_1.bias" or k=="conv2d_2.bias" or k=="conv2d_1.bias" or k=="conv2d_1.weight" or k=="linear.bias" or k=="linear.weight" :                                  
                    new_dict[k] = self.HE.encryptFrac(np.ravel(v.numpy().astype(np.float64))) 
                else:
                    new_dict[k]=v
            print("Number of parameters in the model:",num_params)
            return new_dict
        elif self.args.encryption_scheme=="DiffPrivacy":
            print("Differential privacy mechanism is used")
            
            new_dict={}
            num_params=0
            
            for k,v in weights.items():
                num_params+=math.prod(list(v.size()))
                data_np=v.numpy()
                new_dict[k]=torch.from_numpy(self.add_noise(data_np,  idx))
            print("Number of parameters in the model:",num_params)
            # if idx==0 and (self.args.model=="lr") :
            #     file_name_="Noise_"+str(self.args.model)
            #     f1 = open(file_name_, 'w')
            #     str_w="Before adding noise:\n"
            #     f1.write(str_w)
            #     f1.writelines(str(weight) + '\n' for weight in weights["linear.bias"].tolist())
            #     str_w="After adding noise:\n"
            #     f1.write(str_w)
            #     f1.writelines(str(weight) + '\n' for weight in new_dict["linear.bias"].tolist())
            #     f1.close()
            # elif idx==0 and (self.args.model=="cnn"):
            #     file_name_="Noise_"+str(self.args.model)
            #     f1 = open(file_name_, 'w')
            #     str_w="Before adding noise:\n"
            #     f1.write(str_w)
            #     f1.writelines(str(weight) + '\n' for weight in weights["conv2d_1.bias"].tolist())
                
            #     str_w="After adding noise:\n"
            #     f1.write(str_w)
            #     f1.writelines(str(weight) + '\n' for weight in new_dict["conv2d_1.bias"].tolist())
                
            #     f1.close()
                
            return new_dict
        else:
            print("No encryption done")
            return weights
        
    
    def decrypt_arr(self, w_global):
        if self.args.encryption_scheme=="Homomorphic":
            if self.args.model=="lr":
                w_global["linear.bias"] = self.HE.decryptFrac(w_global["linear.bias"])[:10]
                w_global["linear.bias"] = torch.FloatTensor(w_global["linear.bias"])
                
                w_global["linear.weight"] = self.HE.decryptFrac(w_global["linear.weight"])[:7840]
                w_global["linear.weight"] = torch.FloatTensor(w_global["linear.weight"].reshape(10,784))
                
            elif self.args.model=="cnn":
                w_global["conv2d_1.weight"] = self.HE.decryptFrac(w_global["conv2d_1.weight"])[:288]
                w_global["conv2d_1.weight"] = torch.FloatTensor(w_global["conv2d_1.weight"].reshape(32,1,3,3))
                
                w_global["conv2d_1.bias"] = self.HE.decryptFrac(w_global["conv2d_1.bias"])[:32]
                w_global["conv2d_1.bias"] = torch.FloatTensor(w_global["conv2d_1.bias"])
                
                w_global["conv2d_2.bias"] = self.HE.decryptFrac(w_global["conv2d_2.bias"])[:64]
                w_global["conv2d_2.bias"] = torch.FloatTensor(w_global["conv2d_2.bias"])
                
                w_global["linear_2.bias"] = self.HE.decryptFrac(w_global["linear_2.bias"])[:62]
                w_global["linear_2.bias"] = torch.FloatTensor(w_global["linear_2.bias"])
                
                w_global["linear_1.bias"] = self.HE.decryptFrac(w_global["linear_1.bias"])[:128]
                w_global["linear_1.bias"] = torch.FloatTensor(w_global["linear_1.bias"])
                
                w_global["linear_2.weight"] = self.HE.decryptFrac(w_global["linear_2.weight"])[:7936]
                w_global["linear_2.weight"] = torch.FloatTensor(w_global["linear_2.weight"].reshape(62,128))
            return w_global 
        elif self.args.encryption_scheme=="DiffPrivacy":
            print("No Differential Privacy decryption")
            return w_global
        else:
            print("No decryption done")
            return w_global
        

    def train(self):
        logging.info("self.model_trainer = {}".format(self.model_trainer))
        w_global = self.model_trainer.get_model_params()
        mlops.log_training_status(mlops.ClientConstants.MSG_MLOPS_CLIENT_STATUS_TRAINING)
        mlops.log_aggregation_status(mlops.ServerConstants.MSG_MLOPS_SERVER_STATUS_RUNNING)
        mlops.log_round_info(self.args.comm_round, -1)
        agg_time=0
        dec_time=0
        
        
        str_write="Model:"+str(self.args.model)+"\n"
        str_write+="Dataset:"+str(self.args.dataset)+"\n"
        str_write+="Encryption Scheme:"+str(self.args.encryption_scheme)+"\n"
        str_write+="Epochs:"+str(self.args.epochs)+"\n"
        str_write+="Number of Clients:"+str(self.args.client_num_in_total)+"\n"
        str_write+="Number of Clients per round:"+str(self.args.client_num_per_round)+"\n"
        str_write+="Number of rounds:"+str(self.args.comm_round)+"\n"
        str_write+="Number of high clients:"+str(len(self.client_tier_dict['H']))+"\n"
        str_write+="Number of medium clients:"+str(len(self.client_tier_dict['M']))+"\n"
        str_write+="Number of low clients:"+str(len(self.client_tier_dict['L']))+"\n"
        str_write+="Credits:"+str(','.join(str(v) for v in self.credits))+"\n"
        
        str_write+="#"*20+"\n"
        str_write+="Round\t\t\tTier Accuracy\tTrain Accuracy\tTest Accuracy\n"
        
        for round_idx in range(self.args.comm_round):

            logging.info("################Communication round : {}".format(round_idx))

            w_locals = []

            """
            for scalability: following the original FedAvg algorithm, we uniformly sample a fraction of clients in each round.
            Instead of changing the 'Client' instances, our implementation keeps the 'Client' instances and then updates their local dataset 
            """
            client_num_in_total=self.args.client_num_in_total
            
            num_tier=self.num_tier

            client_indexes = self._client_sampling(
                round_idx, self.client_total, self.args.client_num_per_round
            )

            logging.info("client_indexes = " + str(client_indexes))

            enc_time=0
            for idx, client in enumerate(self.client_list):
                # To ensure that idx is within client_indexes
                if idx >= len(client_indexes):
                    break
                # update dataset
                client_idx = client_indexes[idx]
                client.update_local_dataset(
                    client_idx,
                    self.train_data_local_dict[client_idx],
                    self.test_data_local_dict[client_idx],
                    self.train_data_local_num_dict[client_idx],
                )

                # train on new dataset
                mlops.event("train", event_started=True, event_value="{}_{}".format(str(round_idx), str(idx)))
                w = client.train(copy.deepcopy(w_global))
                mlops.event("train", event_started=False, event_value="{}_{}".format(str(round_idx), str(idx)))
                if self.args.encryption_scheme == "DiffPrivacy" or self.args.encryption_scheme == "Homomorphic":
                    st = time.time()
                    w=self.encrypt_arr(w,idx)
                    et = time.time()
                    enc_time+=(et-st)
                else:
                    print("No encryption scheme is used")
                w_locals.append((client.get_sample_number(), copy.deepcopy(w)))

            # update global weights
            mlops.event("agg", event_started=True, event_value=str(round_idx))
            st = time.time()
            w_global = self._aggregate(w_locals)
            et = time.time()
            agg_time+=(et-st)
            if self.args.encryption_scheme == "DiffPrivacy" or self.args.encryption_scheme == "Homomorphic":
                st = time.time()
                w_global=self.decrypt_arr(w_global)
                et = time.time()
                dec_time+=(et-st)
            self.model_trainer.set_model_params(w_global)
            mlops.event("agg", event_started=False, event_value=str(round_idx))

            if round_idx == self.args.comm_round - 1:
                (train_acc,test_acc)=self._local_test_on_all_clients(round_idx)
            # per {frequency_of_the_test} round
            elif round_idx % self.args.frequency_of_the_test == 0:
                if self.args.dataset.startswith("stackoverflow"):
                    self._local_test_on_validation_set(round_idx)
                else:
                    (train_acc,test_acc)=self._local_test_on_all_clients(round_idx)

            if round_idx%self.update_prob==0 and round_idx>=self.update_prob and round_idx != self.args.comm_round - 1:
                logging.info("Previous tierwise accuracies = %s" % str(self.tier_accuracy_prev))
                logging.info("Current tierwise accuracies = %s" % str(self.tier_accuracy))
                if self.tier_accuracy[self.selected_tier]>self.tier_accuracy_prev[self.selected_tier]:
                    logging.info("Tier probabilities updating")
                    self.change_probs(copy.deepcopy(self.tier_accuracy))
                else:
                    logging.info("Tier probabilities do not need update")
                self.tier_accuracy_prev = copy.deepcopy(self.tier_accuracy)
                self.tier_accuracy= [0,0,0]

            mlops.log_round_info(self.args.comm_round, round_idx)
            str_write+=str(round_idx)+"\t"+str(self.tier_accuracy_prev[0])+","+str(self.tier_accuracy_prev[1])+","+str(self.tier_accuracy_prev[2])
            if round_idx == self.args.comm_round - 1 or round_idx % self.args.frequency_of_the_test == 0:
                str_write+="\t"+str(train_acc)+"\t"+str(test_acc)+"\n"
            else:
                str_write+="\t-\t-\n"
        str_write+="#"*20+"\n"
        str_write+="Average time for aggregation:"+str(agg_time/self.args.comm_round)+" sec"+"\n"
        str_write+="Average time for encryption:"+str(enc_time)+" sec"+"\n"
        str_write+="Average time for decryption:"+str(dec_time/self.args.comm_round)+" sec"+"\n"
        file_name="output_"+str(self.args.encryption_scheme)+"_"+str(self.args.model)+"_"+str(self.args.dataset)+"_sensitivity0.1"
        f = open(file_name, 'w')
        f.write(str_write)
        f.close()

        print("DreamFEDML: Line 126 Model: "+str(self.args.model)+" Dataset: "\
              +str(self.args.dataset)+" Feature size of local weight: " + str(len(w_locals[0][1])))
        mlops.log_training_finished_status()
        mlops.log_aggregation_finished_status()
        
        # filename1='log_loss.txt'
        # filename2 = 'log_acc.txt'

        # f = open(filename1, 'w')
        
        # f.write(str(self.round_loss))
        # f.close()
        # f = open(filename2, 'w')
        # f.write(str(self.round_acc))
        # f.close()
        # #print(len(self.round_loss),len(self.round_acc))
        # #self.plot(self.round_loss,self.round_acc,self.client_select_count)

    def _client_sampling(self, round_idx, client_num_in_total, client_num_per_round):
        tier = copy.deepcopy(self.tier)
        probs = copy.deepcopy(self.new_prob)
        types = ['H', 'M', 'L']

        random.seed(round_idx)

        while True:
                self.selected_tier = random.choices(tier, weights=probs, k=1)[0]
                # logging.info(self.new_prob)
                if self.credits[self.selected_tier]!=0:
                    break
                else:
                    # logging.info("Insufficient credits for tier %s" %str(tier_name[self.selected_tier]))
                    del_index = tier.index(self.selected_tier)
                    del tier[del_index]
                    del probs[del_index]
        
        num_clients = min(client_num_per_round, len(self.client_tier_dict[types[self.selected_tier]]))
        np.random.seed(round_idx)  # make sure for each comparison, we are selecting the same clients each round
        client_indexes = np.random.choice(self.client_tier_dict[types[self.selected_tier]], num_clients, replace=False)
        print("Selected tier for round ", round_idx, " is ", types[self.selected_tier])
        logging.info("client_indexes = %s" % str(client_indexes))
        self.credits[self.selected_tier] -= 1
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
        plt.yticks(np.arange(0, 101, 10))
        plt.title("Frequency of client selection TIFL")
        plt.grid()
        plt.show()

        data = pd.DataFrame(loss_list)
        lst = np.arange(0, self.global_rounds, ((self.global_rounds-1) / 10))
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
            title="Total clients={}\nClients per round=10\nMode=Clustering (5:3:1)\nMethod=TIFL(Adaptive 25:15:10)".format(
                self.client_total))
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
            title="Total clients={}\nClients per round=10\nMode=Clustering (5:3:1)\nMethod=TIFL(Adaptive 25:15:10)".format(
                self.client_total))
        plt.grid()
        plt.show()

    def argsort(self,seq):
        result = [x for x, y in sorted(enumerate(seq), key=lambda x: x[1])]
        return result

    def change_probs(self,tier_acc):
        #num_tier=3-self.credits.count(0)
        num_tier=3
        array = []
        for i in range(num_tier):
            array.append(tier_acc[i])
        A = self.argsort(array)
        D=num_tier*(num_tier+1)/2

        for i in range(1, num_tier+1):
            self.new_prob[A[i - 1]] = (num_tier - i +1) / D

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

        types = ['H', 'M', 'L']

        train_acc_tier = {}
        test_acc_tier = {}

        train_loss_tier = {}
        test_loss_tier = {}

        tier_train_metrics = {"num_samples": [], "num_correct": [], "losses": []}

        tier_test_metrics = {"num_samples": [], "num_correct": [], "losses": []}

        for tier in self.tier:

            train_metrics = {"num_samples": [], "num_correct": [], "losses": []}

            test_metrics = {"num_samples": [], "num_correct": [], "losses": []}

            client = self.client_list[0]

            for client_idx in self.client_tier_dict[types[tier]]:
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
            
            # test on training dataset of tier
            train_acc_tier[tier] = sum(train_metrics["num_correct"]) / sum(train_metrics["num_samples"])
            train_loss_tier[tier] = sum(train_metrics["losses"]) / sum(train_metrics["num_samples"])

            self.tier_accuracy = copy.deepcopy(train_acc_tier)

            # test on test dataset of tier
            test_acc_tier[tier] = sum(test_metrics["num_correct"]) / sum(test_metrics["num_samples"])
            test_loss_tier[tier] = sum(test_metrics["losses"]) / sum(test_metrics["num_samples"])

            tier_train_metrics["num_samples"].extend(train_metrics["num_samples"])
            tier_train_metrics["num_correct"].extend(train_metrics["num_correct"])
            tier_train_metrics["losses"].extend(train_metrics["losses"])

            tier_test_metrics["num_samples"].extend(test_metrics["num_samples"])
            tier_test_metrics["num_correct"].extend(test_metrics["num_correct"])
            tier_test_metrics["losses"].extend(test_metrics["losses"])

        train_metrics = tier_train_metrics
        test_metrics = tier_test_metrics

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
        return (train_acc,test_acc)


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
