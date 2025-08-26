import numpy as np
import torch
import random
import math
import torch.nn.functional as F
from torch.distributions import Normal
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.kl import kl_divergence


class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, km_num, max_size=int(18e4)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.km_num = km_num
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device)
        )

    def sample1(self, batch_size, temp_number, actor, action_dim):
        ind11 = np.zeros((0), dtype=int)
        dis_list = []

        kl_div_var = 0.15
        ref_gaussian = MultivariateNormal(torch.zeros(action_dim).to(self.device),
                                          torch.eye(action_dim).to(self.device) * kl_div_var)

        # Iterate through each cluster
        for i in range(len(temp_number) - 1):
            # Check if the current cluster has samples
            if temp_number[i] < temp_number[i + 1]:
                # Randomly sample from the current cluster
                ind1 = random.sample(range(temp_number[i], temp_number[i + 1]),
                                     min(int(math.ceil(batch_size / self.km_num)), temp_number[i + 1] - temp_number[i]))

                current_action1 = actor(torch.FloatTensor(self.state[ind1]).to(self.device))

                ai = torch.FloatTensor(self.action[ind1]).to(self.device)

                # Compute the difference batch
                diff_action_batch = ai - current_action1

                # Get the mean and covariance matrix for the
                mean = torch.mean(diff_action_batch, dim=0)
                cov = torch.mm(torch.transpose(diff_action_batch - mean, 0, 1),
                               diff_action_batch - mean) / batch_size

                multivar_gaussian = MultivariateNormal(mean, cov)

                dis = (kl_divergence(multivar_gaussian, ref_gaussian) + kl_divergence(ref_gaussian,
                                                                                      multivar_gaussian)) / 2

                # dis = F.mse_loss(actor(torch.FloatTensor(self.state[ind1]).to(self.device)),torch.FloatTensor(self.action[ind1]).to(self.device)).mean()

                # Calculate the loss for the current cluster's samples
                dis_list.append(torch.neg(dis))

        # Normalize the loss with softmax
        if dis_list:
            dis_softmax = F.softmax(torch.tensor(dis_list), dim=0)
        else:
            dis_softmax = torch.tensor([])

        # Initialize the list for the sampling result
        sampled_ind = []

        # Repeat the sampling process until batch_size samples are collected
        while len(sampled_ind) < batch_size:
            # Generate a random number to find clusters with priority higher than this random number
            rand_priority = random.random()
            cum_prob = 0

            # Iterate through each cluster and find clusters with priority higher than rand_priority
            for j, prob in enumerate(dis_softmax):
                cum_prob += prob.item()
                if cum_prob > rand_priority:
                    # Randomly select a sample from this cluster
                    sampled_ind.append(random.randint(temp_number[j], temp_number[j + 1] - 1))
                    break

        # Merge the indices of all the sampled samples
        ind = np.hstack((ind11, sampled_ind))

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device)
        )

    def sampleall(self):
        return (
            torch.FloatTensor(self.state).to(self.device),
            torch.FloatTensor(self.action).to(self.device),
            torch.FloatTensor(self.next_state).to(self.device),
            torch.FloatTensor(self.reward).to(self.device),
            torch.FloatTensor(self.not_done).to(self.device)
        )

    def Choose_sample(self, result):
        index = []
        for i in range(self.km_num):
            index0 = np.where(result == i)
            index0 = np.array(index0)
            index0 = index0.tolist()
            index0 = index0[0]
            index0 = np.array(index0)
            index.append(index0)

        return index

    def sample_ind(self, ind):
        sample = []
        for i in range(len(ind)):
            temp_sample = [self.state[ind[i]], self.action[ind[i]], self.next_state[ind[i]], self.reward[ind[i]],
                           self.not_done[ind[i]]]
            sample.append(temp_sample)
        return sample


class ReplayBuffer1(object):
    def __init__(self, state_dim, action_dim, km_num, max_size=int(2e5)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.km_num = km_num
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device)
        )

    def sample1(self, batch_size, temp_number, actor, action_dim):
        ind11 = np.zeros((0), dtype=int)
        dis_list = []

        kl_div_var = 0.15
        ref_gaussian = MultivariateNormal(torch.zeros(action_dim).to(self.device),
                                          torch.eye(action_dim).to(self.device) * kl_div_var)

        # Iterate through each cluster
        for i in range(len(temp_number) - 1):
            # Check if the current cluster has samples
            if temp_number[i] < temp_number[i + 1]:
                # Randomly sample from the current cluster
                ind1 = random.sample(range(temp_number[i], temp_number[i + 1]),
                                     min(int(math.ceil(batch_size / self.km_num)), temp_number[i + 1] - temp_number[i]))

                current_action1 = actor(torch.FloatTensor(self.state[ind1]).to(self.device))

                ai = torch.FloatTensor(self.action[ind1]).to(self.device)

                # Compute the difference batch
                diff_action_batch = ai - current_action1

                # Get the mean and covariance matrix for the
                mean = torch.mean(diff_action_batch, dim=0)
                cov = torch.mm(torch.transpose(diff_action_batch - mean, 0, 1),
                               diff_action_batch - mean) / batch_size

                multivar_gaussian = MultivariateNormal(mean, cov)

                dis = (kl_divergence(multivar_gaussian, ref_gaussian) + kl_divergence(ref_gaussian,
                                                                                      multivar_gaussian)) / 2

                # dis = F.mse_loss(actor(torch.FloatTensor(self.state[ind1]).to(self.device)),torch.FloatTensor(self.action[ind1]).to(self.device)).mean()

                # Calculate the loss for the current cluster's samples
                dis_list.append(torch.neg(dis))

        # Normalize the loss with softmax
        if dis_list:
            dis_softmax = F.softmax(torch.tensor(dis_list), dim=0)
        else:
            dis_softmax = torch.tensor([])

        # Initialize the list for the sampling result
        sampled_ind = []

        # Repeat the sampling process until batch_size samples are collected
        while len(sampled_ind) < batch_size:
            # Generate a random number to find clusters with priority higher than this random number
            rand_priority = random.random()
            cum_prob = 0

            # Iterate through each cluster and find clusters with priority higher than rand_priority
            for j, prob in enumerate(dis_softmax):
                cum_prob += prob.item()
                if cum_prob > rand_priority:
                    # Randomly select a sample from this cluster
                    sampled_ind.append(random.randint(temp_number[j], temp_number[j + 1] - 1))
                    break

        # Merge the indices of all the sampled samples
        ind = np.hstack((ind11, sampled_ind))

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device)
        )

    def sampleall(self):
        return (
            torch.FloatTensor(self.state).to(self.device),
            torch.FloatTensor(self.action).to(self.device),
            torch.FloatTensor(self.next_state).to(self.device),
            torch.FloatTensor(self.reward).to(self.device),
            torch.FloatTensor(self.not_done).to(self.device)
        )

    def Choose_sample(self, result):
        index = []
        for i in range(self.km_num):
            index0 = np.where(result == i)
            index0 = np.array(index0)
            index0 = index0.tolist()
            index0 = index0[0]
            index0 = np.array(index0)
            index.append(index0)

        return index

    def sample_ind(self, ind):
        sample = []
        for i in range(len(ind)):
            temp_sample = [self.state[ind[i]], self.action[ind[i]], self.next_state[ind[i]], self.reward[ind[i]],
                           self.not_done[ind[i]]]
            sample.append(temp_sample)
        return sample


class ReplayBuffer2(ReplayBuffer1):
    pass


class ReplayBuffer3(ReplayBuffer1):
    pass


class ReplayBuffer4(ReplayBuffer1):
    pass


class ReplayBuffer5(ReplayBuffer1):
    pass


class ReplayBuffer6(ReplayBuffer1):
    pass


class ReplayBuffer7(ReplayBuffer1):
    pass


class ReplayBuffer8(ReplayBuffer1):
    pass


class ReplayBuffer9(ReplayBuffer1):
    pass


class ReplayBuffer10(ReplayBuffer1):
    pass


class ReplayBuffer11(ReplayBuffer1):
    pass


class ReplayBuffer12(ReplayBuffer1):
    pass


class ReplayBuffer13(ReplayBuffer1):
    pass


class ReplayBuffer14(ReplayBuffer1):
    pass


class ReplayBuffer15(ReplayBuffer1):
    pass


class ReplayBuffer16(ReplayBuffer1):
    pass


class ReplayBuffer17(ReplayBuffer1):
    pass


class ReplayBuffer18(ReplayBuffer1):
    pass


class ReplayBuffer19(ReplayBuffer1):
    pass


class ReplayBuffer20(ReplayBuffer1):
    pass
