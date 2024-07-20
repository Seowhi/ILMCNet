import torch
import torch.nn as nn
from torch.autograd import Variable


class CRF(nn.Module):

    def __init__(self, num_tag, use_cuda=False):
        if num_tag <= 0:
            raise ValueError("Invalid value of num_tag: %d" % num_tag)
        super(CRF, self).__init__()
        self.num_tag = num_tag
        self.start_tag = num_tag
        self.end_tag = num_tag + 1
        self.use_cuda = use_cuda

        self.transitions = nn.Parameter(torch.Tensor(num_tag + 2, num_tag + 2))
        nn.init.uniform_(self.transitions, -0.1, 0.1)
        self.transitions.data[self.end_tag, :] = -10000
        self.transitions.data[:, self.start_tag] = -10000

    def real_path_score(self, features, tags):

        r = torch.LongTensor(range(features.size(0)))
        if self.use_cuda:
            pad_start_tags = torch.cat([torch.cuda.LongTensor([self.start_tag]), tags])
            pad_stop_tags = torch.cat([tags, torch.cuda.LongTensor([self.end_tag])])
            r = r.cuda()
        else:
            pad_start_tags = torch.cat([torch.LongTensor([self.start_tag]), tags])
            pad_stop_tags = torch.cat([tags, torch.LongTensor([self.end_tag])])
        # Transition score + Emission score
        score = torch.sum(self.transitions[pad_start_tags, pad_stop_tags]).cpu() + torch.sum(
            features[r, tags])
        return score

    def all_possible_path_score(self, features):

        time_steps = features.size(0)

        forward = Variable(torch.zeros(self.num_tag))
        if self.use_cuda:
            forward = forward.cuda()
        for i in range(0, time_steps):
            emission_start = forward.expand(self.num_tag, self.num_tag).t()
            emission_end = features[i, :].expand(self.num_tag, self.num_tag)
            if i == 0:
                trans_score = self.transitions[self.start_tag, :self.start_tag].cpu()
            else:
                trans_score = self.transitions[:self.start_tag, :self.start_tag].cpu()
            sum = emission_start + emission_end + trans_score
            forward = log_sum(sum, dim=0)
        forward = forward + self.transitions[:self.start_tag, self.end_tag].cpu()
        total_score = log_sum(forward, dim=0)
        return total_score

    def negative_log_loss(self, inputs, output_mask, tags):

        if not self.use_cuda:
            inputs = inputs.cpu()
            output_mask = output_mask.cpu()
            tags = tags.cpu()

        loss = Variable(torch.tensor(0.), requires_grad=True)
        num_chars = torch.sum(output_mask.detach()).float()

        for ix, (features, tag) in enumerate(zip(inputs, tags)):
            # print(f'features shape = {features.shape}')  # 200 x 9
            # print(f'tag shape = {tag.shape}')    # 200

            # features (time_steps, num_tag)
            # output_mask (batch_size, time_step)
            num_valid = torch.sum(output_mask[ix].detach())
            features = features[output_mask[ix] == 1]
            tag = tag[:num_valid]
            real_score = self.real_path_score(features, tag)
            total_score = self.all_possible_path_score(features)
            cost = total_score - real_score
            loss = loss + cost
        return loss / num_chars

    def viterbi(self, features):
        time_steps = features.size(0)
        forward = Variable(torch.zeros(self.num_tag))
        if self.use_cuda:
            forward = forward.cuda()

        back_points, index_points = [self.transitions[self.start_tag, :self.start_tag].cpu()
                                    ], [torch.LongTensor([-1]).expand_as(forward)]
        for i in range(1, time_steps):
            emission_start = forward.expand(self.num_tag, self.num_tag).t()
            emission_end = features[i, :].expand(self.num_tag, self.num_tag)
            trans_score = self.transitions[:self.start_tag, :self.start_tag].cpu()
            sum = emission_start + emission_end + trans_score
            forward, index = torch.max(sum.detach(), dim=0)
            back_points.append(forward)
            index_points.append(index)
        back_points.append(forward +
                           self.transitions[:self.start_tag, self.end_tag].cpu())
        return back_points, index_points

    def get_best_path(self, features):
        back_points, index_points = self.viterbi(features)

        best_last_point = argmax(back_points[-1])
        index_points = torch.stack(index_points)
        m = index_points.size(0)

        best_path = [best_last_point]

        for i in range(m - 1, 0, -1):
            best_index_point = index_points[i][best_last_point]
            best_path.append(best_index_point)
            best_last_point = best_index_point
        best_path.reverse()
        return best_path

    def get_batch_best_path(self, inputs, output_mask):
        if not self.use_cuda:
            inputs = inputs.cpu()
            output_mask = output_mask.cpu()
        batch_best_path = []
        max_len = inputs.size(1)
        for ix, features in enumerate(inputs):
            features = features[output_mask[ix] == 1]
            best_path = self.get_best_path(features)
            best_path = torch.Tensor(best_path).long()
            best_path = padding(best_path, max_len)
            batch_best_path.append(best_path)
        batch_best_path = torch.stack(batch_best_path, dim=0)
        return batch_best_path


def log_sum(matrix, dim):

    clip_value = torch.max(matrix)
    clip_value = int(clip_value.data.tolist())
    log_sum_value = clip_value + torch.log(torch.sum(torch.exp(matrix - clip_value), dim=dim))
    return log_sum_value


def argmax(matrix, dim=0):
    """(0.5, 0.4, 0.3)"""
    _, index = torch.max(matrix, dim=dim)
    return index


def padding(vec, max_len, pad_token=-1):
    new_vec = torch.zeros(max_len).long()
    new_vec[:vec.size(0)] = vec
    new_vec[vec.size(0):] = pad_token
    return new_vec
