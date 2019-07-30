import torch
from torch.autograd import Function
from itertools import repeat
import numpy

class DiceLoss(Function):

    def __init__(self, *args, **kwargs):
        pass

    def forward(self, predict, target):
        eps = 0.00001
        target = target.squeeze()
        #b:batch_size, z:depth, y:height w:width
        b, z, y, x = target.shape
        target_ = target.view(b, -1)
        result_ = predict.argmax(1)# dim 2 is of length 2. Reduce the length to 1 and label it with the class with highest probability

        result = result_.cuda().float()
        target = target_.cuda().float()
        self.save_for_backward(result, target)

        self.intersect = torch.zeros(predict.shape[0]).cuda()
        self.union = torch.zeros(predict.shape[0]).cuda()
        dice = torch.zeros(predict.shape[0]).cuda()

        for i in range(predict.shape[0]):
            self.intersect[i] = torch.dot(result[i, :], target[i, :])
            result_sum = torch.sum(result[i, :])
            target_sum = torch.sum(target[i, :])
            self.union[i] = result_sum + target_sum

            dice[i] = 2 * self.intersect[i] / (self.union[i] + eps)
            print('union: {}\t intersect: {}\t dice_coefficient: {:.7f}'.format(str(self.union[i]), str(self.intersect[i]), dice[i]))

        sum_dice = torch.sum(dice)

        return sum_dice

    def backward(self, grad_output):
        input, target = self.saved_tensors
        intersect, union = self.intersect, self.union

        grad_input = torch.zeros(target.shape[0], 2, target.shape[1])
        grad_input = grad_input.cuda()
        for i in range(input.shape[0]):
            part1 = torch.div(target[i, :], union[i])
            part2 = intersect[i] / (union[i] * union[i])
            part2 = torch.mul(input[i, :], part2)
            dice = torch.add(torch.mul(part1, 2), torch.mul(part2, -4)).cuda()
            grad_input[i, 0, :] = torch.mul(dice, grad_output.item())
            grad_input[i, 1, :] = torch.mul(dice, -grad_output.item())

        return grad_input, None

def dice_loss(input, target):
    return DiceLoss()(input, target)

def dice_error(input, target):
    eps = 0.00001
    _, result_ = input.max(1)
    result_ = torch.squeeze(result_)
    resut = result_.cuda().float()
    target_ = target.cuda().float()
    result.copy_(result_.data)
    target_.copy_(target.data)
    target = target_
    intersect = torch.dot(result, target)
    result_sum = torch.sum(result)
    target_sum = torch.sum(target)
    union = result_sum + target_sum
    intersect = numpy.max([eps, intersect])
    dice = 2 * intersect / (union + eps)

    return dice
