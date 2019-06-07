import bisect
import math
#============================================
__author__ = "Sachin Mehta"
__maintainer__ = "Sachin Mehta"
#============================================

class CyclicLR(object):
    '''
    CLass that defines cyclic learning rate with warm restarts that decays the learning rate linearly till the end of cycle and then restarts
    at the maximum value.
    See https://arxiv.org/abs/1811.11431 for more details
    '''
    def __init__(self, min_lr=0.1, cycle_len=5, steps=[51, 101, 131, 161, 191, 221, 251, 281], gamma=0.5, step=True):
        super(CyclicLR, self).__init__()
        assert len(steps) > 0, 'Please specify step intervals.'
        assert 0 < gamma <= 1, 'Learing rate decay factor should be between 0 and 1'
        self.min_lr = min_lr # minimum learning rate
        self.m = cycle_len
        self.steps = steps
        self.warm_up_interval = 1 # we do not start from max value for the first epoch, because some time it diverges
        self.counter = 0
        self.decayFactor = gamma # factor by which we should decay learning rate
        self.count_cycles = 0
        self.step_counter = 0
        self.stepping = step

    def step(self, epoch):
        if epoch%self.steps[self.step_counter] == 0 and epoch > 1 and self.stepping:
            self.min_lr = self.min_lr * self.decayFactor
            self.count_cycles = 0
            if self.step_counter < len(self.steps) - 1:
                self.step_counter += 1
            else:
                self.stepping = False
        current_lr = self.min_lr
        # warm-up or cool-down phase
        if self.count_cycles < self.warm_up_interval:
            self.count_cycles += 1
            # We do not need warm up after first step.
            # so, we set warm up interval to 0 after first step
            if self.count_cycles == self.warm_up_interval:
                self.warm_up_interval = 0
        else:
            #Cyclic learning rate with warm restarts
            # max_lr (= min_lr * step_size) is decreased to min_lr using linear decay before
            # it is set to max value at the end of cycle.
            if self.counter >= self.m:
                self.counter = 0
            current_lr = round((self.min_lr * self.m) - (self.counter * self.min_lr), 5)
            self.counter += 1
            self.count_cycles += 1
        return current_lr

    def __repr__(self):
        fmt_str = 'Scheduler ' + self.__class__.__name__ + '\n'
        fmt_str += '    Min. base LR: {}\n'.format(self.min_lr)
        fmt_str += '    Max. base LR: {}\n'.format(self.min_lr * self.m)
        fmt_str += '    Step interval: {}\n'.format(self.steps)
        fmt_str += '    Decay lr at each step by {}\n'.format(self.decayFactor)
        return fmt_str


class FixedMultiStepLR(object):
    '''
        Fixed LR scheduler with steps
    '''
    def __init__(self, base_lr=0.1, steps=[30, 60, 90], gamma=0.1, step=True):
        super(FixedMultiStepLR, self).__init__()
        assert len(steps) > 1, 'Please specify step intervals.'
        self.base_lr = base_lr
        self.steps = steps
        self.decayFactor = gamma # factor by which we should decay learning rate
        self.stepping = step
        print('Using Fixed LR Scheduler')

    def step(self, epoch):
        return round(self.base_lr * (self.decayFactor ** bisect.bisect(self.steps, epoch)), 5)

    def __repr__(self):
        fmt_str = 'Scheduler ' + self.__class__.__name__ + '\n'
        fmt_str += '    Base LR: {}\n'.format(self.base_lr)
        fmt_str += '    Step interval: {}\n'.format(self.steps)
        fmt_str += '    Decay lr at each step by {}\n'.format(self.decayFactor)
        return fmt_str


class PolyLR(object):
    '''
        Polynomial LR scheduler with steps
    '''
    def __init__(self, base_lr, max_epochs, power=0.99):
        super(PolyLR, self).__init__()
        assert 0 < power < 1
        self.base_lr = base_lr
        self.power = power
        self.max_epochs = max_epochs

    def step(self, epoch):
        curr_lr = self.base_lr * (1 - (float(epoch) / self.max_epochs)) ** self.power
        return round(curr_lr, 6)

    def __repr__(self):
        fmt_str = 'Scheduler ' + self.__class__.__name__ + '\n'
        fmt_str += '    Total Epochs: {}\n'.format(self.max_epochs)
        fmt_str += '    Base LR: {}\n'.format(self.base_lr)
        fmt_str += '    Power: {}\n'.format(self.power)
        return fmt_str


class LinearLR(object):
    def __init__(self, base_lr, max_epochs):
        super(LinearLR, self).__init__()
        self.base_lr = base_lr
        self.max_epochs = max_epochs

    def step(self, epoch):
        curr_lr = self.base_lr - (self.base_lr * (epoch / (self.max_epochs)))
        return round(curr_lr, 6)

    def __repr__(self):
        fmt_str = 'Scheduler ' + self.__class__.__name__ + '\n'
        fmt_str += '    Total Epochs: {}\n'.format(self.max_epochs)
        fmt_str += '    Base LR: {}\n'.format(self.base_lr)
        return fmt_str


class HybirdLR(object):
    def __init__(self, base_lr, clr_max, max_epochs, cycle_len=5):
        super(HybirdLR, self).__init__()
        self.linear_epochs = max_epochs - clr_max + 1
        steps = [clr_max]
        self.clr = CyclicLR(min_lr=base_lr, cycle_len=cycle_len, steps=steps, gamma=1)
        self.decay_lr = LinearLR(base_lr=base_lr, max_epochs=self.linear_epochs)
        self.cyclic_epochs = clr_max

        self.base_lr=base_lr
        self.max_epochs = max_epochs
        self.clr_max = clr_max
        self.cycle_len = cycle_len

    def step(self, epoch):
        if epoch < self.cyclic_epochs:
            curr_lr = self.clr.step(epoch)
        else:
            curr_lr = self.decay_lr.step(epoch - self.cyclic_epochs + 1)
        return round(curr_lr, 6)

    def __repr__(self):
        fmt_str = 'Scheduler ' + self.__class__.__name__ + '\n'
        fmt_str += '    Total Epochs: {}\n'.format(self.max_epochs)
        fmt_str += '    Cycle with length of {}: {}\n'.format(self.cycle_len , int(self.clr_max/self.cycle_len))
        fmt_str += '    Base LR with {} cycle length: {}\n'.format(self.cycle_len, self.base_lr)
        fmt_str += '    Cycle with length of {}: {}\n'.format(self.linear_epochs, 1)
        fmt_str += '    Base LR with {} cycle length: {}\n'.format(self.linear_epochs, self.base_lr)
        return fmt_str


class CosineLR(object):
    def __init__(self, base_lr, max_epochs):
        super(CosineLR, self).__init__()
        self.base_lr = base_lr
        self.max_epochs = max_epochs

    def step(self, epoch):
        return round(self.base_lr * (1 + math.cos(math.pi * epoch / self.max_epochs)) / 2, 6)

    def __repr__(self):
        fmt_str = 'Scheduler ' + self.__class__.__name__ + '\n'
        fmt_str += '    Total Epochs: {}\n'.format(self.max_epochs)
        fmt_str += '    Base LR : {}\n'.format(self.base_lr)
        return fmt_str


if __name__ == '__main__':
    max_epochs = 100
    lrSched = PolyLR(0.007, max_epochs=max_epochs)
    #lrSched = CosineLR(0.01, max_epochs=200)
    #for i in range(max_epochs):
    #    print(i, lrSched.step(i))
    #exit()
    lrSched = HybirdLR(0.007, clr_max=51, max_epochs=max_epochs)
    for i in range(max_epochs):
        print(i, lrSched.step(i))
    exit()
    lrSched = PolyLR(0.007, max_epochs=50)
    for i in range(50):
        print(i, lrSched.step(i))

    max_epochs = 240
    lrSched = CyclicLR(min_lr=0.01, steps=[51, 161, 201], gamma=0.1)
    print(lrSched)
    for i in range(max_epochs):
        print(i, lrSched.step(i))
