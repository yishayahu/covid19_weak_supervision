import torch
import wandb
from torch.utils.data import DataLoader
class DataLoaderWrapper(DataLoader):
    def __init__(self,ds,source_dl,target_dl,steps_per_epoch,source_decay_rate,batch_size):
        super(DataLoaderWrapper, self).__init__(ds,batch_size=batch_size)
        self.source_dl = source_dl
        self.target_dl = target_dl
        self.steps_per_epoch = steps_per_epoch
        self.source_decay_rate = source_decay_rate
        self.source_amount = batch_size -1
        self.curr_step = 0

    def __next__(self):
        if self.curr_step == self.steps_per_epoch:
            self.curr_step = 0
            if self.source_dl is not None:
                self.source_amount -= self.source_decay_rate
                if self.source_amount <= 0:
                    self.source_dl = None

            raise StopIteration()
        samples = []
        masks = []
        labels = []
        if self.source_dl is not None:
            for source_sample in self.source_dl:
                samples.append(source_sample[0])
                labels.append(source_sample[1])
                if len(samples) >= self.source_amount:
                    break

        for target_sample in self.target_dl:
            samples.append(target_sample['images'])
            masks.append(target_sample['masks'])
            if len(samples) == self.batch_size:
                break
        self.curr_step+=1
        if labels:
            labels = torch.cat(labels)

        return {'images':torch.cat(samples),'masks':torch.cat(masks),'labels':labels}

    def __iter__(self):
        return self