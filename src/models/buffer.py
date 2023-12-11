import torch
import numpy as np

class ReportBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        if self.buffer_size > 0:
            # the current capacity of the buffer
            self.curr_cap = 0
            # initialize buffer as empty list
            self.buffer = []


    def __call__(self, fake_labels):
        # the buffer is not used
        if self.buffer_size == 0:
            return fake_labels
        
        return_labels = []
        for label in fake_labels:
            if self.curr_cap < self.buffer_size:
                self.curr_cap += 1
                self.buffer.append(label)
                return_labels.append(label)
            else:
                p = np.random.uniform(0, 1)
                # swap the buffer with probability 0.5
                if p > 0.5:
                    idx = np.random.randint(0, self.buffer_size)
                    return_labels.append(self.buffer[idx].clone())
                    self.buffer[idx] = label
                else:
                    return_labels.append(label)

        return torch.stack(return_labels)
    


class ImageBuffer(object):
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        if self.buffer_size > 0:
            # the current capacity of the buffer
            self.curr_cap = 0
            # initialize buffer as empty list
            self.buffer = []
    
    def __call__(self, imgs):
        # the buffer is not used
        if self.buffer_size == 0:
            return imgs
        
        return_imgs = []
        for img in imgs:
            img = img.unsqueeze(dim=0)
            
            # fill buffer to maximum capacity
            if self.curr_cap < self.buffer_size:
                self.curr_cap += 1
                self.buffer.append(img)
                return_imgs.append(img)
            else:
                p = np.random.uniform(low=0., high=1.)
                
                # swap images between input and buffer with probability 0.5
                if p > 0.5:
                    idx = np.random.randint(low=0, high=self.buffer_size)
                    tmp = self.buffer[idx].clone()
                    self.buffer[idx] = img
                    return_imgs.append(tmp)
                else:
                    return_imgs.append(img)
        return torch.cat(return_imgs, dim=0)