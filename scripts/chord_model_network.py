"""Inference-only adaptation of Music X Lab's MIT licensed ISMIR 2019 ChordNet.

Source: music-x-lab/ISMIR2019-Large-Vocabulary-Chord-Recognition,
commit 481f4ce703f8822b99f4037e9104ba1760e21ea3, chordnet_ismir_naive.py.
The original parameter names, convolution blocks and bidirectional LSTM are retained.
Training dependencies are omitted; outputs are reshaped to [batch, frames, classes].
See worker/resources/licenses/MUSIC_X_LAB_CHORD_LICENSE.txt.
"""
import torch
from torch import nn
from torch.nn import functional as F


class CNNFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1a=nn.Conv2d(1,16,(3,3),padding=(1,1));self.norm1a=nn.InstanceNorm2d(16)
        self.conv1b=nn.Conv2d(16,16,(3,3),padding=(1,1));self.norm1b=nn.InstanceNorm2d(16)
        self.conv1c=nn.Conv2d(16,16,(3,3),padding=(1,1));self.norm1c=nn.InstanceNorm2d(16)
        self.pool1=nn.MaxPool2d((1,3))
        self.conv2a=nn.Conv2d(16,32,(3,3),padding=(1,1));self.norm2a=nn.InstanceNorm2d(32)
        self.conv2b=nn.Conv2d(32,32,(3,3),padding=(1,1));self.norm2b=nn.InstanceNorm2d(32)
        self.conv2c=nn.Conv2d(32,32,(3,3),padding=(1,1));self.norm2c=nn.InstanceNorm2d(32)
        self.pool2=nn.MaxPool2d((1,3))
        self.conv3a=nn.Conv2d(32,64,(3,3),padding=(1,1));self.norm3a=nn.InstanceNorm2d(64)
        self.conv3b=nn.Conv2d(64,64,(3,3),padding=(1,1));self.norm3b=nn.InstanceNorm2d(64)
        self.pool3=nn.MaxPool2d((1,4))
        self.conv4a=nn.Conv2d(64,80,(3,3),padding=(1,0));self.norm4a=nn.InstanceNorm2d(80)
        self.conv4b=nn.Conv2d(80,80,(3,3),padding=(1,0));self.norm4b=nn.InstanceNorm2d(80)

    def forward(self,x):
        batch,frames=x.shape[0],x.shape[1]
        x=x.reshape(batch,1,frames,252)
        x=F.selu(self.norm1a(self.conv1a(x)));x=F.selu(self.norm1b(self.conv1b(x)));x=F.selu(self.norm1c(self.conv1c(x)));x=self.pool1(x)
        x=F.selu(self.norm2a(self.conv2a(x)));x=F.selu(self.norm2b(self.conv2b(x)));x=F.selu(self.norm2c(self.conv2c(x)));x=self.pool2(x)
        x=F.selu(self.norm3a(self.conv3a(x)));x=F.selu(self.norm3b(self.conv3b(x)));x=self.pool3(x)
        x=F.selu(self.norm4a(self.conv4a(x)));x=F.selu(self.norm4b(self.conv4b(x)))
        return x.transpose(1,2).contiguous().reshape(batch,frames,240)


class StructuredChordNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.audio_feature_block=CNNFeatureExtractor()
        # Retained for strict checkpoint compatibility, though unused by upstream forward.
        self.condition_linear=nn.Linear(270,128)
        self.lstm1=nn.LSTM(240,96,num_layers=1,bidirectional=True,batch_first=True)
        self.final_fc1=nn.Linear(192,100)

    def forward(self,features):
        x=self.audio_feature_block(features)
        # Implicit zero initial states follow the input device, unlike upstream CUDA globals.
        x=self.final_fc1(self.lstm1(x)[0])
        return x[:,:,:73],x[:,:,73:86],x[:,:,86:90],x[:,:,90:94],x[:,:,94:97],x[:,:,97:100]
