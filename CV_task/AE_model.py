import torch
import torch.nn as nn

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(nn.Conv1d(3, 64, 1),
                                     nn.BatchNorm1d(64),
                                     nn.ReLU(True),

                                     nn.Conv1d(64, 128, 1),
                                     nn.BatchNorm1d(128),
                                     nn.ReLU(True),

                                     nn.Conv1d(128, 128, 1),
                                     nn.BatchNorm1d(128),
                                     nn.ReLU(True),

                                     nn.Conv1d(128, 256, 1),
                                     nn.BatchNorm1d(256),
                                     nn.ReLU(True),

                                     nn.Conv1d(256, 128, 1),
                                     nn.BatchNorm1d(128),
                                     nn.ReLU(True),
                                     nn.MaxPool1d(2048, 1),
                                     )
        self.decoder = nn.Sequential(
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),

            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Linear(256, 2048 * 3),
        )

    def forward(self, x):
        encode = self.encoder(x)
        encode = encode.view(-1, 128)
        decode = self.decoder(encode)
        decode = decode.view(-1, 3, 2048)
        return encode, decode