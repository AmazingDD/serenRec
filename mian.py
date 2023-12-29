import argparse
from srsnn.utils import *



inters = Interactions(config)
inters.build()

train_dataset = SequentialDataset(inters.train_data)
test_dataset = SequentialDataset(inters.test_data)
train_dataloader = get_dataloader(train_dataset, batch_size=128, shuffle=True)
test_dataloader = get_dataloader(train_dataset, batch_size=128, shuffle=False)

