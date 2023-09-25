import json
import os
import time
from typing import Iterable

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from trlx.data.ppo_types import PPORLBatch, PPORLElement
from trlx.pipeline import BaseRolloutStore
from trlx.utils.sampler import WeightDecaySampler


class PPORolloutStorage(BaseRolloutStore):
    """
    Rollout storage for training PPO
    """

    def __init__(self, pad_token_id, samples_weight):
        super().__init__()

        self.samples_weight = samples_weight
        self.pad_token_id = pad_token_id
        self.history: Iterable[PPORLElement] = [None]

    def push(self, exps: Iterable[PPORLElement]):
        self.history += exps

    def clear_history(self):
        self.history = []

    def export_history(self, location: str):
        assert os.path.exists(location)

        fpath = os.path.join(location, f"epoch-{str(time.time())}.json")

        def exp_to_dict(exp):
            {k: v.cpu().tolist() for k, v in exp.__dict__.items()}

        data = [exp_to_dict(exp) for exp in self.history]
        with open(fpath, "w") as f:
            f.write(json.dumps(data, indent=2))

    def __getitem__(self, index: int) -> PPORLElement:
        return self.history[index]

    def __len__(self) -> int:
        return len(self.history)

    def create_loader(
        self,
        batch_size: int,
        shuffle: bool,
    ) -> DataLoader:
        def collate_fn(elems: Iterable[PPORLElement]):
            return PPORLBatch(
                # Left padding of already left-padded queries
                pad_sequence(
                    [elem.query_tensor.flip(0) for elem in elems],
                    padding_value=self.pad_token_id,
                    batch_first=True,
                ).flip(1),
                # Right pad the rest, to have a single horizontal query/response split
                pad_sequence(
                    [elem.response_tensor for elem in elems],
                    padding_value=self.pad_token_id,
                    batch_first=True,
                ),
                pad_sequence(
                    [elem.logprobs for elem in elems],
                    padding_value=0.0,
                    batch_first=True,
                ),
                pad_sequence([elem.values for elem in elems], padding_value=0.0, batch_first=True),
                pad_sequence(
                    [elem.rewards for elem in elems],
                    padding_value=0.0,
                    batch_first=True,
                ),
            )
        ##################### weight decay #######################
        sampler = WeightDecaySampler(self.samples_weight.type('torch.DoubleTensor'))
        return DataLoader(self, batch_size, sampler=sampler, collate_fn=collate_fn)
        ##################### weight decay #######################
        # return DataLoader(self, batch_size, shuffle=shuffle, collate_fn=collate_fn)