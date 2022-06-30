from torch.utils.data import Dataset


class PlanDataset(Dataset):
    def __init__(self, plans, idxs):
        self.plans = plans
        self.idxs = [int(i) for i in idxs]
        assert len(self.plans) == len(self.idxs)

    def __len__(self):
        return len(self.plans)

    def __getitem__(self, i: int):
        return self.idxs[i], self.plans[i]
