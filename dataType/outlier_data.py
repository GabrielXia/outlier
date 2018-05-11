import torch.utils.data as data


class OutlierDataset(data.Dataset):
    def __getratio__(self):
        raise NotImplementedError


class Mislabeled(OutlierDataset):
    def __get_true_label__(self, index):
        return NotImplementedError
    def __get_if_outlier__(self, index):
        raise NotImplementedError


class Different(OutlierDataset):
    def __get_if_outlier__(self, index):
        raise NotImplementedError


#todo mix outlier class