from typing import Dict, List
from torch.utils.data import Dataset
import pandas as pd 


class LazySupervisedDataset(Dataset):
    """
    Dataset for supervised fine-tuning 
    """

    def __init__(
        self, 
        data_path: str, 
    ) -> None:
        super(LazySupervisedDataset, self).__init__()
        self.csv = pd.read_csv(data_path)

    def __len__(self) -> int:
        return len(self.csv) 

    def construct_messages(self, text):
        message = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"{text}\nSummarize above sentence in one word: "}
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": f"<emb>."}
                ]
            },
        ]
        return message

    def get_instance(self, index):
        sent0, sent1, sent2 = self.csv['sent0'][index], self.csv['sent1'][index], self.csv['hard_neg'][index]
        message1 = self.construct_messages(sent0)
        message2 = self.construct_messages(sent1)
        message3 = self.construct_messages(sent2)

        # return message1, message2, message3 
        return message1, message2 

    def __getitem__(self, i) -> Dict[str, List]:      
        return self.get_instance(i)