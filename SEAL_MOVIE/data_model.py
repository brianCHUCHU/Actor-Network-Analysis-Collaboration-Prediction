import os.path as osp
import torch
from torch_geometric.data import Data, InMemoryDataset

class MovieDataset(InMemoryDataset):
    def __init__(
            self, 
            root,
            name,
            transform=None,
            pre_transform=None,
            pre_filter=None
        ):
        self.name = name
        self.url = './'
        self.folder = './dataset/movie_actor'
        self.filename = 'movie.pt'
        super().__init__(root, transform, pre_transform, pre_filter)
        self.load(self.processed_paths[0])
    
    @property
    def raw_file_names(self):
        return ['movie.pt']
    
    @property
    def processed_file_names(self):
        return ['movie.pt']
    
    def process(self):
        data_list = self.read_movie_data(self.folder, self.filename)
        self.save(data_list, self.processed_paths[0])
    
    def read_movie_data(self, folder: str, filename: str) -> Data:
        data = torch.load(osp.join(folder, filename))
        return [data]

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)
