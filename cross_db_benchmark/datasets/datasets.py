from dataclasses import dataclass


@dataclass()
class SourceDataset:
    name: str
    osf: bool = True


@dataclass()
class Database:
    db_name: str
    _source_dataset: str = None
    _data_folder: str = None
    max_no_joins: int = 4
    scale: int = 1
    contain_unicode: bool = False

    @property
    def source_dataset(self) -> str:
        if self._source_dataset is None:
            return self.db_name
        return self._source_dataset

    @property
    def data_folder(self) -> str:
        if self._data_folder is None:
            return self.db_name
        return self._data_folder


# datasets that can be downloaded from osf and should be unzipped
source_dataset_list = [
    # original datasets
    SourceDataset('airline'),
    SourceDataset('imdb'),
    SourceDataset('ssb'),
    SourceDataset('tpc_h'),
    SourceDataset('walmart'),
    SourceDataset('financial'),
    SourceDataset('basketball'),
    SourceDataset('accidents'),
    SourceDataset('movielens'),
    SourceDataset('baseball'),
    SourceDataset('hepatitis'),
    SourceDataset('tournament'),
    SourceDataset('genome'),
    SourceDataset('credit'),
    SourceDataset('employee'),
    SourceDataset('carcinogenesis'),
    SourceDataset('consumer'),
    SourceDataset('geneea'),
    SourceDataset('seznam'),
    SourceDataset('fhnk'),
]

database_list = [
    # unscaled
    Database('airline', max_no_joins=5),
    Database('imdb'),
    Database('ssb', max_no_joins=3),
    Database('tpc_h', max_no_joins=5),
    Database('walmart', max_no_joins=2),
    # scaled batch 1
    Database('financial', _data_folder='scaled_financial', scale=4),
    Database('basketball', _data_folder='scaled_basketball', scale=200),
    Database('accidents', _data_folder='accidents', scale=1, contain_unicode=True),
    Database('movielens', _data_folder='scaled_movielens', scale=8),
    Database('baseball', _data_folder='scaled_baseball', scale=10),
    # scaled batch 2
    Database('hepatitis', _data_folder='scaled_hepatitis', scale=2000),
    Database('tournament', _data_folder='scaled_tournament', scale=50),
    Database('credit', _data_folder='scaled_credit', scale=5),
    Database('employee', _data_folder='scaled_employee', scale=3),
    Database('consumer', _data_folder='scaled_consumer', scale=6),
    Database('geneea', _data_folder='scaled_geneea', scale=23, contain_unicode=True),
    Database('genome', _data_folder='scaled_genome', scale=6),
    Database('carcinogenesis', _data_folder='scaled_carcinogenesis', scale=674),
    Database('seznam', _data_folder='scaled_seznam', scale=2),
    Database('fhnk', _data_folder='scaled_fhnk', scale=2)
]

ext_database_list = database_list + [Database('imdb_full', _data_folder='imdb')]
