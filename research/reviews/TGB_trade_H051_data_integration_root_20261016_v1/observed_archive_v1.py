"""Pinned in-memory TGB observations and an explicit TRAIN-only target reader.
No network, numerical import, labels, files or device action occurs on import.
"""
from pathlib import Path
import csv,hashlib,io,json,urllib.request,zipfile
URL="https://object-arbutus.alliancecan.ca/swift/v1/14c95234f6cd4a21a47deafe20cce2a7/tgb/tgbn-trade.zip"
ARCHIVE_SHA="697e9ca432c03aa6d4a998a75a1e49310bc78682dbe8cec2d8452b5a6604c9fe"
ARCHIVE_BYTES=12780237
MAPPING_SHA="18dfd6a8a6a3a739a2f1653b1a5172e4dbf564c1073063608cc4d8373c972273"
def require(ok,message):
    if not ok:raise ValueError(message)
def download_observations_v1():
    with urllib.request.urlopen(URL,timeout=45) as response:blob=response.read(13*2**20+1)
    return ObservedArchiveV1(blob)
class ObservedArchiveV1:
    def __init__(self,blob):
        import numpy as np
        require(len(blob)==ARCHIVE_BYTES and hashlib.sha256(blob).hexdigest()==ARCHIVE_SHA,"Exact original H052 archive")
        self._archive=zipfile.ZipFile(io.BytesIO(blob))
        info=self._archive.infolist()
        require(len(info)==2 and sum(x.file_size for x in info)==45725083,"Audited archive members/size")
        names={Path(x.filename).name:x.filename for x in info}
        require(set(names)=={"tgbn-trade_edgelist.csv","tgbn-trade_node_labels.csv"},"Exact observed/label members")
        self._label_member=names["tgbn-trade_node_labels.csv"]
        self.node_ids={};self.edges={};self.year_rows={};self.train_label_member_opened=False
        with self._archive.open(names["tgbn-trade_edgelist.csv"]) as raw:
            reader=csv.reader(io.TextIOWrapper(raw,encoding="utf-8",newline=""))
            require(next(reader)==["year","nation","trading nation","weight"],"Original edge schema")
            for row in reader:
                year=int(row[0]);left,right=row[1:3];w=float(row[3])
                require(1986<=year<=2016 and np.isfinite(w) and w>0,"Audited observation years/weights")
                for name in (left,right):
                    if name not in self.node_ids:self.node_ids[name]=len(self.node_ids)
                u,v=self.node_ids[left],self.node_ids[right]
                require(max(u,v)<255,"Official fixed country count")
                if year not in self.edges:self.edges[year]=(np.zeros((255,255),dtype=np.float64),np.zeros(255,dtype=bool))
                matrix,present=self.edges[year]
                require(matrix[u,v]==0,"No duplicate observed directed pair")
                matrix[u,v]=w;present[u]=True
                self.year_rows[year]=self.year_rows.get(year,0)+1
        digest=hashlib.sha256(json.dumps(self.node_ids,separators=(",",":"),ensure_ascii=False).encode()).hexdigest()
        require(digest==MAPPING_SHA and sum(self.year_rows.values())==468245 and sorted(self.edges)==list(range(1986,2017)),"Exact original observation/country audit")
        self._train_targets=None
    def catalogue_v1(self):
        from research.reviews.TGB_trade_PyG_dataset_source_20261016_v1.trade_tree_core_v1 import CountryCatalogueV1
        return CountryCatalogueV1.from_official_mapping(dict(self.node_ids))
    def load_observed_year_v1(self,year):
        require(type(year) is int and year in self.edges,"Requested complete observed year")
        from research.reviews.TGB_trade_PyG_dataset_source_20261016_v1.trade_tree_core_v1 import observed_year_v1
        matrix,present=self.edges[year]
        return observed_year_v1(year,matrix,present)
    def dataset_v1(self,forecast_years=tuple(range(1987,2017))):
        from research.reviews.TGB_trade_PyG_dataset_source_20261016_v1.trade_tree_pyg_v1 import CausalTradeTreeDatasetV1
        return CausalTradeTreeDatasetV1(self.catalogue_v1(),self.load_observed_year_v1,forecast_years=forecast_years)
    def load_train_target_year_v1(self,year):
        # Reject a held-out request BEFORE opening/reading any label member.
        require(type(year) is int and 1987<=year<=2009,"TRAIN-only target capability")
        if self._train_targets is None:
            import numpy as np
            values={y:(np.zeros((255,255),dtype=np.float64),np.zeros(255,dtype=bool)) for y in range(1987,2010)}
            with self._archive.open(self._label_member) as raw:
                self.train_label_member_opened=True
                reader=csv.reader(io.TextIOWrapper(raw,encoding="utf-8",newline=""))
                require(next(reader)==["year","nation","trading nation","weight"],"Original target schema")
                for row in reader:
                    current=int(row[0])
                    # No held-out country/source-mask/weight is decoded or retained.
                    if current not in values:continue
                    u,v=self.node_ids[row[1]],self.node_ids[row[2]];w=float(row[3])
                    matrix,present=values[current]
                    require(np.isfinite(w) and w>0 and matrix[u,v]==0,"Exact TRAIN labels, no duplicate pair")
                    matrix[u,v]=w;present[u]=True
            self._train_targets=values
        import torch
        from research.reviews.TGB_trade_H051_training_source_20261016_v1.targets_and_metrics_v1 import TargetYearV1
        matrix,present=self._train_targets[year]
        return TargetYearV1(year,torch.from_numpy(matrix.copy()),torch.from_numpy(present.copy()))
