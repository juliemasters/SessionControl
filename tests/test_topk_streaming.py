"""Filesystem read counts must follow the shortlist, without a factor cache."""
import tempfile
from pathlib import Path
import unittest
from types import SimpleNamespace
import torch
from comparison_core import digest
from experiments.zsre_topk.core import ActivationRouter
from experiments.topk_streaming import LazyFactorBank


class StreamingFactorTests(unittest.TestCase):
    def fixture(self,root,n=16,dataset='zsre',rank=1):
        torch.manual_seed(8);model=torch.nn.Linear(3,2,bias=False);files=[];expected={}
        for i in range(n):
            sid=f'case_{i:04d}';file=root/(sid+'.pt')
            payload={'weight':(torch.randn(2) if rank==1 else torch.randn(2,rank),
                               torch.randn(3) if rank==1 else torch.randn(3,rank))}
            torch.save(payload,file);expected[sid]=payload
            files.append(dict(sid=sid,path=str(file),sha256=digest(file)))
        d=dict(dataset=dataset,factor_files=files,bank_sids=[f['sid'] for f in files])
        return LazyFactorBank(d,torch,model,['weight'],n),expected,d

    def test_full_scan_and_top8_read_only_the_scored_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            bank,expected,d=self.fixture(Path(tmp))
            disk,payload=bank.validate_all()
            self.assertEqual(disk,sum(Path(f['path']).stat().st_size for f in d['factor_files']))
            self.assertEqual(payload,16*(2+3)*4)
            index=torch.nn.functional.normalize(torch.randn(16,3),dim=1)
            router=ActivationRouter(bank,['weight'],index,d['bank_sids'])
            keys={'weight':torch.tensor([[1.,.2,.3]])}
            bank.reset_counters();full=router.route(keys)
            self.assertEqual(bank.file_load_count,16)
            self.assertEqual(bank.file_load_bytes,disk)
            reference=ActivationRouter(expected,['weight'],index,d['bank_sids']).route(keys)
            self.assertEqual((full['selected'],full['score']),(reference['selected'],reference['score']))
            bank.reset_counters();short=router.route(keys,8)
            self.assertEqual(bank.file_load_count,8)
            self.assertEqual(bank.file_load_bytes,sum(Path(f['path']).stat().st_size for f in d['factor_files'] if f['sid'] in short['shortlist']))

    def test_mapping_does_not_cache_loaded_payloads(self):
        with tempfile.TemporaryDirectory() as tmp:
            bank,expected,d=self.fixture(Path(tmp),n=2)
            bank.reset_counters();first=bank[d['bank_sids'][0]];second=bank[d['bank_sids'][0]]
            self.assertEqual(bank.file_load_count,2)
            self.assertTrue(torch.equal(first['weight'][0],second['weight'][0]))
            self.assertNotEqual(first['weight'][0].data_ptr(),second['weight'][0].data_ptr())

    def test_canonical_multirank_files_and_source_integrity(self):
        with tempfile.TemporaryDirectory() as tmp:
            bank,expected,d=self.fixture(Path(tmp),n=2,dataset='longmemeval',rank=2)
            _,payload=bank.validate_all();self.assertEqual(payload,2*(2+3)*2*4)
            value=bank[d['bank_sids'][0]];self.assertEqual(value['weight'][0].shape,(2,2))
            d['factor_files'][0]['sha256']='0'*64
            other=LazyFactorBank(d,torch,torch.nn.Linear(3,2,bias=False),['weight'],2)
            with self.assertRaises(ValueError):other.validate_all()


if __name__=='__main__':unittest.main()
