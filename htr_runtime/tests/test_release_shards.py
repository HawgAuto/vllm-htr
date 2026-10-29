import importlib.util
import json
import struct
import tempfile
import unittest
from pathlib import Path

SPEC = importlib.util.find_spec('release_shards')
S = importlib.util.module_from_spec(SPEC) if SPEC else None
if S:
    SPEC.loader.exec_module(S)


def tensor_file(path, entries):
    header = {'__metadata__': {'format': 'pt'}}
    data = b''
    for name, dtype, shape, raw in entries:
        header[name] = dict(dtype=dtype, shape=shape, data_offsets=[len(data), len(data)+len(raw)])
        data += raw
    h = json.dumps(header).encode(); h += b' ' * (-len(h) % 8)
    path.write_bytes(struct.pack('<Q', len(h)) + h + data)


class Shards(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(S, 'release_shards implementation missing')
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_raw_bf16_i8_and_metadata_survive_regrouping(self):
        src = self.root/'src';src.mkdir()
        tensor_file(src/'a.safetensors', [('x','BF16',[2],b'\x00\x80\xff\x7f')])
        tensor_file(src/'b.safetensors', [('y','I8',[3],b'\x80\x00\x7f')])
        (src/'model.safetensors.index.json').write_text(json.dumps({'metadata':{'total_size':7},'weight_map':{'x':'a.safetensors','y':'b.safetensors'}}))
        result = S.repack_safetensors(src, self.root/'out', 10000)
        self.assertEqual(result['tensor_count'], 2)
        self.assertEqual(result['payload_bytes'], 7)
        self.assertEqual(len(result['shards']), 1)
        records = S.inventory(self.root/'out')
        self.assertEqual([(r['name'],r['dtype'],r['shape']) for r in records], [('x','BF16',[2]),('y','I8',[3])])
        self.assertEqual([S.read_tensor(r) for r in records],[b'\x00\x80\xff\x7f',b'\x80\x00\x7f'])
        self.assertTrue(result['all_tensor_hashes_match'])
        self.assertEqual(S.header(self.root/'out'/result['shards'][0]['file'])[0]['__metadata__'],{'format':'pt'})
        with self.assertRaises(FileExistsError): S.repack_safetensors(src, self.root/'out',10000)

    def test_out_of_range_and_wrong_dtype_size_rejected(self):
        for entry in [dict(dtype='I8',shape=[1],data_offsets=[0,2]),dict(dtype='I64',shape=[1],data_offsets=[0,1])]:
            p=self.root/'bad.safetensors';h=json.dumps({'x':entry}).encode();p.write_bytes(struct.pack('<Q',len(h))+h+b'x')
            with self.assertRaises(ValueError):S.header(p)

    def test_duplicate_json_keys_rejected(self):
        p=self.root/'bad.safetensors';h=b'{"x":{},"x":{}}';p.write_bytes(struct.pack('<Q',len(h))+h)
        with self.assertRaises(ValueError):S.header(p)

    def test_shards_preserve_chunks_and_reject_corruption(self):
        src=self.root/'chunks';(src/'L0').mkdir(parents=True)
        payloads={'L0/c0.pt':b'one','L0/c16.pt':b'twotwo','L0/c32.pt':b'three'}
        import hashlib
        manifest={k:hashlib.sha256(v).hexdigest() for k,v in payloads.items()}
        for k,v in payloads.items():(src/k).write_bytes(v)
        out=self.root/'packed';r=S.pack_chunks(src,out,manifest,8)
        self.assertEqual(len(r['shards']),3)
        for k,v in payloads.items():self.assertEqual(S.read_chunk(out,k,manifest[k]),v)
        index=json.loads((out/'chunks.index.json').read_text());p=out/index['chunks']['L0/c0.pt']['file'];p.write_bytes(b'bad')
        with self.assertRaises(ValueError):S.read_chunk(out,'L0/c0.pt',manifest['L0/c0.pt'])

    def test_index_path_escape_rejected(self):
        src=self.root/'src';src.mkdir();tensor_file(self.root/'outside.safetensors',[('x','I8',[1],b'a')])
        (src/'model.safetensors.index.json').write_text(json.dumps({'metadata':{'total_size':1},'weight_map':{'x':'../outside.safetensors'}}))
        with self.assertRaises(ValueError):S.inventory(src)

    def test_source_manifest_mismatch_rejected(self):
        src=self.root/'chunks';(src/'L0').mkdir(parents=True);(src/'L0/c0.pt').write_bytes(b'bad')
        with self.assertRaises(ValueError):S.pack_chunks(src,self.root/'out',{'L0/c0.pt':'0'*64},100)


if __name__ == '__main__':unittest.main()
