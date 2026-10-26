import hashlib, json, lzma, tempfile, unittest, zipfile
from pathlib import Path
from scripts.verify_raw_supplement import verify
from scripts.audit_raw_prompt_coverage import audit

class RawSupplementTests(unittest.TestCase):
    def package(self, folder, member='responses.csv', payload=b'prompt_id,response\na,75\n', recorded_hash=None):
        zpath=folder/'records.zip'
        with zipfile.ZipFile(zpath,'w') as z:
            z.writestr(member,payload)
            z.writestr('MEMBERS.json',json.dumps([dict(path=member,bytes=len(payload),
                sha256=recorded_hash or hashlib.sha256(payload).hexdigest())]))
        (folder/'RAW_DATA_MANIFEST.json').write_text(json.dumps({'assets':[
            dict(name=zpath.name,bytes=zpath.stat().st_size,
                 sha256=hashlib.sha256(zpath.read_bytes()).hexdigest())]}))
        return zpath

    def test_valid_deep_verification(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);self.package(p)
            self.assertEqual(verify(p,True)['members'],1)

    def test_changed_asset_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);z=self.package(p);z.write_bytes(z.read_bytes()+b'changed')
            with self.assertRaisesRegex(ValueError,'Wrong size'):verify(p)

    def test_changed_member_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);self.package(p,recorded_hash='0'*64)
            with self.assertRaisesRegex(ValueError,'Member checksum'):verify(p,True)

    def test_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);self.package(p,member='../outside.csv')
            with self.assertRaisesRegex(ValueError,'Unsafe member'):verify(p,True)

    def test_coverage_keeps_failures_duplicates_and_missing_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            catalogue=p/'catalogue.xz'
            with lzma.open(catalogue,'wt') as f:
                f.write(json.dumps(['p',['known','main','[]',0,'test','nearest',None]])+'\n')
            with zipfile.ZipFile(p/'raw-responses-001.zip','w') as z:
                z.writestr('api/responses.csv',
                    'prompt_id,status\nknown,ok\nknown,blocked\nmissing,invalid\n,error\n')
            result=audit(p,catalogue)['response_tables'][0]
            self.assertEqual(result['rows'],4)
            self.assertEqual(result['matched_rows'],2)
            self.assertEqual(result['unmatched_rows'],1)
            self.assertEqual(result['blank_identifier_rows'],1)
            self.assertEqual(result['statuses']['blocked'],1)

if __name__=='__main__':unittest.main()
