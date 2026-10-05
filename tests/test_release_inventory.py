"""Coverage checks include measured nonvideo data and package-local aliases."""
import csv
import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/verify_public_data.py'
spec = importlib.util.spec_from_file_location('verify_public_data', SCRIPT)
verify_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verify_module)

class ReleaseInventoryTest(unittest.TestCase):
    def test_nonvideo_alias_and_tamper_detection(self):
        previous = verify_module.EXPECTED
        verify_module.EXPECTED = {}
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                (root/'metadata').mkdir()
                (root/'raw_data/fabrication_characterization').mkdir(parents=True)
                source=root/'raw_data/fabrication_characterization/measurement.csv'
                source.write_text('depth\n1.5\n')
                (root/'processed_data').mkdir()
                (root/'processed_data/results.csv').write_text('score\n0.8\n')
                (root/'analysis_ready_data').symlink_to('processed_data',target_is_directory=True)
                (root/'MANIFEST.csv').write_text('dataset,release_path,file_size,sha256\n')
                with (root/'metadata/characterization_file_manifest.csv').open('w') as f:
                    w=csv.writer(f);w.writerow(['release_path','size_bytes','sha256'])
                    w.writerow([str(source.relative_to(root)),source.stat().st_size,verify_module.digest(source)])
                names=['MANIFEST.csv','metadata/characterization_file_manifest.csv',
                       'raw_data/fabrication_characterization/measurement.csv',
                       'processed_data/results.csv','analysis_ready_data/results.csv']
                (root/'CHECKSUMS.sha256').write_text(''.join(verify_module.digest(root/n)+'  '+n+'\n' for n in names))
                self.assertEqual(verify_module.verify(root)['integrity_status'],'PASS')
                extra=root/'raw_data/fabrication_characterization/unlisted.csv';extra.write_text('2\n')
                self.assertIn('manifest_coverage',[e['quantity'] for e in verify_module.verify(root)['errors']])
                extra.unlink();source.write_text('depth\n9.5\n')
                self.assertIn('characterization_integrity',[e['quantity'] for e in verify_module.verify(root)['errors']])
        finally:
            verify_module.EXPECTED=previous

if __name__=='__main__':unittest.main()
