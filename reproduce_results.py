#!/usr/bin/env python3
"""Replot the paper without inference, or explicitly start a new collection."""
import argparse, hashlib, json, os, sys, tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR',str(Path(tempfile.gettempdir())/'nn-paper-matplotlib'))
os.environ['MPLBACKEND']='Agg'
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
 os.environ.setdefault(name,'2')
def verify():
 manifest=json.loads((ROOT/'data/paper/MANIFEST.json').read_text())
 for name,record in manifest['files'].items():
  p=ROOT/name
  if not p.exists() or hashlib.file_digest(p.open('rb'),'sha256').hexdigest()!=record['sha256']:
   raise ValueError(f'Frozen input missing or modified: {name}')
 return len(manifest['files'])
def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--out',type=Path,default=ROOT/'outputs/paper')
 ap.add_argument('--recompute',action='store_true',help='Recompute statistics from case-level scores/labels before plotting (CPU only).')
 ap.add_argument('--query-models',type=Path,metavar='CONFIG',help='Run fresh inference using explicit endpoint/model configuration.')
 ap.add_argument('--dry-run',action='store_true',help='Show collection counts without model calls.')
 ap.add_argument('--studies',default='main,two_nn,ablation,mechanism,reason_after,trace_labels')
 ap.add_argument('--limit',type=int,help='Limit new requests per model and study for a smoke test.')
 ap.add_argument('--no-appendix',action='store_true',help='Skip the eight additional judge/genre comparison grids.')
 a=ap.parse_args();a.out=a.out.resolve()
 if a.query_models:
  from paper.collect import collect
  collect(a.query_models.resolve(),a.out,a.studies.split(','),a.limit,a.dry_run)
  return
 if a.dry_run or a.limit is not None:ap.error('--dry-run/--limit require --query-models')
 checked=verify();data=ROOT/'data/paper/plots'
 if a.recompute:
  from paper.recompute import recompute
  data=recompute(ROOT/'data/paper',a.out/'recomputed')
 from paper.render import render
 render(data,a.out,appendix=not a.no_appendix)
 (a.out/'REPRODUCED.json').write_text(json.dumps({'input_files_verified':checked,'recomputed':a.recompute,'inference_calls':0,'python':sys.version,'reference_pdf_sha256':json.loads((ROOT/'data/paper/MANIFEST.json').read_text())['reference_pdf_sha256']},indent=2)+'\n')
 print(f'Figures: {a.out / "figures.pdf"}\nIndividual PDF, SVG and PNG files: {a.out}')
if __name__=='__main__':main()
