# Private data and the instance-retirement checklist

The initial local backup is `paper_data_backup_20261021` beside the repository
in the Interpretable probe workspace. It is intentionally outside Git.

- `local_reproduction_snapshot.tar.gz`: existing local analyses, frozen inputs,
  reports, annotations and the original API response/journal ZIP uploads.
- `LOCAL_MANIFEST.json`: per-file hashes and archive checksum.
- `remote_archives/<experiment>.tar.gz`: original server experiment records,
  prompts, raw batches, failed attempts, split/weight manifests, probe scores,
  fitted artifacts and code, subject to the explicit exclusion manifest.
- `remote_archives/<experiment>.raw.csv.gz`: lossless CSV exports of committed
  response/annotation records. `raw_record_json` retains every original field;
  the other columns are convenient projections, not replacement records.
- `<experiment>.DONE.json` and `REMOTE_VERIFIED.json`: transfer checksums and
  coverage counts. Each archive also contains `BACKUP_MANIFEST.json`.
- `DEPENDENCIES_VERIFIED.json`, when present: targets of archived symlinks that
  were stored outside the selected experiment roots and copied separately.

CSV rows from different experiments can represent reused judgments. **Do not
concatenate them and count them as independent observations.** Use the canonical
selection manifests and source/attempt IDs. A blank convenience column means the
source record did not supply that field; it is not a negative label. Active-run
exports may include committed batch records before a whole-task response file
exists. Preserve their source path when reconciling a later snapshot.

Restore into a new directory:

```bash
python scripts/verify_backup.py --backup /path/to/paper_data_backup_20261021
python scripts/restore_backup.py --backup /path/to/paper_data_backup_20261021 \
  --out /path/to/restored --analysis-only
```

Omit `--analysis-only` to restore the server records as well. Restoration checks
archive checksums, rejects unsafe member paths and avoids writing absolute
symlinks to the original server. It requires additional free space for the
uncompressed files. Raw CSVs can be read directly with pandas or any gzip-aware
CSV reader, without restoring the large archives.

## Before deleting the instance

1. The 21 October missing-cell collection must finish or be explicitly stopped.
2. Take a **new final snapshot** of `paper-complete-20261021`. The current backup
   is a live snapshot and cannot contain responses generated afterward.
3. Verify local hashes against the final snapshot, inspect failed/ambiguous
   coverage, and ensure all required symlink targets resolve locally.
4. Review every excluded artifact. Large activation arrays and model weights
   were omitted; preserve them elsewhere if avoiding regeneration is important.
5. Keep a second copy of the local backup on independent storage. Git contains
   code, not a redundant copy of the research data.

No shutdown, deletion, instance termination, or regeneration is performed by
this repository. A Git push does not establish that the active run is complete.
