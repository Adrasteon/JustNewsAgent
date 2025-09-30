#!/usr/bin/env python3
"""Safely remove temporary archive helper files created during the archive-detection workflow.
This script is idempotent and logs to /tmp/archive_cleanup.log. It only removes a conservative
set of files so it won't touch other repo data.
"""
from pathlib import Path

log = Path('/tmp/archive_cleanup.log')
with log.open('a') as fo:
    fo.write('CLEANUP START\n')
    candidates = [
        'scripts/archive_dryrun_runner.py',
        'scripts/generate_archive_candidates.py',
        '/tmp/archive_candidate_paths.txt',
        '/tmp/archive_candidates.tsv',
        '/tmp/archive_generator.log',
        '/tmp/archive_dryrun.log',
        '/tmp/archive_dryrun.pid',
        '/tmp/archive_generator.pid',
        '/tmp/archive_dryrun.meta',
        '/tmp/archive_dryrun_report.txt',
    ]
    for p in candidates:
        try:
            if Path(p).exists():
                if Path(p).is_dir():
                    fo.write(f'RM_DIR {p}\n')
                    for child in Path(p).glob('**/*'):
                        try:
                            if child.is_file():
                                child.unlink()
                                fo.write(f'RM_FILE {child}\n')
                        except Exception as e:
                            fo.write(f'ERR_REMOVING {child} {e}\n')
                else:
                    Path(p).unlink()
                    fo.write(f'REMOVED {p}\n')
            else:
                fo.write(f'NOT_FOUND {p}\n')
        except Exception as e:
            fo.write(f'ERROR {p} {e}\n')
    fo.write('CLEANUP END\n')
print('Started cleanup, see /tmp/archive_cleanup.log')
