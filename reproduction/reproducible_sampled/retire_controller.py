import os,signal,time
from common import *
pid=393599;proc=Path('/proc')/str(pid)
if proc.exists():
 assert b'/gemma31b-backbone-20261003/judge_stage.py' in (proc/'cmdline').read_bytes()
 assert '\nState:\tT' in (proc/'status').read_text()
 children=(proc/'task'/str(pid)/'children').read_text().split()
 assert all('\nState:\tZ' in (Path('/proc')/c/'status').read_text() for c in children),'Live child; leave stopped and inspect'
 os.kill(pid,signal.SIGTERM);os.kill(pid,signal.SIGCONT)
write(ROOT/'OLD_CONTROLLER_RETIRED.json',dict(pid=pid,time=time.time(),note='Only completed zombie child remained; preserved all outputs, keepalives untouched'))
