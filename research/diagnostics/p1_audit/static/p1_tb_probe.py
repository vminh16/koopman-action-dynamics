import json
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
out={}
for seed in (42,43,44):
 paths=sorted(Path(f'outputs/lightning_logs/C02-s{seed}').glob('version_*/events*'))
 versions=[]
 for p in paths:
  ea=EventAccumulator(str(p),size_guidance={'scalars':0});ea.Reload();stats={}
  for tag in ea.Tags()['scalars']:
   if any(t in tag.lower() for t in ('loss','lr','epoch')):
    vals=ea.Scalars(tag)
    stats[tag]={'count':len(vals),'first':vals[0].value,'last':vals[-1].value,'min':min(v.value for v in vals),'max':max(v.value for v in vals),'last_step':vals[-1].step,'samples':[{'step':v.step,'value':v.value} for v in vals[::max(1,len(vals)//8)]]}
  versions.append({'path':str(p),'stats':stats})
 out[str(seed)]=versions
Path('/tmp/p1_tb_probe.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
