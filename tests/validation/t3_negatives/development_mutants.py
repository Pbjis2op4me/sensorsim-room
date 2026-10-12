# SPDX-License-Identifier: Apache-2.0
"""Development-only falsification of mechanism controls; never an acceptance runner.

Usage: python development_mutants.py --candidate TREE --output NEW_DIRECTORY
Each candidate is imported in a fresh process. No production files are edited.
The verifier must independently assess the oracle and sign controls before use.
"""
from __future__ import annotations
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

MUTANTS = (
    ('uniform_weights', 'np.cumsum(self.probabilities)',
     'np.cumsum(np.ones(len(self.rows))/len(self.rows))',
     'test_pool.py::test_source_weight_is_consumed_and_cap_is_aggregate'),
    ('no_semantic_review', 'def _review(review, item, *, scope):',
     'def _review(review, item, *, scope):\n    return',
     'test_pool.py::test_semantic_controls'),
    ('no_source_cap', 'if probabilities[mask].sum() > cap + 1e-12:',
     'if False:', 'test_pool.py::test_source_weight_is_consumed_and_cap_is_aggregate'),
    ('wrong_counterfactual', 'floor = mix_events(bed, [empty]+events[1:], replay, cfg).audio',
     'floor = mix_events(bed, [empty], replay, cfg).audio',
     'test_context.py::test_independent_linear_capture_oracle_and_bad_double_capture'),
)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    results=[]
    for name,old,new,test in MUTANTS:
        root=args.output/name;root.mkdir()
        shutil.copytree(args.candidate/'src',root/'src')
        shutil.copytree(args.candidate/'tests',root/'tests')
        shutil.copy2(args.candidate/'pyproject.toml',root/'pyproject.toml')
        module=root/'src/sensorsim/training_negatives.py'
        text=module.read_text()
        if text.count(old)!=1:raise ValueError(f'missing unique mutation hook: {name}')
        command=[sys.executable,'-m','pytest','-q','-p','no:cacheprovider',
                 '--basetemp='+str(root/'pytest'), 'tests/unit/t3_negatives/'+test]
        env=dict(os.environ,PYTHONPATH=str(root/'src')+os.pathsep+str(root))
        pair={}
        for kind in ('correct','bad'):
            if kind=='bad':module.write_text(text.replace(old,new))
            start=datetime.datetime.now(datetime.timezone.utc).isoformat();clock=time.monotonic()
            with (root/(kind+'.stdout')).open('w') as out,(root/(kind+'.stderr')).open('w') as err:
                run=subprocess.run(command,cwd=root,env=env,stdout=out,stderr=err)
            pair[kind]=dict(exit_code=run.returncode,command=command,start=start,
                end=datetime.datetime.now(datetime.timezone.utc).isoformat(),elapsed_s=time.monotonic()-clock,
                imported_module_path=str(module),module_sha256=hashlib.sha256(module.read_bytes()).hexdigest(),
                stdout=kind+'.stdout',stderr=kind+'.stderr')
        results.append(dict(mutant=name,**pair,killed=pair['correct']['exit_code']==0 and pair['bad']['exit_code']==1))
        (root/'run.json').write_text(json.dumps(results[-1],indent=2)+'\n')
    summary=dict(domain='synthetic development proxy',harness='unqualified pending independent review',
                 results=results,survivors=[r['mutant'] for r in results if not r['killed']])
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(dict(killed=sum(r['killed'] for r in results),total=len(results),survivors=summary['survivors'])))
    return int(bool(summary['survivors']))


if __name__=='__main__':
    raise SystemExit(main())
