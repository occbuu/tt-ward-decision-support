"""Audit returned LLM records without modifying server outputs; save independent metrics."""
from collections import Counter
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import re
import statistics

HERE = Path(__file__).resolve().parent
P4 = HERE.parent
BUNDLE = HERE
OUT = P4 / 'results' / 'llm_benchmark_20261008'
OUT.mkdir(parents=True, exist_ok=True)
spec = importlib.util.spec_from_file_location('benchmark', BUNDLE / 'benchmark.py')
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)

def csv_read(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def csv_write(path, rows):
    with path.open('w',encoding='utf-8',newline='') as f:
        w = csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

def independent_metrics(actual, pred, labels):
    counts = Counter(zip(actual,pred))
    n = len(actual)
    correct = sum(counts[c,c] for c in labels)
    support = Counter(actual)
    assigned = Counter(pred)
    chance = sum(support[c]*assigned[c] for c in labels) / n**2
    f1 = [2*counts[c,c]/(support[c]+assigned[c]) if support[c]+assigned[c] else 0 for c in labels]
    return {'accuracy':correct/n,'macro_f1':statistics.mean(f1),
            'cohen_kappa':(correct/n-chance)/(1-chance) if chance<1 else None,
            'invalid_or_failed':sum(p not in labels for p in pred)}

items, frame = b.load_inputs()
ids = [i['seg_id'] for i in items]
original_sample = {r['seg_id']:r for r in csv_read(P4/'human_coding'/'gold_sample.csv')}
original_gold = csv_read(P4/'human_coding'/'results'/'final_gold_standard_single_label_130.csv')
gold_rows = csv_read(BUNDLE/'data/gold_labels.csv')
assert gold_rows == original_gold, 'Exported gold changed'
for item in items:
    assert all(item[k]==original_sample[item['seg_id']][k] for k in item), 'Test input changed'
gold = {r['seg_id']:r for r in gold_rows}
actual = [gold[i]['gold_code'] for i in ids]
codes = [r['code'] for r in frame]
support = Counter(actual)
supported = [c for c in codes if support[c]]
prompt_groups = {}
for j,item in enumerate(items):
    key = tuple(re.sub(r'\s+',' ',item[k].strip()).casefold() for k in ('field_desc','text'))
    prompt_groups.setdefault(key,[]).append(j)
clusters = list(prompt_groups.values())
conflicts = [{'seg_ids':[ids[j] for j in group],'gold_counts':dict(Counter(actual[j] for j in group))}
             for group in clusters if len({actual[j] for j in group})>1]
summary, runs, class_rows, error_rows, subgroup_rows, metadata = [],[],[],[],[],[]
predictions_by_model = {}
hashes = []

for folder in sorted((BUNDLE/'runs').iterdir()):
    if not folder.is_dir() or not (folder/'manifest.json').exists():
        continue
    manifest = json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
    identity = manifest['identity']
    config = identity['config']
    model = config['model']
    assert manifest['status']=='complete' and manifest['n_items']==130
    assert manifest['gold_used_for_inference'] is False
    assert identity['input_sha256']==b.digest(BUNDLE/'data/test_inputs.csv')
    assert identity['frame_sha256']==b.digest(BUNDLE/'data/coding_frame.csv')
    assert identity['runner_sha256']==b.digest(BUNDLE/'benchmark.py')
    assert not identity['examples'] and config['strategies']==['zero_shot'] and config['repeats']==3
    evaluation = json.loads((folder/'evaluation.json').read_text(encoding='utf-8'))
    assert evaluation['gold_sha256']==b.digest(BUNDLE/'data/gold_labels.csv')
    vectors, raw_vectors, stats = [],[],[]
    for repeat in (1,2,3):
        file = folder/f'zero_shot_run{repeat}.jsonl'
        rows = [json.loads(line) for line in file.read_text(encoding='utf-8').splitlines()]
        assert len(rows)==130 and len({r['seg_id'] for r in rows})==130 and {r['seg_id'] for r in rows}==set(ids)
        records = {r['seg_id']:r for r in rows}
        pred = []
        reasons = Counter()
        raw_texts = []
        response_ids = []
        for item in items:
            record = records[item['seg_id']]
            assert record['strategy']=='zero_shot' and record['repeat']==repeat
            assert record['messages']==b.prompt(item,frame,'zero_shot',[]), 'Prompt/input mismatch'
            response = record['response']
            assert response['model']==model, 'Response model mismatch'
            choice = response['choices'][0]
            raw = choice['message']['content']
            assert raw==record['raw_text'], 'Raw response differs'
            expected = raw.strip() if raw.strip() in codes else 'INVALID'
            if choice.get('finish_reason')=='length':
                expected='INVALID'
            assert record['predicted_code']==expected, 'Parser mismatch'
            assert record['attempts'][-1]['status']=='ok'
            pred.append(expected)
            raw_texts.append(raw)
            response_ids.append(response['id'])
            reasons[choice.get('finish_reason')]+=1
            if expected!=gold[item['seg_id']]['gold_code']:
                error_rows.append({'model':model,'repeat':repeat,'seg_id':item['seg_id'],
                                  'field':item['field'],'gold_code':gold[item['seg_id']]['gold_code'],
                                  'predicted_code':expected,'finish_reason':choice.get('finish_reason'),
                                  'empty_content':not bool(raw),'reasoning_present':bool(choice['message'].get('reasoning')),
                                  'completion_tokens':response.get('usage',{}).get('completion_tokens'),
                                  'label_source':gold[item['seg_id']]['label_source'],'text':item['text']})
        scores = independent_metrics(actual,pred,codes)
        saved = next(r for r in evaluation['runs'] if r['repeat']==repeat and r['strategy']=='zero_shot')
        for metric in scores:
            assert abs(scores[metric]-saved[metric])<1e-12
        if repeat==1:
            rng=random.Random(evaluation['bootstrap_seed'])
            boots=[]
            for _ in range(evaluation['bootstrap_replicates']):
                index=rng.choices(range(130),k=130)
                boots.append(independent_metrics([actual[j] for j in index],[pred[j] for j in index],codes))
            ci={m:[b.quantile([s[m] for s in boots],.025),b.quantile([s[m] for s in boots],.975)] for m in ('accuracy','macro_f1','cohen_kappa')}
            for m in ci:
                assert all(abs(x-y)<1e-12 for x,y in zip(ci[m],saved['ci95'][m]))
            rng=random.Random(evaluation['bootstrap_seed'])
            cluster_boot=[]
            for _ in range(evaluation['bootstrap_replicates']):
                index=[j for group in rng.choices(clusters,k=len(clusters)) for j in group]
                cluster_boot.append(independent_metrics([actual[j] for j in index],[pred[j] for j in index],codes))
            cluster_ci={m:[b.quantile([s[m] for s in cluster_boot],.025),b.quantile([s[m] for s in cluster_boot],.975)]
                        for m in ('accuracy','macro_f1','cohen_kappa')}
            _,per_class=b.metrics(actual,pred,codes)
            class_rows.extend({'model':model,**r} for r in per_class)
            counts=Counter(zip(actual,pred))
            for dimension in ('field','label_source'):
                groups=sorted({i['field'] if dimension=='field' else gold[i['seg_id']]['label_source'] for i in items})
                for group in groups:
                    index=[j for j,i in enumerate(items) if (i['field'] if dimension=='field' else gold[i['seg_id']]['label_source'])==group]
                    s=independent_metrics([actual[j] for j in index],[pred[j] for j in index],codes)
                    subgroup_rows.append({'model':model,'dimension':dimension,'group':group,'n':len(index),**s})
            top_errors=counts.most_common()
            print(model, 'correct', sum(a==p for a,p in zip(actual,pred)), 'scores', scores,
                  'supported7_f1', independent_metrics(actual,pred,supported)['macro_f1'],
                  'wrong_pairs', [(g,p,n) for (g,p),n in top_errors if g!=p][:5])
        stats.append(scores)
        vectors.append(pred)
        raw_vectors.append(raw_texts)
        runs.append({'model':model,'repeat':repeat,**scores,'finish_reasons':dict(reasons),
                     'unique_response_ids':len(set(response_ids)), 'n':130})
        hashes.append({'file':str(file.relative_to(P4)), 'sha256':b.digest(file)})
    assert vectors[0]==vectors[1]==vectors[2], 'Replicate predictions changed: adjust summary'
    predictions_by_model[model]=vectors[0]
    record={'model':model,'correct':sum(a==p for a,p in zip(actual,vectors[0])),'n':130,
            **stats[0],'macro_f1_supported7':independent_metrics(actual,vectors[0],supported)['macro_f1'],
            **{f'{m}_{side}':ci[m][j] for m in ci for j,side in enumerate(('low','high'))},
            **{f'cluster_{m}_{side}':cluster_ci[m][j] for m in cluster_ci for j,side in enumerate(('low','high'))},
            'prediction_disagreements_between_repeats':0,
            'raw_content_identical_between_repeats':raw_vectors[0]==raw_vectors[1]==raw_vectors[2]}
    summary.append(record)
    metadata.append({'model':model,'started_utc':manifest['started_utc'],'finished_utc':manifest['finished_utc'],
                     'request_options':config['request_options'], **config['server_metadata']})
summary.sort(key=lambda r:r['accuracy'],reverse=True)
report={'checks':'All 21 runs match original inputs and gold, exact prompts, response labels, file hashes and independent score/CI recomputation.',
        'unique_test_items':130,'models':len(summary),'requests':len(runs)*130,
        'normalized_question_text_groups':len(clusters),'duplicate_group_size_distribution':dict(Counter(map(len,clusters))),
        'conflicting_duplicate_gold_groups':conflicts,
        'gold_support':dict(support),'runs':runs,'models_summary':summary,'model_metadata':metadata,'raw_file_hashes':hashes}
(OUT/'audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
csv_write(OUT/'model_comparison.csv',summary)
csv_write(OUT/'per_class.csv',class_rows)
csv_write(OUT/'errors.csv',error_rows)
csv_write(OUT/'subgroups.csv',subgroup_rows)
csv_write(OUT/'model_metadata.csv',[{**m,'request_options':json.dumps(m['request_options'])} for m in metadata])
csv_write(OUT/'gold_support.csv',[{'code':c,'n':support[c]} for c in codes])
csv_write(P4/'tables'/'T21_llm_model_comparison.csv',summary)
csv_write(P4/'tables'/'T22_llm_gold_support.csv',[{'code':c,'n':support[c]} for c in codes])
print('ALL CHECKS PASSED', len(runs), 'runs;', len(runs)*130, 'records')
print('GOLD SUPPORT',dict(support))
print('CLUSTERS',len(clusters),'CONFLICTS',conflicts)
print('TOP CLUSTER CI',[(r['model'],r['cluster_accuracy_low'],r['cluster_accuracy_high']) for r in summary[:2]])
