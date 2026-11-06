"""Aggregate the supplied 708-session demographic survey without retaining participant IDs."""
import argparse
import ast, collections, csv, hashlib, json, math, statistics
from pathlib import Path

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source-root',type=Path,required=True,help='Directory containing group train/ and test/ CSVs')
parser.add_argument('--output-dir',type=Path,required=True)
args=parser.parse_args()
ROOT=args.source_root
OUT=args.output_dir
OUT.mkdir(parents=True,exist_ok=True)
FIELDS=['age','gender','ethnicity','education','income','occupation','politicalIdentity','politicalViews','bibleBelief','evangelical','religion','maritalStatus','childrenSchool','residence','origin','residency']
MAPPING={'demo_age_fig':'age','demo_gender_fig':'gender','demo_ethnicity_fig':'ethnicity','demo_education_fig':'education','demo_income_fig':'income','demo_occupuation_fig':'occupation','demo_political_party_fig':'politicalIdentity','demo_political_view_fig':'politicalViews','demo_bible_fig':'bibleBelief','demo_evangelical_fig':'evangelical','demo_religion_fig':'religion','demo_marriage_fig':'maritalStatus','demo_children_fig':'childrenSchool','demo_residence_fig':'residence','demo_origin_fig':'origin'}
paths=sorted(ROOT.glob('*/*.csv'))
assert len(paths)==708
subjects=collections.defaultdict(dict); participants=collections.defaultdict(set)
conflicts=collections.Counter(); repeated=collections.Counter(); source_hashes={}; missing_worker=0
for path in paths:
    source_hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    with path.open() as f:
        for row in csv.DictReader(f):
            worker=row.get('worker_id','').strip()
            if not worker:
                missing_worker+=1
                continue
            participants[path.stem].add(worker)
            if row.get('event_type')!='exit_survey': continue
            field=row.get('field','').strip()
            if field not in FIELDS: continue
            value=row.get('text','').strip()
            person=(path.stem,worker)
            if field in subjects[person]:
                if subjects[person][field]!=value: conflicts[field]+=1
                else: repeated[field]+=1
            subjects[person][field]=value

def choices(value):
    if value is None or value.strip().lower() in ('','nan','null'): return []
    if value.startswith('['):
        obj=ast.literal_eval(value)
        assert isinstance(obj,(list,tuple)),value
        return list(dict.fromkeys(str(x).strip() for x in obj if x is not None and str(x).strip().lower() not in ('','nan','null')))
    return [value]

assert not conflicts,dict(conflicts)
all_people={(session,worker) for session,workers in participants.items() for worker in workers}
unique_workers={worker for session,worker in all_people}
respondents={key:values for key,values in subjects.items() if any(choices(v) for v in values.values())}
fields={}; excluded_ages=collections.Counter(); missing_age=0; ages=[]
for values in respondents.values():
    v=values.get('age','')
    try: age=float(v)
    except(ValueError,TypeError): missing_age+=1; continue
    if not math.isfinite(age):missing_age+=1;continue
    if age<18 or age>120: excluded_ages[str(age)]+=1;continue
    ages.append(age)
agecounts=collections.Counter(ages)
fields['age']={'response_n':len(ages),'survey_respondents':len(respondents),'missing_or_non_numeric_n':missing_age,'excluded_out_of_range_n':sum(excluded_ages.values()),'excluded_out_of_range_counts':dict(excluded_ages),'valid_range_rule':'18 <= age <= 120','multi_select':False,'minimum':min(ages),'maximum':max(ages),'mean':statistics.mean(ages),'sample_sd':statistics.stdev(ages),'population_sd':statistics.pstdev(ages),'median':statistics.median(ages),'categories':[{'label':str(int(k) if k.is_integer() else k),'value':k,'count':v,'percent':100*v/len(ages)} for k,v in sorted(agecounts.items())]}
for field in FIELDS[1:]:
    counts=collections.Counter();response_n=0;multi=False;list_encoded=False
    for person,values in respondents.items():
        raw=values.get(field,'')
        vals=choices(raw)
        if vals:
            response_n+=1
            counts.update(vals)
            multi |= len(vals)>1
            list_encoded |= raw.startswith('[')
    fields[field]={'response_n':response_n,'survey_respondents':len(respondents),'missing_or_empty_n':len(respondents)-response_n,'multi_select':multi,'list_encoded':list_encoded,'selection_count':sum(counts.values()),'categories':[{'label':label,'count':n,'percent':100*n/response_n} for label,n in counts.most_common()]}

result={'date':'2026-11-05','source_root':str(ROOT),'source_files':len(paths),'selection':'Union of supplied original Ruixuan group train and test CSVs; all 708 sessions.','deduplication':'One value per session, worker_id, demographic field; repeated identical source rows collapse. No conflicting values were found.','unique_worker_ids_across_all_events':len(unique_workers),'participant_session_records':len(all_people),'participants_per_session_histogram':dict(collections.Counter(map(len,participants.values()))),'worker_ids_appearing_in_multiple_sessions':sum(v>1 for v in collections.Counter(worker for session,worker in all_people).values()),'rows_missing_worker_id':missing_worker,'survey_respondents_any_item':len(respondents),'survey_unique_worker_ids':len({key[1] for key in respondents}),'survey_participation_percent':100*len(respondents)/len(all_people),'participation_definition':'At least one nonempty demographic response; does not imply a complete questionnaire.','field_denominator_definition':'For each item, denominator is the number of respondents providing a nonempty value; age additionally excludes nonnumeric or out-of-range values. Multi-select percentages use respondents, not number of selections.','repeated_identical_field_rows':dict(repeated),'conflicting_field_values':dict(conflicts),'figure_field_mapping':MAPPING,'fields':fields}
result['recommended_text_values']={'participants':len(all_people),'survey_respondents':len(respondents),'survey_percent_1dp':f'{result["survey_participation_percent"]:.1f}','age_n':len(ages),'age_range':f'{min(ages):.0f}--{max(ages):.0f}','age_mean_1dp':f'{statistics.mean(ages):.1f}','age_sample_sd_1dp':f'{statistics.stdev(ages):.1f}','gender_percent_1dp':{x['label']:f"{x['percent']:.1f}" for x in fields['gender']['categories']},'ethnicity_percent_1dp':{x['label']:f"{x['percent']:.1f}" for x in fields['ethnicity']['categories']},'evangelical_percent_1dp':{x['label']:f"{x['percent']:.1f}" for x in fields['evangelical']['categories']}}
(OUT/'aggregates.json').write_text(json.dumps(result,indent=2)+'\n')
(OUT/'source_hashes.json').write_text(json.dumps(source_hashes,indent=2)+'\n')
for field,record in fields.items():
    with (OUT/f'{field}.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['label','count','percent','response_n'])
        writer.writeheader()
        writer.writerows({'label':c['label'],'count':c['count'],'percent':c['percent'],'response_n':record['response_n']} for c in record['categories'])
print(json.dumps({k:v for k,v in result.items() if k not in ('fields','repeated_identical_field_rows')},indent=2))
print('Field denominators:',json.dumps({field:{k:record.get(k) for k in ['response_n','missing_or_empty_n','multi_select','selection_count']} for field,record in fields.items()},indent=2))
