"""An offline round trip through the real simulator, annotator and metric core."""
import csv
import json
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from pipelines import annotate
from pipelines.vendor import simulation
from pipelines.tests.conftest import write_session, PREFIX
from reproduction import evaluate


@pytest.mark.parametrize("mode",["v0","v1_5"])
def test_generation_annotation_metrics(mode,tmp_path,monkeypatch):
    source=write_session(tmp_path/(PREFIX+".csv"))
    generated_posts=[]
    def fake_model(prompt):
        last=prompt.to_messages()[-1].content
        if "My Tweet:" in last:
            text="My Tweet: GENERATED_TWEET"
        elif "select your level of agreement" in last:
            text="Lean agree"
        elif "In light of conversations" in last:
            generated_posts.append(len(generated_posts)+1)
            text="GENERATED_POST_"+str(generated_posts[-1])
        else:
            text="My Response: GENERATED_CHAT"
        return AIMessage(content=text)
    monkeypatch.setattr(simulation,"ChatOpenAI",lambda **kwargs:RunnableLambda(fake_model))
    simulation.main(api_type="huggingface",model_name="meta-llama/Llama-3.1-8B-Instruct",data_prefix=PREFIX,version=mode,
                    input_path=source,output_root=tmp_path/"sim",endpoint_url="http://unused.invalid/v1")
    generated=next((tmp_path/"sim/simulation").rglob(f"simulation-{mode}.csv"))
    phase={"source":"human","post_calls":0}
    def judge(**payload):
        messages=payload["messages"]
        if len(messages)==1:
            # Judge the target message, not the earlier conversation quoted in
            # the same prompt. Generated text has stance4; human text stance5.
            target=messages[0]["content"].rsplit('Here is the message: "',1)[-1]
            answer="Lean agree" if target.startswith("GENERATED_") else "Probably agree"
        else:
            is_model_post=phase["source"]=="llm" and "# Post Opinion Validity Judge" in messages[0]["content"]
            if is_model_post:
                phase["post_calls"]+=1
            answer="INVALID" if is_model_post and phase["post_calls"]==1 else "VALID"
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=answer))])
    import openai
    monkeypatch.setattr(openai,"OpenAI",lambda **kwargs:SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=judge))))
    labels={}
    for who,input_file in (("human",source),("llm",generated)):
        phase["source"]=who
        labels[who]=tmp_path/(who+"_arbitrary_filename.csv")
        annotate.main(["--input",str(input_file),"--output",str(labels[who]),"--cache",str(tmp_path/(who+"_cache.jsonl")),
                       "--source",who,"--topic","Public libraries are useful"])
    out=tmp_path/"metrics"
    evaluate.main(["--human",str(labels["human"]),"--llm",str(labels["llm"]),"--session",PREFIX,
                   "--simulation-mode",mode,"--model","synthetic-model","--output-dir",str(out),"--bootstrap-reps","20"])
    row=next(csv.DictReader((out/"summary_metrics.csv").open()))
    assert float(row["public_stance_alignment_abs"])==1
    assert float(row["private_stance_alignment_abs"])==1
    assert float(row["public_stance_change_error_abs"])==1
    assert float(row["private_stance_change_error_abs"])==1
    assert float(row["public_opinion_convergence_error"])==0
    assert float(row["private_opinion_convergence_error"])==0
    assert int(row["public_alignment_count"])==8
    assert int(row["private_alignment_count"])==3  # INVALID model post excluded despite arbitrary filename.
    manifest=json.loads((out/"evaluation_manifest.json").read_text())
    assert manifest["inputs"][0]["alignment"]=="key"
    assert manifest["inputs"][0]["key_match_rate"]==1
    assert manifest["inputs"][0]["llm_rows"]["excluded_by_validity"]==1
    assert manifest["validity_protocols"]==["generated-text"]


def test_bridge_rejects_inherited_human_validity_for_llm(tmp_path):
    from reproduction.run import load_renderer
    load_renderer()
    import generate_plots as core
    path=write_session(tmp_path/"stance_only.csv")
    with pytest.raises(ValueError,match="missing source-specific llm_validity"):
        evaluate.load_labels(path,source="llm",session=PREFIX,validity_policy="valid-only",core=core)


def test_bridge_excludes_unknown_even_with_inherited_valid_field(tmp_path):
    from reproduction.run import load_renderer
    from pipelines.tests.conftest import synthetic_rows
    load_renderer()
    import generate_plots as core
    rows=synthetic_rows()
    for row in rows:
        row.update(likert_pred="5",llm_validity="UNKNOWN")
    path=write_session(tmp_path/"unknown.csv",rows)
    labels,counts,_=evaluate.load_labels(path,source="llm",session=PREFIX,validity_policy="valid-only",core=core)
    assert labels and all(row["likert_pred"]=="" for row in labels)
    assert counts["excluded_by_validity"]==len(labels)
