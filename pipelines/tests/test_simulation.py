import pandas as pd
import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from pipelines.vendor import simulation


@pytest.mark.parametrize("mode", ["v0", "v1_5"])
@pytest.mark.parametrize("model,family", [("meta-llama/Llama-3.1-8B-Instruct", None), ("local-mistral-adapter", "mistral")])
def test_actual_simulator_context_and_conditioning(mode, model, family, session, tmp_path, monkeypatch):
    requests = []

    def fake_provider(prompt):
        messages = prompt.to_messages() if hasattr(prompt, "to_messages") else prompt
        content = [message.content if hasattr(message, "content") else message["content"] for message in messages]
        requests.append("\n".join(content))
        last = content[-1]
        if "My Tweet:" in last:
            answer = "My Tweet: GENERATED_TWEET"
        elif "select your level of agreement" in last:
            answer = "Lean agree"
        elif "In light of conversations" in last:
            answer = "GENERATED_POST"
        else:
            answer = "My Response: GENERATED_CHAT"
        return AIMessage(content=answer)

    monkeypatch.setattr(simulation, "ChatOpenAI", lambda **kwargs: RunnableLambda(fake_provider))
    simulation.main(api_type="huggingface", model_name=model, model_family=family,
                    data_prefix=session.stem, version=mode, input_path=session,
                    output_root=tmp_path / mode, endpoint_url="http://unused.invalid/v1")
    output = next((tmp_path / mode / "simulation").rglob(f"simulation-{mode}.csv"))
    frame = pd.read_csv(output)
    tweets = frame[frame.event_type == "tweet"]
    assert (tweets[tweets.chat_round_order == 1].llm_text == tweets[tweets.chat_round_order == 1].text).all()
    assert (tweets[tweets.chat_round_order > 1].llm_text == "GENERATED_TWEET").all()
    assert (frame[frame.event_type == "message_sent"].llm_text == "GENERATED_CHAT").all()
    assert (frame[frame.event_type == "Post Opinion"].llm_text == "GENERATED_POST").all()
    assert (frame[frame.event_type == "Post Opinion"].agreement_level == "Lean agree").all()
    assert len(requests) == 24 + 8 + 8  # chat, rounds 2/3 tweets, final text/Likert
    joined = "\n".join(requests)
    assert "HUMAN_TWEET_1_" in joined
    assert "HUMAN_TWEET_2_" not in joined
    assert "HUMAN_TWEET_3_" not in joined
    assert "GENERATED_TWEET" in joined
    if mode == "v0":
        assert "GENERATED_CHAT" in joined
        assert "HUMAN_CHAT_" not in joined
    else:
        assert "HUMAN_CHAT_" in joined
        assert "GENERATED_CHAT" not in joined


def test_local_model_identifier_cannot_escape_output_directory():
    archive = simulation.Agent.get_model_archive_name("/models/checkpoints/custom-adapter")
    assert "/" not in archive
    assert archive not in ("", ".", "..")
