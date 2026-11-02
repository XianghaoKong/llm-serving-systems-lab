from pathlib import Path
import sys
import pytest
torch=pytest.importorskip('torch')
transformers=pytest.importorskip('transformers')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from s9_decode_graph import ExactDecodeGraph,_ExactLayer

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='requires CUDA')


@torch.inference_mode()
def test_exact_layer_uses_disjoint_contiguous_outputs():
    k=torch.randn(1,2,7,16,device='cuda',dtype=torch.bfloat16)
    v=torch.randn_like(k);newk=torch.randn_like(k[:,:,:1]);newv=torch.randn_like(newk)
    outk=torch.empty(1,2,8,16,device='cuda',dtype=k.dtype);outv=torch.empty_like(outk)
    saved=k.clone();layer=_ExactLayer(k,v,outk,outv)
    assert layer.get_seq_length()==7
    actualk,actualv=layer.update(newk,newv)
    assert actualk.data_ptr()==outk.data_ptr() and actualk.is_contiguous()
    assert torch.equal(actualk,torch.cat((k,newk),dim=-2))
    assert torch.equal(actualv,torch.cat((v,newv),dim=-2))
    assert torch.equal(k,saved) and layer.get_seq_length()==8


@pytest.mark.parametrize('length',[7,8])
@torch.inference_mode()
def test_repeated_real_greedy_requests_and_lifecycle(length):
    torch.manual_seed(522)
    config=transformers.Qwen2Config(vocab_size=128,hidden_size=64,intermediate_size=128,
        num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=64)
    config._attn_implementation='sdpa'
    model=transformers.Qwen2ForCausalLM(config).to(device='cuda',dtype=torch.bfloat16).eval()
    p=torch.randint(0,128,(1,length),device='cuda')
    runner=ExactDecodeGraph(model,p,5)
    with pytest.raises(RuntimeError):runner.step(p[:,:1])
    for _ in range(2):
        prompt=torch.randint(0,128,(1,length),device='cuda')
        expected=[];cache=None;tokens=prompt
        for i in range(6):
            output=model(input_ids=tokens,past_key_values=cache,use_cache=True,logits_to_keep=1)
            expected.append(output.logits.clone());cache=output.past_key_values
            tokens=output.logits[:,-1,:].argmax(-1,keepdim=True)
        actual=runner.begin(prompt)
        assert torch.equal(actual.view(torch.int16),expected[0].view(torch.int16))
        with pytest.raises(RuntimeError):runner.begin(prompt)
        with pytest.raises(ValueError):runner.step(prompt)
        for i in range(1,6):
            actual=runner.step(actual[:,-1,:].argmax(-1,keepdim=True))
            assert torch.equal(actual.view(torch.int16),expected[i].view(torch.int16))
        with pytest.raises(RuntimeError):runner.step(tokens)
    assert runner.replays==10 and runner.prefills==2
    with pytest.raises(ValueError):runner.begin(p.to(torch.int32))
    with torch.enable_grad():
        with pytest.raises(RuntimeError):runner.begin(p)
    with torch.inference_mode(False),torch.no_grad():
        with pytest.raises(RuntimeError,match='inference_mode'):runner.begin(p)
    model.train()
    with pytest.raises(RuntimeError,match='eval'):runner.begin(p)
    model.eval()
    with torch.cuda.stream(torch.cuda.Stream()):
        with pytest.raises(RuntimeError):runner.begin(p)
    runner.begin(p);runner.reset();runner.begin(p);runner.reset()
