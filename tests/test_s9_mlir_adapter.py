"""Explicit GPU integration tests; CPU CI skips these without generated libs."""
import os
from pathlib import Path
import sys
import pytest

torch=pytest.importorskip('torch')
pytest.importorskip('triton')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from s9_mlir_adapter import MLIRRMSNorm
from s9_numeric_probe import reference_stages


@pytest.fixture
def adapter():
    path=os.environ.get('S9_MLIR_LIBRARIES')
    if not torch.cuda.is_available() or not path:
        pytest.skip('requires CUDA and S9_MLIR_LIBRARIES')
    return MLIRRMSNorm(path)


@torch.inference_mode()
def test_strided_inputs_and_weight(adapter):
    torch.manual_seed(17)
    x=torch.randn(346,3072,device='cuda').bfloat16()[:,::2]
    w=torch.randn(3072,device='cuda').bfloat16()[::2]
    actual=adapter(x,w,1e-6)
    expected=reference_stages(x,w,1e-6)['output']
    assert torch.equal(actual.view(torch.int16),expected.view(torch.int16))
    assert adapter.hits==1 and adapter.fallbacks==0


@torch.inference_mode()
def test_uncompiled_shape_falls_back(adapter):
    x=torch.randn(3,1536,device='cuda').bfloat16()
    w=torch.ones(1536,device='cuda',dtype=x.dtype)
    assert torch.equal(adapter(x,w,1e-6),reference_stages(x,w,1e-6)['output'])
    assert adapter.hits==0 and adapter.fallbacks==1


@torch.inference_mode()
def test_fp16_falls_back(adapter):
    x=torch.randn(1,1536,device='cuda',dtype=torch.float16)
    w=torch.ones(1536,device='cuda',dtype=x.dtype)
    assert torch.equal(adapter(x,w,1e-6),reference_stages(x,w,1e-6)['output'])
    assert adapter.fallbacks==1


@torch.inference_mode()
def test_empty_rows(adapter):
    x=torch.empty(0,1536,device='cuda',dtype=torch.bfloat16)
    w=torch.ones(1536,device='cuda',dtype=x.dtype)
    assert adapter(x,w,1e-6).shape==x.shape
    assert adapter.hits==adapter.fallbacks==0


def test_autograd_rejected(adapter):
    x=torch.zeros(1,1536,device='cuda',dtype=torch.bfloat16)
    with pytest.raises(RuntimeError,match='inference-only'):
        adapter(x,x[0],1e-6)


@torch.inference_mode()
def test_invalid_epsilon_and_dtype(adapter):
    x=torch.zeros(1,1536,device='cuda',dtype=torch.bfloat16)
    for eps in (0,-1,float('nan'),float('inf')):
        with pytest.raises(ValueError):adapter(x,x[0],eps)
    with pytest.raises(ValueError):adapter(x,x[0].float(),1e-6)


@torch.inference_mode()
def test_unaligned_contiguous_row_base(adapter):
    # contiguous() alone does not fix a storage offset of one BF16 element.
    x=torch.randn(346*1536+1,device='cuda').bfloat16()[1:].reshape(346,1536)
    assert x.is_contiguous() and x.data_ptr()%8==2
    w=torch.randn(1536,device='cuda').bfloat16()
    actual,mean,_=adapter.run(x,w,1e-6)
    expected=reference_stages(x,w,1e-6)
    assert torch.equal(mean.view(torch.int32),expected['mean'].view(torch.int32))
    assert torch.equal(actual.view(torch.int16),expected['output'].view(torch.int16))


@pytest.mark.parametrize('rows',[1,2,4,8,16,33])
@torch.inference_mode()
def test_logical_lane_boundaries(adapter,rows):
    # Perturb both sides of every 128-element logical-warp boundary. These
    # inputs distinguish the different ATen block-width schedules.
    torch.manual_seed(4100+rows)
    x=torch.randn(rows,1536,device='cuda').bfloat16()
    x[:,127::128]*=64
    x[:,128::128]*=0.0001
    w=torch.randn(1536,device='cuda').bfloat16()
    actual,mean,kernel=adapter.run(x,w,1e-6)
    expected=reference_stages(x,w,1e-6)
    assert torch.equal(mean.view(torch.int32),expected['mean'].view(torch.int32))
    assert torch.equal(actual.view(torch.int16),expected['output'].view(torch.int16))
    if adapter.manifest[f'{rows}x1536'].get('implementation')=='warp32':
        assert kernel.metadata.num_warps==1


@pytest.mark.parametrize('rows',[1,8,33])
@pytest.mark.parametrize('exponent',[-68,56])
@torch.inference_mode()
def test_subnormal_and_large_finite_squares(adapter,rows,exponent):
    pattern=(torch.arange(1536,device='cuda',dtype=torch.float32)%7+1)/8
    x=(pattern*(2.0**exponent)).bfloat16().repeat(rows,1)
    x[:,::3].neg_()
    w=torch.ones(1536,device='cuda',dtype=x.dtype)
    actual,mean,_=adapter.run(x,w,1e-6)
    expected=reference_stages(x,w,1e-6)
    assert torch.isfinite(mean).all()
    assert torch.equal(mean.view(torch.int32),expected['mean'].view(torch.int32))
    assert torch.equal(actual.view(torch.int16),expected['output'].view(torch.int16))


@torch.inference_mode()
def test_signed_zeros(adapter):
    x=torch.zeros(1,1536,device='cuda',dtype=torch.bfloat16)
    x[:,1::2]=-0.0
    w=torch.ones(1536,device='cuda',dtype=x.dtype)
    actual,mean,_=adapter.run(x,w,1e-6)
    expected=reference_stages(x,w,1e-6)
    assert torch.equal(mean.view(torch.int32),expected['mean'].view(torch.int32))
    assert torch.equal(actual.view(torch.int16),expected['output'].view(torch.int16))


def test_reject_wrong_warp_launch_contract(adapter,tmp_path):
    import json
    entry=dict(next(iter(adapter.manifest.values())))
    entry.update(implementation='warp32',num_warps=4)
    (tmp_path/'libraries.json').write_text(json.dumps({'1x1536':entry}))
    with pytest.raises(ValueError,match='launch contract'):
        MLIRRMSNorm(tmp_path)


@pytest.mark.parametrize('rows',[1,2,4,8,16,33,346,4096])
@torch.inference_mode()
def test_hot_path_matches_diagnostic_path(adapter,rows):
    torch.manual_seed(rows+702)
    x=torch.randn(rows,1536,device='cuda').bfloat16()
    w=torch.randn(1536,device='cuda').bfloat16()
    y=adapter(x,w,1e-6)
    diagnostic,_,_=adapter.run(x,w,1e-6)
    assert torch.equal(y.view(torch.int16),diagnostic.view(torch.int16))


@torch.inference_mode()
def test_one_validation_no_mean_allocation_and_owned_outputs(adapter,monkeypatch):
    x=torch.randn(1,1536,device='cuda').bfloat16()
    w=torch.ones(1536,device='cuda',dtype=x.dtype)
    adapter(x,w,1e-6)  # Compilation is outside the allocation observation.
    counts={'validate':0,'empty':0,'empty_like':0}
    validate=adapter._validate
    empty,empty_like=torch.empty,torch.empty_like
    def checked(*args):
        counts['validate']+=1
        return validate(*args)
    def allocated(*args,**kwargs):
        counts['empty']+=1
        return empty(*args,**kwargs)
    def allocated_like(*args,**kwargs):
        counts['empty_like']+=1
        return empty_like(*args,**kwargs)
    monkeypatch.setattr(adapter,'_validate',checked)
    monkeypatch.setattr(torch,'empty',allocated)
    monkeypatch.setattr(torch,'empty_like',allocated_like)
    y=adapter(x,w,1e-6)
    assert counts=={'validate':1,'empty':0,'empty_like':1}
    saved=y.clone()
    other=adapter(x+1,w,1e-6)
    assert y.data_ptr()!=other.data_ptr() and torch.equal(y,saved)
