"""Exact-length decode graphs with double-buffered, contiguous KV caches.

One graph per decode position preserves the eager SDPA shape/algorithm. Cache
capacity is not exposed as an attention length; no padded masked attention is
introduced. This is an inference-only, batch-one, single-stream request runner.
"""
import time
import torch
from transformers.cache_utils import Cache,DynamicLayer


class _ExactLayer(DynamicLayer):
    def __init__(self,keys,values,next_keys,next_values):
        super().__init__()
        self.keys,self.values=keys,values
        self.next_keys,self.next_values=next_keys,next_values
        self.dtype,self.device=keys.dtype,keys.device
        self.is_initialized=True

    def update(self,key_states,value_states,cache_kwargs=None):
        # cat is a data movement operation. The out tensors have exactly the
        # contiguous shape/strides of ordinary DynamicCache outputs.
        torch.cat((self.keys,key_states),dim=-2,out=self.next_keys)
        torch.cat((self.values,value_states),dim=-2,out=self.next_values)
        self.keys,self.values=self.next_keys,self.next_values
        return self.keys,self.values


class ExactDecodeGraph:
    """Prepared bucket for one prompt length and a bounded decode horizon.

    begin() runs a fresh eager prefill and overwrites the initial KV state.
    step(token) consumes a caller-supplied GPU token and advances one position.
    Returned logits are borrowed buffers, overwritten on reuse of that graph.
    Call begin/step/reset under torch.inference_mode(). No concurrent/interleaved
    requests or stream changes are permitted; the model must remain in eval.
    Re-prepare after changing model weights, dtype, device, or forward backend.
    """
    @torch.inference_mode()
    def __init__(self,model,example_prompt,decode_steps=31):
        if model.training or decode_steps<1:
            raise ValueError('eval model and positive decode horizon required')
        if (example_prompt.ndim!=2 or example_prompt.shape[0]!=1 or
            example_prompt.shape[1]<1 or example_prompt.dtype!=torch.long or not example_prompt.is_cuda):
            raise ValueError('nonempty batch-one CUDA int64 prompt required')
        if model.config.model_type!='qwen2' or model.config._attn_implementation!='sdpa':
            raise ValueError('this exact-shape runner supports Qwen2 SDPA only')
        if model.device!=example_prompt.device or model.dtype!=torch.bfloat16:
            raise ValueError('BF16 model and prompt must share a CUDA device')
        self.model=model
        self.device=example_prompt.device
        self.prompt_length=example_prompt.shape[1]
        self.decode_steps=decode_steps
        self.stream=torch.cuda.current_stream(self.device)
        self.token=torch.zeros((1,1),device=self.device,dtype=torch.long)
        self.graphs=[];self.logits=[];self.buffers=[]
        self._next=None;self.replays=0;self.prefills=0
        start=time.perf_counter_ns()
        pre=model(input_ids=example_prompt,use_cache=True,logits_to_keep=1)
        if any(layer.is_sliding for layer in pre.past_key_values.layers):
            raise ValueError('sliding-window cache is not supported by this exact-length runner')
        self.shapes=[tuple(layer.keys.shape) for layer in pre.past_key_values.layers]
        # Every layer can be checked independently; do not rely on a loop's
        # last cache object for dtype or head geometry.
        for shape,layer in zip(self.shapes,pre.past_key_values.layers):
            b,h,_,d=shape
            if layer.keys.dtype!=torch.bfloat16 or b!=1:
                raise ValueError('BF16 batch-one cache required')
            size=b*h*(self.prompt_length+decode_steps)*d
            self.buffers.append(tuple(torch.empty(size,device=self.device,dtype=layer.keys.dtype) for _ in range(4)))
        self._copy_prefill(pre.past_key_values)
        # Warm every exact shape before graph capture. Only persistent buffers
        # survive; the KV footprint is O(max_context), not O(graphs*context).
        for i in range(decode_steps):
            cache=self._cache(i)
            model(input_ids=self.token,past_key_values=cache,use_cache=True,logits_to_keep=1)
        self._copy_prefill(pre.past_key_values)
        torch.cuda.synchronize(self.device)
        pool=torch.cuda.graph_pool_handle()
        for i in range(decode_steps):
            cache=self._cache(i)
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph,pool=pool):
                output=model(input_ids=self.token,past_key_values=cache,use_cache=True,logits_to_keep=1)
            graph.replay()
            self.graphs.append(graph);self.logits.append(output.logits)
        torch.cuda.synchronize(self.device)
        self.prepare_ms=(time.perf_counter_ns()-start)/1e6
        self.kv_buffer_bytes=sum(t.numel()*t.element_size() for group in self.buffers for t in group)

    def _view(self,index,slot,length):
        b,h,_,d=self.shapes[index]
        return self.buffers[index][slot][:b*h*length*d].view(b,h,length,d)

    def _cache(self,step):
        n=self.prompt_length+step
        source=(step%2)*2;target=((step+1)%2)*2
        return Cache(layers=[_ExactLayer(self._view(i,source,n),self._view(i,source+1,n),
            self._view(i,target,n+1),self._view(i,target+1,n+1)) for i in range(len(self.shapes))])

    def _copy_prefill(self,cache):
        if len(cache.layers)!=len(self.shapes):raise ValueError('cache layer count changed')
        for i,layer in enumerate(cache.layers):
            if tuple(layer.keys.shape)!=self.shapes[i]:raise ValueError('cache geometry changed')
            self._view(i,0,self.prompt_length).copy_(layer.keys)
            self._view(i,1,self.prompt_length).copy_(layer.values)

    def _guard(self):
        if torch.is_grad_enabled():raise RuntimeError('decode graphs are inference-only')
        if not torch.is_inference_mode_enabled():
            raise RuntimeError('decode graph buffers require torch.inference_mode()')
        if self.model.training:raise RuntimeError('prepared model must remain in eval mode')
        if torch.cuda.current_stream(self.device)!=self.stream:
            raise RuntimeError('decode graph must run on its prepared stream')

    def begin(self,prompt):
        self._guard()
        if self._next is not None and self._next<self.decode_steps:
            raise RuntimeError('finish or reset the active request before begin')
        if prompt.shape!=(1,self.prompt_length) or prompt.dtype!=torch.long or prompt.device!=self.device:
            raise ValueError('prompt does not match prepared bucket')
        output=self.model(input_ids=prompt,use_cache=True,logits_to_keep=1)
        self._copy_prefill(output.past_key_values)
        self._next=0;self.prefills+=1
        return output.logits

    def step(self,token):
        self._guard()
        if self._next is None or self._next>=self.decode_steps:
            raise RuntimeError('begin a request before decode; horizon must not be exceeded')
        if token.shape!=(1,1) or token.dtype!=torch.long or token.device!=self.device:
            raise ValueError('one CUDA int64 token on the prepared device required')
        self.token.copy_(token)
        i=self._next;self.graphs[i].replay();self._next+=1;self.replays+=1
        return self.logits[i]

    def reset(self):
        self._guard()
        self._next=None
