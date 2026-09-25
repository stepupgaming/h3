# SPDX-License-Identifier: GPL-3.0-only
"""009jev: native MiniMax H3 SLA with Jev keep-rate decisions from step one."""
import json, math, logging, subprocess, time, sys, os
from pathlib import Path
import torch
import comfy.patcher_extension as pe
from comfy_extras.nodes_sparse_attention import SparseAttnPatch, install_override, h3_sparse_attention, h3_eligible
try:
    from .native_sla_policy import (
        CONST_POLICIES,
        TABLE_POLICY,
        VALID_POLICIES,
        apply_confidence_floor,
        initial_keeps,
        parse_keep_table_from_context,
        step_keeps,
        uses_jev_worker,
    )
except ImportError:
    from native_sla_policy import (
        CONST_POLICIES,
        TABLE_POLICY,
        VALID_POLICIES,
        apply_confidence_floor,
        initial_keeps,
        parse_keep_table_from_context,
        step_keeps,
        uses_jev_worker,
    )

def emit(event):
    logging.info('[009jev] ' + json.dumps(event, allow_nan=False))
    raw = os.environ.get('GEMMY_JEV_EVENTS_PATH', '').strip()
    if raw:
        path = Path(raw)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('a', encoding='utf-8') as f:
            f.write(json.dumps(event, allow_nan=False) + '\n')

def _dump(name, obj):
    raw = os.environ.get('GEMMY_JEV_DUMP_DIR', '').strip()
    if not raw:
        return
    path = Path(raw) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, allow_nan=False, ensure_ascii=False), encoding='utf-8')

def _jev_request_meta(state, answer, seconds, n_questions, step):
    usage = (answer or {}).get('usage') or {}
    return {
        'input_tokens': usage.get('input_tokens'),
        'output_tokens': usage.get('output_tokens'),
        'raw_state_chars': len(json.dumps(state, ensure_ascii=False)),
        'n_questions': n_questions,
        'step': step,
        'request_duration': seconds,
    }

def sample(x, layout):
    out = {}
    for a, b, kind in layout.segments:
        if kind in ('audio', 'video') and b > a:
            ids = torch.linspace(a, b - 1, min(64, b - a), device=x.device).long()
            out[kind] = x.index_select(0, ids)[:, ::max(1, x.shape[-1] // 16)][:, :16].detach().float()
    return out

class Controller:

    def __init__(self, patch, sdk, initial_policy='fixed5', initial_context='', n_steps=4):
        self.patch = patch
        self.sdk = sdk
        self.step = 0
        self.n_steps = int(n_steps)
        if self.n_steps < 1:
            raise ValueError(f'n_steps must be >= 1 (got {n_steps})')
        self.pending = {}
        self.previous = {}
        self.requests = 0
        self.disabled = not uses_jev_worker(initial_policy)
        self.actual = {}
        self.initial_policy = initial_policy
        self.initial_context = initial_context
        context = {}
        try:
            context = json.loads(initial_context or '{}')
        except Exception:
            context = {}
        self.keep_table = parse_keep_table_from_context(context) if initial_policy == TABLE_POLICY else None
        if initial_policy == TABLE_POLICY:
            if self.keep_table is None:
                raise ValueError('table policy requires initial_context.keep_table (N x 50)')
            if len(self.keep_table) != self.n_steps:
                raise ValueError(
                    f'keep_table has {len(self.keep_table)} steps but sampler has {self.n_steps}'
                )
            self.keeps = step_keeps(self.keep_table, 1)
        else:
            self.keeps = initial_keeps(initial_policy)

    def initialize(self):
        if self.initial_policy == TABLE_POLICY:
            emit({
                'event': 'initial_decision',
                'reason': 'table',
                'initial_policy': self.initial_policy,
                'applied_layer_keep_percent': self.keeps[:],
                'decision_seconds': 0.0,
                'request_count': self.requests,
            })
            return
        if self.initial_policy in CONST_POLICIES:
            emit({
                'event': 'initial_decision',
                'reason': 'const',
                'initial_policy': self.initial_policy,
                'applied_layer_keep_percent': self.keeps[:],
                'decision_seconds': 0.0,
                'request_count': self.requests,
            })
            return
        if self.initial_policy != 'jev_first':
            return
        state = json.loads(self.initial_context)
        state.update(initialization=True, next_step=1, choices_percent=[1, 3, 5, 10], constraints='Choose all50 layers; block0 only5 or10. Historical pilot statistics are not current-run measurements; first pass forms scene and typography. No quota or forced variation.')
        start = time.perf_counter()
        self.requests += 1
        event = {'event': 'initial_decision', 'state': state}
        try:
            _dump('jev_request_initial.json', state)
            r = subprocess.run([self.sdk, '-B', '-X', 'utf8', str(Path(__file__).with_name('native_sla_worker.py'))], input=json.dumps(state), capture_output=True, text=True, encoding='utf-8', timeout=25, creationflags=subprocess.CREATE_NO_WINDOW)
            assert r.returncode == 0
            answer = json.loads(r.stdout)
            _dump('jev_response_initial.json', answer)
            assert set(answer['decisions']) == {str(b) for b in range(50)}
            new = []
            for b in range(50):
                d = answer['decisions'][str(b)]
                k = float(d['choice'])
                c = float(d['confidence'])
                assert k in ((5, 10) if b == 0 else (1, 3, 5, 10)) and math.isfinite(c) and (0 <= c <= 1)
                applied, _, _ = apply_confidence_floor(k, c, first_step=True)
                new.append(applied)
            self.keeps = new
            event.update(answer=answer, reason='jev')
            event['jev_request'] = _jev_request_meta(state, answer, time.perf_counter() - start, 50, 1)
        except Exception as e:
            self.keeps = [10.0] * 50
            self.disabled = True
            event['reason'] = 'fallback:' + type(e).__name__
        event.update(applied_layer_keep_percent=self.keeps[:], decision_seconds=time.perf_counter() - start, request_count=self.requests)
        emit(event)

    def observe(self, b, before, after):
        vals = []
        for kind in ('audio', 'video'):
            a = before[kind]
            d = after[kind] - a
            prev = self.previous.get((b, kind))
            vals.extend([d.norm() / a.norm().clamp_min(1e-08), d.new_tensor(-1.0) if prev is None else (d - prev).norm() / prev.norm().clamp_min(1e-08)])
            self.previous[b, kind] = d
        self.pending[b] = torch.stack(vals)

    def done(self, step):
        assert step == self.step
        event = {'event': 'step', 'policy': 'native_sla', 'step': step + 1, 'n_steps': self.n_steps, 'applied_keep_percent': sum(self.keeps) / 50, 'applied_layer_keep_percent': self.keeps[:], 'actual_attention': dict(self.actual)}
        last = self.n_steps - 1
        if step < last and (not self.disabled):
            start = time.perf_counter()
            rows = torch.stack([self.pending[b] for b in range(50)]).cpu().tolist()
            assert all((math.isfinite(v) for r in rows for v in r))
            ranks = {k: sorted(range(50), key=lambda b: rows[b][i]) for k, i in [('audio', 0), ('video', 2)]}
            state = {'step_measured': step + 1, 'next_step': step + 2, 'choices_percent': [1, 3, 5, 10], 'protected': f'First step policy={self.initial_policy}; block0 on later steps keep5. All blocks execute. Native SLA: conditioning KV and audio query rows exact; no activation reuse. Same 009 normal weights and FFN.', 'objective': 'Allocate attention selectively to preserve speech/song, identity, motion and text while reducing compute. No forced variation or quota.', 'limitations': 'Sampled residual magnitude/rank and temporal change are uncalibrated proxies, not quality scores or measured sparse error. Audio exact query rows do not guarantee identical downstream audio.', 'blocks': {str(b): {kind: {'residual_relative_l2': round(rows[b][i], 6), 'rank': round((ranks[kind].index(b) + 1) / 50, 3), 'cross_step_change': None if rows[b][i + 1] < 0 else round(rows[b][i + 1], 6)} for kind, i in [('audio', 0), ('video', 2)]} for b in range(50)}, 'current_keep': self.keeps}
            self.requests += 1
            try:
                _dump(f'jev_request_step{step + 2}.json', state)
                r = subprocess.run([self.sdk, '-B', '-X', 'utf8', str(Path(__file__).with_name('native_sla_worker.py'))], input=json.dumps(state), capture_output=True, text=True, encoding='utf-8', timeout=25, creationflags=subprocess.CREATE_NO_WINDOW)
                assert r.returncode == 0, 'worker failed'
                answer = json.loads(r.stdout)
                _dump(f'jev_response_step{step + 2}.json', answer)
                assert set(answer['decisions']) == {str(b) for b in range(1, 50)}
                new = [5.0] * 50
                for b in range(1, 50):
                    d = answer['decisions'][str(b)]
                    k = float(d['choice'])
                    c = float(d['confidence'])
                    assert k in (1, 3, 5, 10) and math.isfinite(c) and (0 <= c <= 1)
                    applied, _, _ = apply_confidence_floor(k, c, first_step=False)
                    new[b] = applied
                self.keeps = new
                event.update(answer=answer, reason='jev')
                event['jev_request'] = _jev_request_meta(state, answer, time.perf_counter() - start, 49, step + 2)
            except Exception as e:
                self.disabled = True
                self.keeps = [5.0] * 50
                event['reason'] = 'fallback:' + type(e).__name__
            event.update(state=state, decision_seconds=time.perf_counter() - start, next_layer_keep_percent=self.keeps[:], request_count=self.requests)
        else:
            if self.keep_table is not None:
                event['reason'] = 'table'
                if step < last:
                    self.keeps = step_keeps(self.keep_table, step + 2)
                    event['next_layer_keep_percent'] = self.keeps[:]
            elif step == last:
                event['reason'] = 'last_step'
            elif self.initial_policy in CONST_POLICIES:
                event['reason'] = 'const'
            else:
                event['reason'] = 'circuit_open'
        emit(event)
        self.pending = {}
        self.actual = {}
        self.step += 1

class H3JevNativeSLAPatch:

    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'model': ('MODEL',), 'sdk_python': ('STRING', {'default': ''})}, 'optional': {'initial_policy': (list(VALID_POLICIES),), 'initial_context': ('STRING', {'default': '{}', 'multiline': True}), 'prompt_context': ('STRING', {'default': '', 'multiline': True})}}
    RETURN_TYPES = ('MODEL',)
    FUNCTION = 'patch'
    CATEGORY = 'Jev/Local experiment'

    def patch(self, model, sdk_python='', initial_policy='jev_first', initial_context='{}', prompt_context=''):
        if initial_policy not in VALID_POLICIES:
            raise ValueError('Unknown initial policy')
        needs_jev = uses_jev_worker(initial_policy)
        if needs_jev and not os.environ.get('TYPESAFE_API_KEY', '').strip():
            raise RuntimeError('Set TYPESAFE_API_KEY in the ComfyUI process environment')
        sdk_python = sdk_python.strip()
        if needs_jev:
            sdk_python = sdk_python or sys.executable
            if not Path(sdk_python).is_file():
                raise ValueError('sdk_python must point to an existing Python executable')
        else:
            sdk_python = sdk_python or ''
        context = json.loads(initial_context or '{}')
        if not isinstance(context, dict):
            raise ValueError('initial_context must be a JSON object')
        if prompt_context.strip():
            context['prompt'] = prompt_context
        if initial_policy == 'jev_first' and (not str(context.get('prompt', '')).strip()):
            raise ValueError('009jev requires prompt_context or initial_context.prompt for its first decision')
        if initial_policy == TABLE_POLICY:
            parse_keep_table_from_context(context)
            if context.get('keep_table') is None:
                raise ValueError('table policy requires initial_context.keep_table')
        json.dumps(context, allow_nan=False)
        initial_context = json.dumps(context)
        sampling = model.get_model_object('model_sampling')
        p = SparseAttnPatch(tau=1.3, topk_ratio=0.05, vsa=False, sigma_start=float(sampling.percent_to_sigma(0)), sigma_end=float(sampling.percent_to_sigma(1)), min_tokens=12288, dense_blocks=set(), sink_conditioning='exact_kv_and_rows', extra_tokens=0, verbose=False)
        m = model.clone()
        install_override(p, m.model_options['transformer_options'])
        m.add_callback_with_key(pe.CallbacksMP.ON_PREPARE_STATE, 'jev_native_sla', lambda mp, t, opts: install_override(p, opts['transformer_options']))
        m.add_callback_with_key(pe.CallbacksMP.ON_CLEANUP, 'jev_native_sla', lambda mp: p.reset())

        def make(block, b):

            def attention(h, rope_freqs=None, transformer_options=None):
                return h3_sparse_attention(block.attn, h, rope_freqs, transformer_options, p, b)

            def call(args, extra):
                c = p.controller
                p.topk_ratio = c.keeps[b] / 100
                eligible = h3_eligible(block.attn, args['img'], args['rope_freqs'], args['transformer_options'], p, b)
                c.actual[str(b)] = 'sla' if eligible else 'dense_fallback'
                before = sample(args['img'], args['layout']) if c.step < c.n_steps - 1 else None
                out = extra['original_block']({**args, 'attention': attention} if eligible else args)
                if before is not None:
                    c.observe(b, before, sample(out['img'], args['layout']))
                return out
            return call
        blocks = model.get_model_object('diffusion_model').blocks
        assert len(blocks) == 50
        for b, block in enumerate(blocks):
            m.set_model_patch_replace(make(block, b), 'dit', 'double_block', b)

        def wrapper(executor, model_wrap, sigmas, extra_args, callback, noise, latent_image=None, denoise_mask=None, disable_pbar=False):
            n_steps = int(len(sigmas) - 1)
            if n_steps < 1:
                raise AssertionError(f'native SLA needs at least 1 sampler step (got len(sigmas)={len(sigmas)})')
            fn = getattr(executor.class_obj, 'sampler_function', None)
            sampler_name = getattr(fn, '__name__', '')
            if n_steps == 4:
                if sampler_name != 'sample_res_multistep':
                    raise AssertionError(
                        f'4-step native SLA requires sample_res_multistep (got {sampler_name!r})'
                    )
            elif sampler_name != 'sample_euler':
                raise AssertionError(
                    f'{n_steps}-step native SLA requires sample_euler HQ (got {sampler_name!r}); '
                    'do not run 20-step quality on res_multistep'
                )
            c = Controller(p, sdk_python, initial_policy, initial_context, n_steps=n_steps)
            p.controller = c
            c.initialize()
            emit({
                'event': 'begin',
                'policy': 'native_sla',
                'initial_policy': initial_policy,
                'first_step_keep': c.keeps[:],
                'later_block0_keep': 5,
                'n_steps': n_steps,
                'sampler': sampler_name,
                'sigmas': sigmas.detach().cpu().tolist(),
            })

            def done(i, denoised, x, total):
                c.done(i)
                if callback is not None:
                    callback(i, denoised, x, total)
            try:
                return executor(model_wrap, sigmas, extra_args, done, noise, latent_image, denoise_mask, disable_pbar)
            finally:
                emit({'event': 'end', 'requests': c.requests})
        m.add_wrapper_with_key(pe.WrappersMP.SAMPLER_SAMPLE, 'jev_native_sla', wrapper)
        return (m,)
NODE_CLASS_MAPPINGS = {'H3JevNativeSLAPatch': H3JevNativeSLAPatch}
NODE_DISPLAY_NAME_MAPPINGS = {'H3JevNativeSLAPatch': '009jev: Jev-guided Native SLA'}
