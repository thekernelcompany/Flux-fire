# Fixing “Must pass exactly one of `num_inference_steps` or `timesteps`” when using FLUX.1-Kontext

## TL;DR
The error comes from **our custom monkey-patch** for `DPMSolverMultistepScheduler.set_timesteps`.
The patch advertises the extra keyword arguments (`sigmas`, `custom_sigmas`) expected by
`FluxKontextPipeline`, but then **forwards neither `num_inference_steps` nor `timesteps` to the
original implementation** when those keywords are used.  As a result the scheduler raises:

```
ValueError: Must pass exactly one of `num_inference_steps` or `timesteps`.
```

Add a tiny guard in the patched method to translate a received `sigmas` list into
`num_inference_steps=len(sigmas)` (or forward an explicit `timesteps` list).  After that the
pipeline compiles and runs correctly.

---

## What exactly happens
1. **`FluxKontextPipeline.__call__` → `retrieve_timesteps()`**
   * With the DPM-Solver scheduler the pipeline often prepares a *custom sigma schedule*, so it
     calls:
     ```python
     scheduler.set_timesteps(sigmas=sigmas, device=device)
     ```
2. **Our compatibility patch** (`flux_kontext_optimized.py`, lines 78-95)
   ```python
   def _patched_set_timesteps(self, *args, sigmas=None, custom_sigmas=None, **kwargs):
       # NOTE: currently ignores `sigmas` / `custom_sigmas`
       return _orig_set_timesteps(self, *args, **kwargs)
   ```
   The function **silently drops** `sigmas` and forwards neither `num_inference_steps` nor
   `timesteps` to `_orig_set_timesteps`.
3. **`DPMSolverMultistepScheduler.set_timesteps` (upstream code)** detects that it received neither
   `num_inference_steps` nor `timesteps` and throws the ValueError.

```133:140:flux_kontext_optimized.py
    def _patched_set_timesteps(self, *args, sigmas=None, custom_sigmas=None, **kwargs):
        # NOTE: We intentionally ignore `sigmas` / `custom_sigmas` – the original
        # method will compute timesteps based on `num_inference_steps` or
        # `timesteps`. This is good enough for FluxKontext since it mainly needs
        # the keyword to exist for its feature-gate.
        return _orig_set_timesteps(self, *args, **kwargs)
```

---

## Minimal fix (preferred)
Patch the shim so that it **derives** `num_inference_steps` or `timesteps` when only `sigmas` (or
`custom_sigmas`) is provided:

```python
# inside flux_kontext_optimized.py right after the signature check
if "sigmas" not in _sig.parameters and "custom_sigmas" not in _sig.parameters:

    def _patched_set_timesteps(self, *args, sigmas=None, custom_sigmas=None, **kwargs):
        # If caller supplied a sigma schedule but no explicit step count/timesteps,
        # translate it for the original implementation.
        if sigmas is not None and "num_inference_steps" not in kwargs and "timesteps" not in kwargs:
            kwargs["num_inference_steps"] = len(sigmas)
        if custom_sigmas is not None and "num_inference_steps" not in kwargs and "timesteps" not in kwargs:
            kwargs["num_inference_steps"] = len(custom_sigmas)
        return _orig_set_timesteps(self, *args, **kwargs)
```

*(Don’t forget to keep the `__signature__` assignment that exposes the new kwargs.)*

---

## Alternative work-arounds
1. **Pass an explicit `num_inference_steps` every time you call the pipeline**
   ```python
   self.pipe(..., num_inference_steps=14)
   ```
   This forces `retrieve_timesteps()` down the “standard path” that never uses the `sigmas` keyword.
2. **Upgrade to a newer `diffusers` version** where `DPMSolverMultistepScheduler` already accepts
   `sigmas` / `custom_sigmas` natively.  Then the monkey-patch can be removed entirely.

---

## Quick checklist when you see this error again
- [ ] Are we still running the local monkey-patch?  (Search for `_patched_set_timesteps`.)
- [ ] Does the patched function forward a valid `num_inference_steps` / `timesteps`?
- [ ] Are we on an old `diffusers` release (< 0.28) without native sigma support?
- [ ] Did we forget to pass `num_inference_steps` in a custom pipeline call?

If all boxes are ticked, apply the *minimal fix* above or simply upgrade `diffusers`.

---

# Fixing "TypeError: flash_attention_forward() got an unexpected keyword argument 'image_rotary_emb'"

## TL;DR
The FlashAttention patch function signature doesn't match what FLUX.1-Kontext expects. The original attention layer expects `image_rotary_emb` and other kwargs, but our patch only accepts `hidden_states`, `encoder_hidden_states`, and `attention_mask`.

## What exactly happens
1. **FLUX.1-Kontext attention layer** calls the patched `flash_attention_forward` with:
   ```python
   self.attn(hidden_states=..., encoder_hidden_states=..., image_rotary_emb=..., **kwargs)
   ```
2. **Our FlashAttention patch** only accepts:
   ```python
   def flash_attention_forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None):
   ```
3. **Python raises TypeError** because `image_rotary_emb` is an unexpected keyword argument.

## Minimal fix
Update the FlashAttention patch function signature to accept all the arguments that FLUX.1-Kontext passes:

```python
def flash_attention_forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, image_rotary_emb=None, **kwargs):
    # ... rest of implementation stays the same
```

## Alternative work-arounds
1. **Disable FlashAttention patch** by setting `enable_optimizations['flash_attention'] = False`
2. **Use a more sophisticated patch** that handles rotary embeddings properly
3. **Upgrade to a version** where FlashAttention is natively supported in FLUX.1-Kontext

## Quick checklist when you see this error
- [ ] Does the patched function accept all expected arguments?
- [ ] Are we passing `image_rotary_emb` and other kwargs through?
- [ ] Should we disable FlashAttention for this model version?
- [ ] Is there a newer version with native FlashAttention support? 