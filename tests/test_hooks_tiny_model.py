"""Hook-convention tests against a tiny RANDOMLY INITIALISED model.

No checkpoint is downloaded: a 3-block Llama-architecture model is built from a
config with random weights. It takes about a second and it verifies the two
things DESIGN.md §A.11 names as the silent killers of this pipeline:

  1. LAYER INDEXING (§A.11 item 2). ``residuals[i]`` must be the OUTPUT of block
     ``i``. HF's own ``output_hidden_states=True`` tuple has
     ``hidden_states[j]`` = INPUT of block ``j``, so the assertion is
     ``residuals[i] == hidden_states[i+1]``. If our hook were on the input side,
     this test fails by exactly one.

  2. PRE- vs POST-LAYERNORM (§A.11 item 1). ``residuals[n_layers-1]`` must be
     the un-normalized residual stream, i.e. it must DIFFER from
     ``hidden_states[n_layers]``, which HF produces after applying the final
     ``model.norm``. If they were equal we would be capturing normalized
     states, every token would sit on a common radius, and the geometry would be
     wrong in a way that still produces publishable-looking clusters.

Run on the pod immediately after pod_bootstrap.sh, before the G1 gate: it is
cheap and it catches a transformers-version change to the block signature.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

try:
    import numpy as np
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM
    HAVE_STACK = True
except Exception:  # noqa: BLE001
    HAVE_STACK = False

if HAVE_STACK:
    from ling.hooks import (ExtractionError, FINAL, ResidualCapture,
                            extract_aligned, forward_once)
    from ling.model import LoadedModel, resolve_layers_path


class TinyFastTokenizer:
    """Whitespace/digit tokenizer with a real offset mapping and small ids."""

    is_fast = True

    def __init__(self, vocab_size=64):
        self.vocab_size = vocab_size

    def __call__(self, text, return_offsets_mapping=False, add_special_tokens=True,
                 return_tensors=None, padding=False, truncation=False, **kw):
        spans, i, n = [], 0, len(text)
        while i < n:
            if text[i].isspace():
                i += 1
                continue
            j, kind = i, text[i].isdigit()
            while j < n and not text[j].isspace() and text[j].isdigit() == kind:
                j += 1
            start = i - 1 if (i > 0 and text[i - 1] == " ") else i
            spans.append((start, j))
            i = j
        ids = [3 + (hash(text[s:e]) % (self.vocab_size - 4)) for s, e in spans]
        offsets = list(spans)
        if add_special_tokens:
            ids.insert(0, 1)
            offsets.insert(0, (0, 0))
        out = {"input_ids": ids}
        if return_offsets_mapping:
            out["offset_mapping"] = offsets
        return out

    def decode(self, ids):
        return "<{}>".format(",".join(str(i) for i in ids))


def build_tiny(n_layers=3, d_model=32, vocab=64):
    cfg = LlamaConfig(
        vocab_size=vocab, hidden_size=d_model, intermediate_size=2 * d_model,
        num_hidden_layers=n_layers, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=128,
    )
    torch.manual_seed(0)
    model = LlamaForCausalLM(cfg)
    model.eval()
    model.requires_grad_(False)
    layers_path = resolve_layers_path(model, n_layers, None)
    return LoadedModel(
        model=model, tokenizer=TinyFastTokenizer(vocab), config={},
        model_id="tiny-random-llama", layers_path=layers_path,
        n_layers=n_layers, d_model=d_model, device="cpu", dtype=torch.float32,
        load_info={"attn_implementation": "eager"},
    )


TEXT = "Natalia sold 48 clips in April and 24 in May."


@unittest.skipUnless(HAVE_STACK, "torch/transformers not installed")
class TestHookConvention(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.loaded = build_tiny()

    def _ids(self, text=TEXT):
        enc = self.loaded.tokenizer(text)
        return torch.tensor([enc["input_ids"]], dtype=torch.long)

    def test_layers_path_resolves_to_the_decoder_stack(self):
        self.assertEqual(self.loaded.layers_path, "model.layers")
        self.assertEqual(len(self.loaded.layer_modules()), self.loaded.n_layers)

    def test_index_i_is_the_OUTPUT_of_block_i(self):
        """§A.11 item 2: off-by-one between input and output of layers[i]."""
        ids = self._ids()
        with ResidualCapture(self.loaded) as cap:
            forward_once(self.loaded, ids)
            cap.assert_complete()
            ours = [r.clone() for r in cap.residuals]

        with torch.inference_mode():
            ref = self.loaded.model(input_ids=ids, attention_mask=torch.ones_like(ids),
                                    use_cache=False, output_hidden_states=True)
        hs = ref.hidden_states
        # HF: hs[j] is the INPUT of block j for j < L; hs[L] is post-final-norm,
        # so it is compared separately in the layernorm test below.
        self.assertEqual(len(hs), self.loaded.n_layers + 1)
        for i in range(self.loaded.n_layers - 1):
            self.assertTrue(
                torch.equal(ours[i], hs[i + 1].cpu()),
                "residuals[{}] is not the output of block {} (== input of block "
                "{}). Off-by-one in the layer indexing convention.".format(i, i, i + 1))
        # explicit off-by-one guards, in both directions:
        #   our layer 0 is NOT the embedding output (that is hs[0]) ...
        self.assertFalse(torch.equal(ours[0], hs[0].cpu()))
        #   ... and our layer i is not shifted the other way either
        for i in range(1, self.loaded.n_layers - 1):
            self.assertFalse(torch.equal(ours[i], hs[i].cpu()))

    def test_last_residual_is_PRE_final_layernorm(self):
        """§A.11 item 1: which side of the layernorm we took."""
        ids = self._ids()
        with ResidualCapture(self.loaded) as cap:
            forward_once(self.loaded, ids)
            last = cap.residuals[-1].clone()
        with torch.inference_mode():
            ref = self.loaded.model(input_ids=ids, attention_mask=torch.ones_like(ids),
                                    use_cache=False, output_hidden_states=True)
        post_norm = ref.hidden_states[-1].cpu()
        self.assertEqual(last.shape, post_norm.shape)
        self.assertFalse(torch.equal(last, post_norm),
                         "captured the POST-norm state; RMSNorm would put every "
                         "token on a common radius and destroy the geometry")
        # and it must equal the norm's input
        with torch.inference_mode():
            self.assertTrue(torch.equal(
                self.loaded.model.model.norm(last), post_norm))

    def test_hook_refuses_batched_input(self):
        """§A.4 / §A.11 item 3: batch size 1, no padding, ever."""
        ids = self._ids()
        batched = torch.cat([ids, ids], dim=0)
        with self.assertRaises(ExtractionError) as cm:
            with ResidualCapture(self.loaded):
                forward_once(self.loaded, batched)
        self.assertIn("batch size 1", str(cm.exception))

    def test_forward_refuses_a_padded_attention_mask(self):
        ids = self._ids()
        mask = torch.ones_like(ids)
        mask[0, -1] = 0
        with self.assertRaises(ExtractionError) as cm:
            forward_once(self.loaded, ids, attention_mask=mask)
        self.assertIn("padded", str(cm.exception))

    def test_hooks_are_removed_on_exit(self):
        n_before = sum(len(l._forward_hooks) for l in self.loaded.layer_modules())
        with ResidualCapture(self.loaded):
            n_during = sum(len(l._forward_hooks) for l in self.loaded.layer_modules())
        n_after = sum(len(l._forward_hooks) for l in self.loaded.layer_modules())
        self.assertEqual(n_during, n_before + self.loaded.n_layers)
        self.assertEqual(n_after, n_before)

    def test_two_passes_are_bitwise_identical_on_cpu(self):
        """The CPU shadow of gate G1. Not a substitute — CUDA is the real test."""
        from ling.determinism import assert_bitwise_identical
        ids = self._ids()
        with ResidualCapture(self.loaded) as cap:
            forward_once(self.loaded, ids)
            a = [r.clone() for r in cap.residuals]
        with ResidualCapture(self.loaded) as cap:
            forward_once(self.loaded, ids)
            b = [r.clone() for r in cap.residuals]
        for i, (x, y) in enumerate(zip(a, b)):
            assert_bitwise_identical(x, y, name="layer {}".format(i))

    def test_assert_bitwise_identical_rejects_a_single_perturbed_element(self):
        """Guards against the check degenerating into an allclose."""
        from ling.determinism import DeterminismError, assert_bitwise_identical
        ids = self._ids()
        with ResidualCapture(self.loaded) as cap:
            forward_once(self.loaded, ids)
            a = cap.residuals[1].clone()
        b = a.clone()
        b[0, 0, 0] = b[0, 0, 0] + torch.finfo(b.dtype).eps
        with self.assertRaises(DeterminismError) as cm:
            assert_bitwise_identical(a, b, name="layer 1 residual")
        msg = str(cm.exception)
        self.assertIn("layer 1 residual", msg)
        self.assertIn("flat index 0", msg)


@unittest.skipUnless(HAVE_STACK, "torch/transformers not installed")
class TestExtractAligned(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.loaded = build_tiny()

    def anchors(self):
        return {
            "n_clips_april": [TEXT.index("48"), TEXT.index("48") + 2],
            "n_clips_may": [TEXT.index("24"), TEXT.index("24") + 2],
            FINAL: [-1, -1],
        }

    def test_shape_and_order(self):
        out, meta = extract_aligned(
            TEXT, self.anchors(), self.loaded.tokenizer, self.loaded,
            return_meta=True)
        self.assertEqual(out.shape,
                         (self.loaded.n_layers, 3, self.loaded.d_model))
        self.assertEqual(out.dtype, np.float32)
        self.assertEqual(meta.names, ["n_clips_april", "n_clips_may", FINAL])
        self.assertEqual(meta.token_indices[-1], meta.n_tokens - 1)

    def test_values_match_the_captured_residuals_at_those_positions(self):
        out, meta = extract_aligned(
            TEXT, self.anchors(), self.loaded.tokenizer, self.loaded,
            return_meta=True)
        ids = torch.tensor([meta.input_ids], dtype=torch.long)
        with ResidualCapture(self.loaded) as cap:
            forward_once(self.loaded, ids)
            for li in range(self.loaded.n_layers):
                for j, ti in enumerate(meta.token_indices):
                    np.testing.assert_array_equal(
                        out[li, j], cap.residuals[li][0, ti].float().numpy())

    def test_no_raw_activation_survives_the_call(self):
        """Sanity on §A.1: only the aligned slice comes back, not [L, T, d]."""
        out, meta = extract_aligned(
            TEXT, self.anchors(), self.loaded.tokenizer, self.loaded,
            return_meta=True)
        self.assertLess(out.shape[1], meta.n_tokens)


if __name__ == "__main__":
    unittest.main(verbosity=2)
