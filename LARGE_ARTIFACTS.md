# Large artifact inventory

This private repository contains the complete source tree, experiment documentation, configuration, validation receipts, plots, aggregate metrics, comparison files, manifests, and training logs. Large binary weights and bulky per-sample generations remain local. Their content is pinned by cryptographic manifests so every reported aggregate can be traced to its source artifact without making an ordinary Git clone impractical.

| Local path | Bytes | SHA-256 | Status |
| --- | ---: | --- | --- |
| `results/arc-20261011-snapshot-2316/adapters/rescue-15000-capacity-route-dft/adapter_model.safetensors` | 479,005,064 | `41ed905ff6dee2dcee18daf7989f8851542e0a39cafc0e4956e4ab3fd3253474` | Local adapter weights; config, tokenizer, README, and provenance remain tracked |
| `results/arc-20261011-snapshot-2316/adapters/concise-2000-dft/adapter_model.safetensors` | 479,005,064 | `dacdbecf400141a908c35cd13ab867c4583285fc76fd2cadab7fe259097889ae` | Local adapter weights; config, tokenizer, README, and provenance remain tracked |
| `results/arc-20261011-snapshot-2316/adapters/concise-2000-sft/adapter_model.safetensors` | 479,005,064 | `f580aebed4c2352c8f1dd8b6440525521d50135882d38d215e1b627a06c361ac` | Local adapter weights; config, tokenizer, README, and provenance remain tracked |

The selected training dataset is stored in Git as `artifacts/datasets-final-8k/advanced.json.gz`. Its uncompressed form is 186,353,276 bytes with SHA-256 `dfd90ba66bf97bfe649befcb7edf009e5a539296790a22e0b12c19336d19d10a`.

Per-sample evaluation batches matching `results/**/batch_*.json` are listed with byte sizes and SHA-256 hashes in `results/RAW_PREDICTION_MANIFEST.tsv`. All corresponding `metrics.json`, `manifest.json`, comparison JSON, confidence intervals, figures, and reports are tracked.

Restore it with:

```bash
gzip -dk artifacts/datasets-final-8k/advanced.json.gz
sha256sum artifacts/datasets-final-8k/advanced.json
```

