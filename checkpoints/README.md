# Evaluation checkpoints

The [paper release](https://github.com/RuiziWang0923/operator-aware-restoration/releases/tag/paper) supplies the byte-identical EMA weights used for the recorded evaluations. Download and verify all three files with:

```bash
python scripts/download_checkpoints.py
```

| System | File | Selected epoch | SHA-256 |
|---|---|---:|---|
| PAST | [base.pt](https://github.com/RuiziWang0923/operator-aware-restoration/releases/download/paper/base.pt) | 159 | `7e4c5e2eef01556808f8318879cbfc4b01816d7ccdbacfacdf0b034495905b65` |
| JOME | [jome.pt](https://github.com/RuiziWang0923/operator-aware-restoration/releases/download/paper/jome.pt) | 80 | `f9f4cea2edddbbec4643eafd0399f4f6c24a446b9af83b2286c21b28efed9e0c` |
| ARISE | [arise.pt](https://github.com/RuiziWang0923/operator-aware-restoration/releases/download/paper/arise.pt) | 80 | `53a2e93f2d012b1146c8737acfe2e84bd42214654eadd02282c52dd7ddf5851a` |

JOME and ARISE both require the shared PAST backbone in `base.pt`. Put all files in this directory. Configurations, training seeds, selected epochs, parameter counts and numerical-validation scope are recorded in [MODEL_MANIFEST.json](../model/MODEL_MANIFEST.json).

Original authored weights are licensed under [CC BY-NC-SA 4.0](../LICENSE), with third-party terms described in [NOTICE](../NOTICE).
