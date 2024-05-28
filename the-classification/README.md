Name Gender Classification
=====

Source code for my [name gender classification](https://mikuauahdark.github.io/namegender.html).

Tested on Python 3.11 (3.10 may work but no guarantees).

There are certain files:

* `hitit_v8.py` - Single file Python code to train and inference on the model. Needs NumPy, PyTorch, ONNX.
* `hitit_v8_fastinfer.py` - Single file Python code to perform inference using ONNX exported model. Needs NumPy, ONNX Runtime.
* `modelgol.mjs` - JavaScript ES6 module for browser to load ONNX exported model.

How to Train
-----

```
python hitit_v8.py train <input CSV>
```

Run `python hitit_v8.py --help` for more information and `python hitit_v8.py train --help` for training-specific
command-line options.

`<input CSV>` should contain name followed by the gender either M (male), F (female), or U (unisex). Example:
```
Name,Gender
Mary,F
Elizabeth,U
Ida,F
John,M
Brad,M
... and so on
```

How to Export Model to ONNX
-----

```
python hitit_v8.py export <model.pt file> <model.onnx output>
```

How to JavaScript Inference
-----

Note: Code may not run as-is. It's only to demonstrate how to use the JS API.

```js
import * as modelgol from "./modelgol.mjs"

const onnxmodel = await fetch("path/to/onnxmodel.onnx")
const session = await modelgol.create(await onnxmodel.arrayBuffer(), false)
const result = await session.infer(name1, name2, ...)
// result will be an array of object containing these fields:
// * gender (0 for male, 1 for female, 2 for unisex)
// * genderString (string "Male", "Female", or "Unisex")
// * maleness (in [0, 1] real number)
// * femaleness (in [0, 1] real number)
```

Infer With [NPNN](https://github.com/MikuAuahDark/NPad93#npnn)
-----

TODO
