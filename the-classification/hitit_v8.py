import argparse
import csv
import functools
import html
import http.server
import io
import random
import sys
import urllib.parse

import torch
import torch.backends.mps
import torch.jit
import torch.nn
import torch.onnx
import torch.utils.data
import tqdm

from typing import Any, Callable, IO, Sequence

MAX_CODEPOINT_LEN = 64
NUMBER_OF_BITS = 21  # Unicode max is 0x10FFFF
POSSIBLE_GENDER = ["M", "F", "U"]
POSSIBLE_GENDER_LONG = ["Male", "Female", "Unisex"]
CPU_DEVICE = torch.device("cpu")


class GolModel(torch.nn.Module):
    NUM_LAYERS = 1
    BIDI = False

    def __init__(self, nlen: int, nbits: int, *args, **kwargs) -> None:
        if nlen < 64 or (nlen & (nlen - 1) != 0):
            raise ValueError("Length needs to be at least 64 and PO2")

        super().__init__(*args, **kwargs)

        self.upper_layer = nlen * 2
        self.lstm = torch.nn.LSTM(
            input_size=nbits,
            hidden_size=self.upper_layer,
            num_layers=GolModel.NUM_LAYERS,
            bidirectional=GolModel.BIDI,
            batch_first=True,
        )
        self.fc1 = torch.nn.Linear(self.upper_layer * nlen, self.upper_layer)
        self.result = torch.nn.Linear(self.upper_layer, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dummy = torch.zeros(
            ((GolModel.BIDI + 1) * GolModel.NUM_LAYERS, x.size(0), self.upper_layer), dtype=x.dtype, device=x.device
        )
        lstm, _ = self.lstm(x, (dummy, dummy))
        lstm_r = lstm.reshape(lstm.size(0), -1)
        fc1 = self.fc1(lstm_r)
        result = torch.nn.functional.softmax(self.result(fc1), 1)
        return result


class TheNameDataset(torch.utils.data.Dataset[tuple[torch.Tensor, torch.Tensor, str]]):
    def __init__(self, csvpath: str, device: torch.device = CPU_DEVICE):
        super().__init__()

        self.result: list[tuple[torch.Tensor, torch.Tensor, str]] = []

        with open(csvpath, "r", encoding="UTF-8", newline="") as f:
            reader = csv.reader(f)
            next(reader, None)
            rows = list(reader)
            print("Loading dataset")

            for row in tqdm.tqdm(rows, unit="names"):
                input_data = convert_name_to_tensor(row[0])
                output_data = torch.zeros((2,), dtype=torch.float)

                if row[1] == "U":
                    output_data[0] = output_data[1] = 1.0
                else:
                    output_data[POSSIBLE_GENDER.index(row[1])] = 1.0

                self.result.append((input_data.to(device), output_data.to(device), row[1]))

    def __len__(self):
        return len(self.result)

    def __getitem__(self, index: int):
        return self.result[index]


def load_model(device: torch.device, state_path: str | None = None):
    model = GolModel(MAX_CODEPOINT_LEN, NUMBER_OF_BITS)
    model = model.to(device)

    if state_path is not None:
        model.load_state_dict(torch.load(state_path, device))

    return model


def name_to_boolbits(name: str):
    result: list[list[bool]] = []
    for s in name:
        bits = [False] * NUMBER_OF_BITS
        intval = ord(s)
        for i in range(min(NUMBER_OF_BITS, intval.bit_length())):
            bits[i] = bool(intval & (1 << i))
        result.append(bits)
    return result


def convert_name_to_tensor(name: str, tensor: torch.Tensor | None = None):
    if tensor is None:
        tensor = torch.zeros((MAX_CODEPOINT_LEN, NUMBER_OF_BITS), dtype=torch.float32)

    nametensor = torch.tensor(name_to_boolbits(name[:MAX_CODEPOINT_LEN]), dtype=torch.float32)
    tensor[: nametensor.size(0)] = nametensor

    return tensor


def do_single_epoch(
    model: GolModel,
    device: torch.device,
    train: torch.utils.data.DataLoader[tuple[torch.Tensor, torch.Tensor, str]],
    test: torch.utils.data.DataLoader[tuple[torch.Tensor, torch.Tensor, str]],
    total_train_len: int,
    total_test_len: int,
    lr: float,
):
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    model.train()

    with tqdm.tqdm(total=total_train_len, unit="train") as pbar:
        for batch_in, batch_out, _ in train:
            optimizer.zero_grad()
            out = model(batch_in.to(device))
            loss = criterion(out, batch_out.to(device))
            loss.backward()
            optimizer.step()
            pbar.update(len(batch_in))

    model.eval()

    with torch.no_grad():
        result_loss = 0.0
        accurate = 0

        with tqdm.tqdm(total=total_test_len, unit="infer") as pbar:
            for test_in, test_out, test_gender in test:
                out = model(test_in.to(device))
                loss = criterion(out, test_out.to(device))
                result_loss = result_loss + loss.item() * len(test_out)

                for a, b in zip(out.to(CPU_DEVICE), test_gender):
                    accurate = accurate + (get_guessed_gender(a) == b)

                pbar.update(len(test_in))

            return result_loss, accurate / total_test_len


def get_guessed_gender(result: Sequence[float] | torch.Tensor, mapper: Sequence[str] = POSSIBLE_GENDER):
    if isinstance(result, torch.Tensor):
        r = result.tolist()
    else:
        r = list(result)

    if abs(r[0] - r[1]) < 0.5:
        return mapper[2]
    else:
        return mapper[r.index(max(r))]


def print_result(name: str, result: torch.Tensor, file: IO[str] = sys.stdout):
    best_guess = get_guessed_gender(result, POSSIBLE_GENDER_LONG)
    print(name, file=file)
    for i, gender_name in enumerate(POSSIBLE_GENDER_LONG[:2]):
        percent = result[i].item() * 100.0
        print(gender_name + ":", percent, "%", file=file)
    print("Guessed:", best_guess, file=file)
    print(file=file)


def do_predict(model: GolModel, device: torch.device, name: str | list[str] | torch.Tensor) -> torch.Tensor:
    if isinstance(name, str):
        input_data = convert_name_to_tensor(name).reshape(1, MAX_CODEPOINT_LEN, NUMBER_OF_BITS).to(device)
    elif isinstance(name, list):
        input_data = torch.zeros((len(name), MAX_CODEPOINT_LEN, NUMBER_OF_BITS))
        for i, n in enumerate(name):
            convert_name_to_tensor(n, input_data[i])
        input_data = input_data.to(device)
    else:
        if len(name.size()) == 1:
            input_data = name.reshape(1, MAX_CODEPOINT_LEN).to(device)
        else:
            input_data = name.to(device)

    result = model(input_data)
    return result


class BaseArgs:
    model: str
    device: torch.device


class TrainArgs(BaseArgs):
    rate: float
    batches: int
    epoches: int
    output: str
    dataset: str
    seed: int
    preload: bool


class InferArgs(BaseArgs):
    name: list[str]


class ServeArgs(BaseArgs):
    host: str
    port: int


class TestArgs(BaseArgs):
    dataset: str
    batches: int


class ExportArgs(BaseArgs):
    new: bool
    opset: int
    output: str


def main_infer(args: InferArgs):
    model = load_model(args.device, args.model)
    model.eval()

    result = do_predict(model, args.device, args.name)

    for i, n in enumerate(args.name):
        print_result(n, result[i])


def main_train(args: TrainArgs):
    if args.seed:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    dataset = TheNameDataset(args.dataset, args.device if args.preload else CPU_DEVICE)
    train_set, test_set = torch.utils.data.random_split(dataset, [0.8, 0.2])
    train_dloader = torch.utils.data.DataLoader(train_set, args.batches)
    test_dloader = torch.utils.data.DataLoader(test_set, args.batches)

    existing_model_path = args.model or None
    model = load_model(args.device, existing_model_path)

    try:
        for i in range(args.epoches):
            loss, accuracy = do_single_epoch(
                model, args.device, train_dloader, test_dloader, len(train_set), len(test_set), args.rate
            )
            print("Epoch", i + 1, "Loss", loss, "Accuracy", accuracy * 100, "%")
    except KeyboardInterrupt:
        torch.save(model.state_dict(), args.output)
        raise

    torch.save(model.state_dict(), args.output)


HTML_RESPONSE = """<!DOCTYPE html>
<html>

<head>
    <meta charset="utf-8">
    <script>
        function main() {
            /** @type {HTMLInputElement} */
            const inputName = document.getElementById("input_name")
            /** @type {HTMLAnchorElement} */
            const linkClicker = document.getElementById("link_clicker")
            const hitit = document.getElementById("hitit")

            hitit.addEventListener("click", () => {
                linkClicker.href = inputName.value
                linkClicker.click()
            })
            inputName.value = decodeURI(window.location.pathname.substring(1))
        }

        if (document.readyState !== "loading") {
            main()
        } else {
            document.addEventListener("DOMContentLoaded", main)
        }
    </script>
</head>

<body>
    <div><!-- Response Here --></div>
    <label for="input_name">Name</label>
    <input id="input_name" type="text">
    <a id="link_clicker" href="" hidden></a>
    <button id="hitit">Infer</button>
</body>

</html>
"""


class HTTPServerHandler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def __init__(self, model: GolModel, device: torch.device, request, client_address, server):
        self.model = model
        self.device = device
        super().__init__(request, client_address, server)

    def do_GET(self):
        name = urllib.parse.unquote(self.path[1:])

        if name == "favicon.ico":
            self.send_response(404)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", "0")
            self.end_headers()
            self.wfile.flush()
            return

        with io.StringIO() as response:
            if name:
                print("Inferencing", name)
                result = do_predict(self.model, self.device, name)

                with io.StringIO() as buffer:
                    print_result(name, result[0], buffer)
                    response_text = buffer.getvalue()
            else:
                response_text = "Please specify name to infer in path"

            response.write(
                HTML_RESPONSE.replace(
                    "<!-- Response Here -->", "<br>".join(map(html.escape, response_text.split("\n")))
                )
            )
            response_bytes = response.getvalue().encode("UTF-8")

            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(response_bytes)))
            self.end_headers()
            self.wfile.write(response_bytes)
            self.wfile.flush()


def main_serve(args: ServeArgs):
    model = load_model(args.device, args.model)
    httpd = http.server.HTTPServer((args.host, args.port), functools.partial(HTTPServerHandler, model, args.device))
    model.eval()

    try:
        print("Listening on", args.host, "port", args.port)
        httpd.serve_forever(0.01)
    except KeyboardInterrupt:
        pass


def main_test(args: TestArgs):
    model = load_model(args.device, args.model)
    model.eval()

    total = 0
    accurate = 0

    print("Loading...")
    dataset = TheNameDataset(args.dataset)
    loader = torch.utils.data.DataLoader(dataset, args.batches)

    print("Testing...")
    with tqdm.tqdm(total=len(dataset), unit="infer") as pbar:
        with torch.no_grad():
            for tensor_name, tensor_gender, row_gender in loader:
                result = do_predict(model, args.device, tensor_name)

                for a, b in zip(row_gender, result.to(CPU_DEVICE)):
                    accurate = accurate + (a == get_guessed_gender(b))
                    total = total + 1

                pbar.update(len(tensor_gender))

    print("Tested", total, "names, guessed", accurate, "correctly.")
    print("Accuracy is", accurate * 100 / total, "%")


def main_export(args: ExportArgs):
    model = load_model(CPU_DEVICE, args.model)
    model.eval()

    example_input = torch.zeros((4, MAX_CODEPOINT_LEN, NUMBER_OF_BITS), dtype=torch.float32)
    convert_name_to_tensor("Foo", example_input[0])
    convert_name_to_tensor("Bar", example_input[1])
    convert_name_to_tensor("Hello", example_input[2])
    convert_name_to_tensor("World", example_input[3])

    if args.new:
        if torch.__version__ < "2.1.0" or sys.platform == "win32":
            raise RuntimeError("Dynamo export requires PyTorch 2.1.0 and not Windows")

        exported_model = torch.onnx.dynamo_export(model, example_input)
        exported_model.save(args.output)
    else:
        jit_model = torch.jit.trace(model, example_input)
        print(jit_model.code)

        torch.onnx.export(
            jit_model,
            example_input,
            args.output,
            opset_version=args.opset,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        )


MAIN_COMMAND: dict[str, Callable[[Any], None]] = {
    "infer": main_infer,
    "train": main_train,
    "serve": main_serve,
    "test": main_test,
    "export": main_export,
}

DEVICE_SELECTION: dict[str, Callable[[], bool]] = {
    "cuda": torch.cuda.is_available,
    "mps": torch.backends.mps.is_available,
    "cpu": lambda: True,
}


def select_best_device():
    for k, v in DEVICE_SELECTION.items():
        if v():
            return torch.device(k)
    raise RuntimeError("No backends available")


def main():
    def_torch_device = select_best_device()

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", help="Torch device to use", type=torch.device, default=def_torch_device)

    subparser = parser.add_subparsers(required=True, dest="cmd")
    train_parser = subparser.add_parser("train", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    train_parser.add_argument("--model", default="", help="Existing model to finetune")
    train_parser.add_argument("--rate", type=float, default=0.0005, help="Learning rate")
    train_parser.add_argument("--batches", type=int, default=16, help="Batch size")
    train_parser.add_argument("--epoches", type=int, default=100, help="Epoches")
    train_parser.add_argument("--output", default="modelgol.pt", help="Model output")
    train_parser.add_argument("--seed", type=int, default=0, help="Seed to use for training (0 = random)")
    train_parser.add_argument("--preload", action="store_true", help="Put all dataset to GPU if applicable?")
    train_parser.add_argument("dataset", help="Where to load the dataset")

    infer_parser = subparser.add_parser("infer", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    infer_parser.add_argument("--model", default="modelgol.pt", help="Model to use")
    infer_parser.add_argument("name", help="Name to infer", nargs="+")

    serve_parser = subparser.add_parser("serve", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    serve_parser.add_argument("--host", default="localhost", help="IP to listen")
    serve_parser.add_argument("--port", type=int, default=42060, help="Port to listen")
    serve_parser.add_argument("--model", default="modelgol.pt", help="Model to use")

    tests_parser = subparser.add_parser("test", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    tests_parser.add_argument("--model", default="modelgol.pt", help="Model to use")
    tests_parser.add_argument("dataset", help="Where to load the dataset")
    tests_parser.add_argument("--batches", type=int, default=16, help="Batch size")

    expor_parser = subparser.add_parser("export", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    expor_parser.add_argument("model", help="Model to use")
    expor_parser.add_argument("output", help="Model output")
    expor_parser.add_argument("--new", help="Use TorchDynamo for export", action="store_true")
    expor_parser.add_argument("--opset", help="ONNX Opset version.", type=int, default=10)

    args = parser.parse_args()
    MAIN_COMMAND[args.cmd](args)


if __name__ == "__main__":
    main()
