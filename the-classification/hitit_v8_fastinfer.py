import argparse
import sys

import numpy
import onnxruntime

from typing import IO, Sequence

MAX_CODEPOINT_LEN = 64
NUMBER_OF_BITS = 21  # Unicode max is 0x10FFFF
POSSIBLE_GENDER_LONG = ["Male", "Female", "Unisex"]


def get_guessed_gender(r: list[float], mapper: Sequence[str]):
    if abs(r[0] - r[1]) < 0.5:
        return mapper[2]
    else:
        return mapper[r.index(max(r))]


def print_result(name: str, result: list[float], file: IO[str] = sys.stdout):
    best_guess = get_guessed_gender(result, POSSIBLE_GENDER_LONG)
    print(name, file=file)
    for i, gender_name in enumerate(POSSIBLE_GENDER_LONG[:2]):
        percent = result[i] * 100.0
        print(gender_name + ":", percent, "%", file=file)
    print("Guessed:", best_guess, file=file)
    print(file=file)


def convert_name_to_numpy(name: str, tensor: numpy.ndarray | None = None):
    if tensor is None:
        tensor = numpy.zeros((MAX_CODEPOINT_LEN, NUMBER_OF_BITS), numpy.float32)

    for i in range(min(len(name), MAX_CODEPOINT_LEN)):
        intval = ord(name[i])
        for j in range(NUMBER_OF_BITS):
            tensor[i, j] = bool(intval & (1 << j))

    return tensor


def do_predict(model: onnxruntime.InferenceSession, name: list[str]) -> list[list[float]]:
    input_data = numpy.zeros((len(name), MAX_CODEPOINT_LEN, NUMBER_OF_BITS), numpy.float32)
    for i, n in enumerate(name):
        convert_name_to_numpy(n, input_data[i])
        for j in range(min(MAX_CODEPOINT_LEN, len(n))):
            print("input", input_data[i, j])

    result: numpy.ndarray = model.run(None, {"input": input_data})[0]
    print("output", result)
    return result.tolist()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model", default="modelgol.onnx", help="Existing model to load")
    parser.add_argument("name", help="Name to inference.", nargs="+")
    args = parser.parse_args()

    model = onnxruntime.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    result = do_predict(model, args.name)

    for i, n in enumerate(args.name):
        print_result(n, result[i])
