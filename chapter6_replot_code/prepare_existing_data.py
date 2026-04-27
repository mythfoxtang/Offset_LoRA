import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
INPUT_DIR = ROOT / "inputs"
INPUT_DIR.mkdir(exist_ok=True)
DATA_ROOT = ROOT.parent.parent / "毕设数据大全"


def extract_lists(text: str):
    matches = re.findall(r"\[[^\]]*\]", text, flags=re.S)
    curves = []
    for match in matches:
        values = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", match)]
        if values:
            curves.append(values)
    return curves


def dump_curve(name: str, values):
    path = INPUT_DIR / name
    path.write_text(json.dumps(values, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved: {path}")


def main():
    mrpc_text = (DATA_ROOT / "roberta+mrpc.txt").read_text(encoding="utf-8", errors="ignore")
    mrpc_curves = extract_lists(mrpc_text)
    if len(mrpc_curves) >= 4:
        dump_curve("mrpc_1e3_offset.json", mrpc_curves[0])
        dump_curve("mrpc_1e3_standard.json", mrpc_curves[1])
        dump_curve("mrpc_5e4_offset.json", mrpc_curves[2])
        dump_curve("mrpc_5e4_standard.json", mrpc_curves[3])

    roberta_text = (DATA_ROOT / "roberta+sst对照组.txt").read_text(encoding="utf-8", errors="ignore")
    roberta_curves = extract_lists(roberta_text)
    if len(roberta_curves) >= 6:
        dump_curve("roberta_1e4_offset.json", roberta_curves[0])
        dump_curve("roberta_1e4_standard.json", roberta_curves[1])
        dump_curve("roberta_5e4_offset.json", roberta_curves[2])
        dump_curve("roberta_5e4_standard.json", roberta_curves[3])
        dump_curve("roberta_1e3_offset.json", roberta_curves[4])
        dump_curve("roberta_1e3_standard.json", roberta_curves[5])
        dump_curve("roberta_window_curve_1.json", roberta_curves[0])
        dump_curve("roberta_window_curve_2.json", roberta_curves[1])
        dump_curve("roberta_window_curve_3.json", roberta_curves[2])
        dump_curve("roberta_window_curve_4.json", roberta_curves[3])
        dump_curve("roberta_window_curve_5.json", roberta_curves[4])
        dump_curve("roberta_window_curve_6.json", roberta_curves[5])

    llama_text = (DATA_ROOT / "LIama+SST.txt").read_text(encoding="utf-8", errors="ignore")
    llama_curves = extract_lists(llama_text)
    if len(llama_curves) >= 4:
        dump_curve("llama_1e4_offset.json", llama_curves[0])
        dump_curve("llama_1e4_standard.json", llama_curves[1])
        dump_curve("llama_5e4_offset.json", llama_curves[2])
        dump_curve("llama_5e4_standard.json", llama_curves[3])
        dump_curve("llama_curve_1.json", llama_curves[0])
        dump_curve("llama_curve_2.json", llama_curves[1])
        dump_curve("llama_curve_3.json", llama_curves[2])
        dump_curve("llama_curve_4.json", llama_curves[3])


if __name__ == "__main__":
    main()
