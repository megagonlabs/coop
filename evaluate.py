import json
from pathlib import Path

import click
import pandas as pd
import rouge
import torch

from coop.util import load_tokenizer, load_data, build_model


def evaluate(model, data, num_beams=4, debug=False):
    evaluator = rouge.Rouge(metrics=["rouge-n", "rouge-l"], max_n=2, limit_length=False, apply_avg=True,
                            stemming=True, ensure_compatibility=True)
    hyp, ref = [], []
    for x in data:
        out = model(x["src"], do_generate=True)
        summary_avg = model.generate(out.q.mean.mean(dim=0, keepdim=True), num_beams=num_beams)
        summary_avg = data.tokenizer.decode(summary_avg)
        hyp.extend(summary_avg)
        ref.append(x["summary"])

    sums = evaluator.get_scores(hyp, ref).items()
    scores = {"_".join((metric, "sum", k)): v for metric, vs in sums for k, v in vs.items()}

    if debug:
        print("Generated examples:")
        print("\n".join(hyp[:10]))

    return scores


@click.command()
@click.argument("log_dir", type=click.Path(exists=True))
@click.option("--debug", is_flag=True)
def main(log_dir, debug):
    log_dir = Path(log_dir)
    checkpoint = log_dir / "best.th"

    config = json.load(open(log_dir / "config.json"))
    src_tokenizer, tgt_tokenizer = load_tokenizer(config)
    _, dev, test = load_data(config, src_tokenizer, tgt_tokenizer)
    model = build_model(config).eval()
    model.load_state_dict(torch.load(checkpoint, map_location=lambda storage, loc: storage))

    if torch.cuda.is_available():
        model.cuda()
    scores = {}
    for data_type in ("dev", "test"):
        data = eval(data_type)
        scores[data_type] = evaluate(model, data, debug=debug)

    df = pd.DataFrame(scores)
    df.sort_index(inplace=True)
    print(df)


if __name__ == '__main__':
    main()
