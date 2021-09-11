import csv
import io
import json
import os
import shutil
from pathlib import Path

import click
import requests

from preprocess import strip_text


def amzn(output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    os.system("git clone https://github.com/ixlan/Copycat-abstractive-opinion-summarizer.git")

    for sp in ("dev", "test"):
        ins = []
        for x in csv.DictReader(open(f"Copycat-abstractive-opinion-summarizer/gold_summs/{sp}.csv"),
                                dialect='excel-tab'):
            ins.append({
                "reviews": [strip_text(x[f"rev{i + 1}"]) for i in range(8)],
                "summary": [x[f"summ{i + 1}"] for i in range(3)],
                "category": x["cat"],
                "prod_id": x["prod_id"]
            })

        json.dump(ins, open(output_dir / f"{sp}.json", "w"))
    shutil.rmtree("Copycat-abstractive-opinion-summarizer")


def yelp(output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    file = requests.get("https://s3.us-east-2.amazonaws.com/unsup-sum/summaries_0-200_cleaned.csv").content.decode()
    ins = []
    for x in csv.DictReader(io.StringIO(file)):
        ins.append({
            "reviews": [strip_text(x[f"Input.original_review_{i}"]) for i in range(8)],
            "summary": [x["Answer.summary"]],
            "review_ids": [x[f"Input.original_review_{i}_id"] for i in range(8)],
            "business_id": x["Input.business_id"]
        })
    json.dump(ins[:100], open(output_dir / "dev.json", "w"))
    json.dump(ins[100:], open(output_dir / "test.json", "w"))


@click.command()
@click.argument("data_type", type=click.Choice(("yelp", "amzn")), )
@click.argument("data_dir", type=click.Path() )
def main(data_type, data_dir):
    if data_type == "yelp":
        yelp(data_dir)
    else:
        amzn(data_dir)


if __name__ == '__main__':
    main()
