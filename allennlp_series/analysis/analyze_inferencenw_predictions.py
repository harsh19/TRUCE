import json


def analyze(fname):
    data = open(fname, "r").readlines()
    data = [json.loads(s.strip()) for s in data]
    print("Dumping to ", fname + ".tsv")
    fw = open(fname + ".tsv", "w")
    vals = [
        "str(row['probs'][1])",
        "row['gtlabel']",
        "row['prediction']",
        "row['raw_text'][0]",
    ]
    fw.write("\t".join(vals))
    fw.write("\n")
    for row in data:
        vals = [
            str(row["probs"][1]),
            row["gtlabel"],
            row["prediction"],
            row["raw_text"][0],
        ]
        fw.write("\t".join(vals))
        fw.write("\n")
    fw.close()


if __name__ == "__main__":
    analyze("tmp/inferencebilstm2_13finaldata/predictions.json")
