import csv

gff3_file = r"C:\Users\papan\Desktop\Coffeomics\Coffea_arabica_Red_Bourbon_CC.gff3"
output_file = "transcript_go_carabica_red_bourbon.tsv"

with open(gff3_file) as infile, open(output_file, "w", newline="") as outfile:
    writer = csv.writer(outfile, delimiter="\t")
    writer.writerow(["transcript_id", "GO_term"])

    for line in infile:
        if line.startswith("#"):
            continue

        cols = line.strip().split("\t")
        if len(cols) < 9:
            continue

        feature_type = cols[2]
        attributes = cols[8]

        if feature_type == "mRNA":
            attr_dict = {}
            for field in attributes.split(";"):
                if "=" in field:
                    key, value = field.split("=", 1)
                    attr_dict[key] = value

            transcript_id = attr_dict.get("ID")
            go_terms = attr_dict.get("Ontology_term")

            if transcript_id and go_terms:
                for go in go_terms.split(","):
                    writer.writerow([transcript_id, go])