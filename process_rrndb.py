#!/usr/bin/env python3

import argparse

import pandas as pd


def parse_taxonomy_line(row):
    # Prepare the taxonomy and probability entries
    start_index = 5
    entries = iter(row[start_index:])
    # prob_entries = list(row[start_index::3])
    # tax_entries = list(row[start_index + 1 :: 3])
    # rank_entries = list(row[start_index + 2 :: 3])

    # Mapping for the taxonomic ranks
    tax_mapping = {
        "domain": "domain",
        "phylum": "phylum",
        "class": "class",
        "order": "order",
        "family": "family",
        "genus": "genus",
        "species": "species",
    }

    # Create an empty result dictionary
    result = {
        "otu": row[0],
        "domain": "unknown",
        "domain_p": "unknown",
        "phylum": "unknown",
        "phylum_p": "unknown",
        "class": "unknown",
        "class_p": "unknown",
        "order": "unknown",
        "order_p": "unknown",
        "family": "unknown",
        "family_p": "unknown",
        "genus": "unknown",
        "genus_p": "unknown",
        "species": "unknown",
        "species_p": "unknown",
    }

    # Populate the result dictionary with the provided data
    for tax, rank, prob in zip(entries, entries, entries):
        if rank in tax_mapping:
            result[tax_mapping[rank]] = tax
            result[tax_mapping[rank] + "_p"] = float(prob)

    return pd.Series(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str)
    parser.add_argument("-o", "--output", type=str)

    args = parser.parse_args()
    input_path = args.input
    output_path = args.output

    # Load the data from the tsv
    df = pd.read_table(input_path, header=None, names=list(range(30)))

    # Process each row
    df_res = df.apply(parse_taxonomy_line, axis=1)
    df_res.to_csv(output_path, sep="\t", index=False)

