import argparse
import os
import subprocess
import gzip
import shutil
from glob import glob
from collections import Counter
import tempfile
import sys
from contextlib import contextmanager

import pandas as pd
from Bio import Seq, SeqIO
from Bio.SeqRecord import SeqRecord
from umi_tools import UMIClusterer
from tqdm.auto import tqdm
from rich import print as rprint

from easy_amplicon.trim import (
    get_primer_set,
    get_rc,
    ECREC_DR,
    ECREC_LEADER,
    ECREC_SPACER0,
    RECORDING_PRIMER_3,
    RECORDING_PRIMER_5,
    get_min_overlap,
)
from easy_amplicon.utils import print_command, smart_open, find_paired_end_files
from easy_amplicon.read_processer.map_utils import map_se


@contextmanager
def suppress_stdout():
    """
    Context manager to temporarily suppress stdout.
    """
    # Save the original stdout file descriptor
    original_stdout = sys.stdout
    try:
        # Redirect stdout to a null device
        sys.stdout = open(os.devnull, "w")
        yield
    finally:
        # Restore the original stdout
        sys.stdout.close()
        sys.stdout = original_stdout


def fastp_merge(sample: str, read1: str, read2: str, output_dir: str, cpus: int):
    [
        os.makedirs(os.path.join(output_dir, d), exist_ok=True)
        for d in ["merged_direct", "unpaired", "unmerged", "failed", "log"]
    ]

    merge_cmd = [
        "fastp",
        "-i",
        read1,
        "-I",
        read2,
        "-w",
        str(cpus),
        # "--cut_tail",  # don't cut right, since some middle parts seem low quality
        "--merge",
        "--correction",
        "--overlap_len_require",
        "15",
        "--overlap_diff_limit",
        "5",
        "--overlap_diff_percent_limit",
        "20",
        "--merged_out",
        f"{output_dir}/merged_direct/{sample}.fq.gz",
        "--unpaired1",
        f"{output_dir}/unpaired/{sample}.1.fq.gz",
        "--unpaired2",
        f"{output_dir}/unpaired/{sample}.2.fq.gz",
        "--out1",
        f"{output_dir}/unmerged/{sample}.1.fq.gz",
        "--out2",
        f"{output_dir}/unmerged/{sample}.2.fq.gz",
        "--failed_out",
        f"{output_dir}/failed/{sample}.fq.gz",
        "--html",
        f"{output_dir}/log/{sample}.merge.html",
        "--json",
        f"{output_dir}/log/{sample}.merge.json",
    ]

    with open(f"{output_dir}/log/{sample}.merge.err", "w") as err_file:
        print_command(merge_cmd)
        subprocess.run(merge_cmd, stderr=err_file)


def run_cutadapt(
    input_file: str,
    output_file: str,
    *,
    untrimmed_output: str | bool | None = None,
    log_file: str | None = None,
    adapter_5: str | None = None,
    adapter_3: str | None = None,
    e: int | float = 0.2,
    cores: int,
    min_length: int | None = None,
    rc: bool = False,
):
    # makedirs for all output locations
    [
        os.makedirs(os.path.dirname(f), exist_ok=True)
        for f in [output_file, untrimmed_output, log_file]
        if isinstance(f, str)
    ]

    def add_adapter(cmd: list[str]):
        if adapter_5 is not None:
            cmd.extend(["-g", adapter_5])
        if adapter_3 is not None:
            cmd.extend(["-a", adapter_3])

    if rc:
        seqkit_cmd_1 = [
            "seqkit",
            "seq",
            "-r",
            "-p",
            "-t",
            "DNA",
            "--validate-seq",
            input_file,
        ]
        cutadapt_cmd = [
            "cutadapt",
            "-e",
            str(e),
            "--cores",
            str(cores),
        ]
        add_adapter(cutadapt_cmd)
        if untrimmed_output is False:
            cutadapt_cmd.append("--discard-untrimmed")
        elif isinstance(untrimmed_output, str):
            cutadapt_cmd.extend(["--untrimmed-output", untrimmed_output])
        else:
            raise ValueError("untrimmed_output must be a string or False")
        if min_length is not None:
            cutadapt_cmd.extend(["--minimum-length", str(min_length)])
        cutadapt_cmd.append("-")
        seqkit_cmd_2 = [
            "seqkit",
            "seq",
            "-r",
            "-p",
            "-t",
            "DNA",
            "--validate-seq",
            "-",
            "-o",
            output_file,
        ]

        if log_file is not None:
            with open(log_file, "w") as log:
                print_command(seqkit_cmd_1)
                p1 = subprocess.Popen(seqkit_cmd_1, stdout=subprocess.PIPE)
                print_command(cutadapt_cmd)
                p2 = subprocess.Popen(
                    cutadapt_cmd,
                    stdin=p1.stdout,
                    stdout=subprocess.PIPE,
                    stderr=log,
                )
                print_command(seqkit_cmd_2)
                p3 = subprocess.Popen(
                    seqkit_cmd_2, stdin=p2.stdout, stdout=subprocess.PIPE
                )
                _, _ = p3.communicate()
        else:
            print_command(seqkit_cmd_1)
            p1 = subprocess.Popen(seqkit_cmd_1, stdout=subprocess.PIPE)
            print_command(cutadapt_cmd)
            p2 = subprocess.Popen(
                cutadapt_cmd,
                stdin=p1.stdout,
                stdout=subprocess.PIPE,
            )
            print_command(seqkit_cmd_2)
            p3 = subprocess.Popen(seqkit_cmd_2, stdin=p2.stdout, stdout=subprocess.PIPE)
            _, _ = p3.communicate()

    else:
        cutadapt_cmd = [
            "cutadapt",
            "-e",
            str(e),
            "-o",
            output_file,
            "--cores",
            str(cores),
        ]
        add_adapter(cutadapt_cmd)
        if untrimmed_output is False:
            cutadapt_cmd.append("--discard-untrimmed")
        elif isinstance(untrimmed_output, str):
            cutadapt_cmd.extend(["--untrimmed-output", untrimmed_output])
        elif untrimmed_output is not None:
            raise ValueError("untrimmed_output must be a string or False")
        if min_length is not None:
            cutadapt_cmd.extend(["--minimum-length", str(min_length)])
        cutadapt_cmd.append(input_file)

        if log_file is not None:
            with open(log_file, "w") as log:
                print_command(cutadapt_cmd)
                _ = subprocess.run(cutadapt_cmd, stdout=log)
        else:
            print_command(cutadapt_cmd)
            _ = subprocess.run(cutadapt_cmd)


def cutadapt_fix_1(
    input1: str,
    input2: str,
    output1: str,
    output2: str,
    *,
    log_file: str,
    adapter: str,  # 3' adapter of read 1
    e: int | float = 0.15,
    cores: int,
    min_length: int | None = None,
):
    """Some of reads cannot be merged, likely long amplicons. Here we merge them by
    trimming rightmost DR sequence from 3' end in retain mode. Later we will directly
    stitching together the two reads using biopython by reverse complementing read2.

    In this function, we need to use cutadapt in pair mode, but since read 1 adapter is
    rightmost 3', which cutadapt doesn't implement, we need to first reverse complement
    read 1 and then trim from 5', like in the above run_cutadapt function. We therefore
    specify input and output path for read 2 in cutadapt but pass read 1 by piping.

    Read 2 is not modified at all during cutadapt, but we still need to use pair mode to
    drop those read pairs where read 1 is discarded.

    We will also simply discard untrimmed reads here.
    """
    [
        os.makedirs(os.path.dirname(f), exist_ok=True)
        for f in [output1, output2, log_file]
        if f is not None
    ]
    seqkit_cmd_1 = [
        "seqkit",
        "seq",
        "-r",
        "-p",
        "-t",
        "DNA",
        "--validate-seq",
        input1,
    ]
    cutadapt_cmd = [
        "cutadapt",
        "-e",
        str(e),
        "-g",
        adapter,
        "--discard-untrimmed",
        "--action",
        "retain",
        "--cores",
        str(cores),
        "-o",
        "-",
        "-p",
        output2,
    ]
    if min_length is not None:
        cutadapt_cmd.extend(["--minimum-length", str(min_length)])
    cutadapt_cmd.extend(["-", input2])
    seqkit_cmd_2 = [
        "seqkit",
        "seq",
        "-r",
        "-p",
        "-t",
        "DNA",
        "--validate-seq",
        "-",
        "-o",
        output1,
    ]
    with open(log_file, "w") as log:
        print_command(seqkit_cmd_1)
        p1 = subprocess.Popen(seqkit_cmd_1, stdout=subprocess.PIPE)
        print_command(cutadapt_cmd)
        p2 = subprocess.Popen(
            cutadapt_cmd,
            stdin=p1.stdout,
            stdout=subprocess.PIPE,
            stderr=log,
        )
        print_command(seqkit_cmd_2)
        p3 = subprocess.Popen(seqkit_cmd_2, stdin=p2.stdout, stdout=subprocess.PIPE)
        p3.communicate()


def cutadapt_fix_2(
    input1: str,
    input2: str,
    output1: str,
    output2: str,
    *,
    log_file: str | None = None,
    # adapter1: str,  # 3' adapter of read 1
    adapter2: str,  # 3' adapter of read 2
    e: int | float = 0.15,
    cores: int,
    min_length: int | None = None,
):
    [
        os.makedirs(os.path.dirname(f), exist_ok=True)
        for f in [output1, output2, log_file]
        if f is not None
    ]
    cutadapt_cmd = [
        "cutadapt",
        "-e",
        str(e),
        # "-a",
        # adapter1,
        "-A",
        adapter2,
        "--cores",
        str(cores),
        "-o",
        output1,
        "-p",
        output2,
    ]
    if min_length is not None:
        cutadapt_cmd.extend(["--minimum-length", str(min_length)])
    cutadapt_cmd.extend([input1, input2])
    if log_file is not None:
        with open(log_file, "w") as log:
            print_command(cutadapt_cmd)
            subprocess.run(cutadapt_cmd, stdout=log)
    else:
        print_command(cutadapt_cmd)
        subprocess.run(cutadapt_cmd)


def stitch_reads(read1: str, read2: str, output: str):
    os.makedirs(os.path.dirname(output), exist_ok=True)
    r1 = smart_open(read1)
    r2 = smart_open(read2)
    out = smart_open(output, "w")
    for r1_record, r2_record in zip(SeqIO.parse(r1, "fastq"), SeqIO.parse(r2, "fastq")):
        r1_seq = r1_record.seq
        r2_seq = get_rc(r2_record.seq)
        r1_qual = r1_record.letter_annotations["phred_quality"]
        r2_qual = r2_record.letter_annotations["phred_quality"][::-1]
        record = SeqRecord(
            Seq.Seq(r1_seq + r2_seq),
            id=r1_record.id,
            description=r1_record.description,
        )
        record.letter_annotations["phred_quality"] = r1_qual + r2_qual
        SeqIO.write(record, out, "fastq")
    r1.close()
    r2.close()
    out.close()


def cutadapt_remove_nonedited(
    read1: str,
    read2: str,
    output1: str,
    output2: str,
    *,
    log_file: str | None = None,
    adapter: str,
    cpus: int,
):
    cutadapt_cmd = [
        "cutadapt",
        "-a",
        adapter,
        "--discard-trimmed",
        "-o",
        output1,
        "-p",
        output2,
        "--cores",
        str(cpus),
        read1,
        read2,
    ]
    print_command(cutadapt_cmd)
    if log_file is not None:
        with open(log_file, "w") as log:
            subprocess.run(cutadapt_cmd, stdout=log)
    else:
        subprocess.run(cutadapt_cmd)


def makedirs(*directories):
    [os.makedirs(d, exist_ok=True) for d in directories]


def process_one_sample(
    sample: str,
    read1: str,
    read2: str,
    output_dir: str,
    cpus: int,
    concise_mode: bool = False,
):
    merge_dir = os.path.join(output_dir, "merge")
    os.makedirs(merge_dir, exist_ok=True)

    # remove non edited reads
    os.makedirs(os.path.join(merge_dir, "log"), exist_ok=True)

    if concise_mode:
        os.makedirs(os.path.join(merge_dir, "concise"), exist_ok=True)
        no_editing_seq = f"{ECREC_LEADER}{ECREC_DR}{ECREC_SPACER0}"
        read1_out = f"{merge_dir}/concise/{sample}.1.fq.gz"
        read2_out = f"{merge_dir}/concise/{sample}.2.fq.gz"
        cutadapt_remove_nonedited(
            read1,
            read2,
            output1=read1_out,
            output2=read2_out,
            log_file=f"{merge_dir}/log/{sample}.concise.out",
            adapter=f"{no_editing_seq};min_overlap={int(len(no_editing_seq) * 0.8)}",
            cpus=cpus,
        )
        read1, read2 = read1_out, read2_out

    # Merge pairs and remove adapters
    # log_dir = f"{merge_dir}/log"
    # output_md = f"{merge_dir}/merged_direct/{sample}.fq.gz"
    fastp_merge(
        sample,
        read1=read1,
        read2=read2,
        # read1=f"{merge_dir}/concise/{sample}.1.fq.gz",
        # read2=f"{merge_dir}/concise/{sample}.2.fq.gz",
        output_dir=merge_dir,
        cpus=cpus,
    )

    # unmerged fix
    # output_umf_dir = f"{merge_dir}/unmerged_fix"
    # output_pseduo_merge = f"{merge_dir}/merged_pseudo/{sample}.fq.gz"
    # cutadapt_fix_2(
    #     input1=f"{merge_dir}/unmerged/{sample}.1.fq.gz",
    #     input2=f"{merge_dir}/unmerged/{sample}.2.fq.gz",
    #     output1=f"{output_umf_dir}/{sample}.1.fq.gz",
    #     output2=f"{output_umf_dir}/{sample}.2.fq.gz",
    #     log_file=f"{log_dir}/{sample}.fix.out",
    #     # here adapter is non-internal
    #     adapter2=get_rc(ECREC_DR) + f"X;min_overlap=5",
    #     cores=cpus,
    # )
    # stitch_reads(
    #     read1=f"{output_umf_dir}/{sample}.1.fq.gz",
    #     read2=f"{output_umf_dir}/{sample}.2.fq.gz",
    #     output=output_pseduo_merge,
    # )

    # # concat merged and pseudo-merged
    output_merged = f"{merge_dir}/merged_direct/{sample}.fq.gz"
    # os.makedirs(os.path.dirname(output_merged), exist_ok=True)
    # zcat_proc = subprocess.Popen(
    #     ["zcat", output_md, output_pseduo_merge], stdout=subprocess.PIPE
    # )
    # with smart_open(output_merged, "w") as handle:
    #     pigz_proc = subprocess.Popen(
    #         ["pigz", "-p", str(cpus), "-c"], stdin=zcat_proc.stdout, stdout=handle
    #     )
    #     pigz_proc.communicate()

    # clean illumina adapters
    output_rm_adapters = f"{merge_dir}/cleaned/{sample}.fq.gz"
    run_cutadapt(
        input_file=output_merged,
        output_file=output_rm_adapters,
        log_file=f"{merge_dir}/log/{sample}.clean.out",
        adapter_3=f"{RECORDING_PRIMER_5};rightmost;min_overlap={get_min_overlap(RECORDING_PRIMER_5, frac=0.5)}"
        f"...{RECORDING_PRIMER_3};min_overlap={get_min_overlap(RECORDING_PRIMER_3, frac=0.5)}",
        min_length=100,
        cores=cpus,
    )

    # 5' UMI
    umi5_dir = os.path.join(output_dir, "umi5")
    makedirs(umi5_dir)
    output_umi5_rest = f"{umi5_dir}/rest/{sample}.fq.gz"
    output_umi5_umi = f"{umi5_dir}/umi/{sample}.fq.gz"
    adapter_3 = ECREC_LEADER
    adapter_3 = f"{adapter_3};min_overlap={get_min_overlap(adapter_3)}"
    adapter_5 = f"^NNNNNN{ECREC_LEADER}"
    run_cutadapt(
        input_file=output_rm_adapters,
        output_file=output_umi5_umi,
        untrimmed_output=f"{umi5_dir}/umi_untrimmed/{sample}.fq.gz",
        log_file=f"{umi5_dir}/log/{sample}.umi.out",
        adapter_3=adapter_3,
        e=0.3,
        cores=cpus,
    )
    run_cutadapt(
        input_file=output_rm_adapters,
        output_file=output_umi5_rest,
        untrimmed_output=f"{umi5_dir}/rest_untrimmed/{sample}.fq.gz",
        log_file=f"{umi5_dir}/log/{sample}.rest.out",
        adapter_5=adapter_5,
        e=0.3,
        cores=cpus,
    )

    # 3' UMI
    umi3_dir = os.path.join(output_dir, "umi3")
    makedirs(umi3_dir)
    output_umi3_rest = f"{umi3_dir}/rest/{sample}.fq.gz"
    output_umi3_umi = f"{umi3_dir}/umi/{sample}.fq.gz"
    adapter_5 = ECREC_DR + ECREC_SPACER0
    adapter_5 = f"{adapter_5};min_overlap={get_min_overlap(adapter_5)};rightmost"
    adapter_3 = f"{ECREC_DR}{ECREC_SPACER0}NNNNNN$"
    run_cutadapt(
        input_file=output_umi5_rest,
        output_file=output_umi3_umi,
        untrimmed_output=f"{umi3_dir}/umi_untrimmed/{sample}.fq.gz",
        log_file=f"{umi3_dir}/log/{sample}.umi.out",
        adapter_5=adapter_5,
        cores=cpus,
    )
    run_cutadapt(
        input_file=output_umi5_rest,
        output_file=output_umi3_rest,
        untrimmed_output=f"{umi3_dir}/rest_untrimmed/{sample}.fq.gz",
        log_file=f"{umi3_dir}/log/{sample}.rest.out",
        adapter_3=adapter_3,
        cores=cpus,
    )

    # Spacer rounds
    round_num = 1
    current_input = output_umi3_rest
    while True:
        spacer_dir = os.path.join(output_dir, f"spacer{round_num}")
        makedirs(spacer_dir)
        output_spacer_rest = f"{spacer_dir}/rest/{sample}.fq.gz"
        output_spacer_spacer = f"{spacer_dir}/spacer/{sample}.fq.gz"
        run_cutadapt(
            input_file=current_input,
            output_file=output_spacer_spacer,
            untrimmed_output=f"{spacer_dir}/spacer_untrimmed/{sample}.fq.gz",
            log_file=f"{spacer_dir}/log/{sample}.spacer.out",
            adapter_5=ECREC_DR + f";min_overlap={get_min_overlap(ECREC_DR)};rightmost",
            min_length=5,
            cores=cpus,
        )
        run_cutadapt(
            input_file=current_input,
            output_file=output_spacer_rest,
            untrimmed_output=f"{spacer_dir}/rest_untrimmed/{sample}.fq.gz",
            log_file=f"{spacer_dir}/log/{sample}.rest.out",
            adapter_5=get_rc(ECREC_DR) + f";min_overlap={get_min_overlap(ECREC_DR)}",
            min_length=1,
            cores=cpus,
            rc=True,
        )
        # NOTE: Take the common seq between output_spacer_spacer and output_spacer_rest as the
        # final output_spacer_rest, since we only feed to the next round for those sequences that
        # have spacer at this round.
        # We use seqkit for this and follows its document, which prefer grep over common with 2 files.
        # The input for grep is the bigger file (in our case output_spacer_rest) and the pattern file
        # is the smaller file (output_spacer_spacer).
        # save it to a temp fq.gz
        with tempfile.NamedTemporaryFile(suffix=".fq.gz", delete=False) as temp:
            output_spacer_rest_temp = temp.name
            args_seq = ["seqkit", "seq", "--name", "--only-id", output_spacer_spacer]
            args_grep = [
                "seqkit",
                "grep",
                "--pattern-file",
                "-",
                output_spacer_rest,
                "-o",
                output_spacer_rest_temp,
            ]
            print_command(args_seq)
            prog_seq = subprocess.Popen(
                args_seq, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
            )
            print_command(args_grep)
            prog_grep = subprocess.Popen(
                args_grep,
                stdin=prog_seq.stdout,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
            prog_grep.communicate()
            # args = [
            #     "seqkit",
            #     "common",
            #     output_spacer_rest,
            #     output_spacer_spacer,
            #     "-o",
            #     output_spacer_rest_temp,
            # ]
            # print_command(args)
            # subprocess.run(args)
            shutil.move(output_spacer_rest_temp, output_spacer_rest)

        if all_sequences_empty(output_spacer_rest):
            break

        current_input = output_spacer_rest
        round_num += 1

    # concat spacer for later clustering by running seqkit concat <spacer1> <spacer2> ... <spacern> -o <output>
    for k in range(
        1,
        (
            round_num
            if all_sequences_empty(f"{spacer_dir}/spacer/{sample}.fq.gz")
            else round_num + 1
        ),
    ):
        spacer_files = [
            f"{output_dir}/spacer{i}/spacer/{sample}.fq.gz" for i in range(1, k + 1)
        ]
        output_spacer = f"{output_dir}/spacer{k}/spacer_concat/{sample}.fq.gz"
        os.makedirs(os.path.dirname(output_spacer), exist_ok=True)
        if k == 1:
            shutil.copy(spacer_files[0], output_spacer)
        else:
            seqkit_log = f"{output_dir}/spacer{k}/log/{sample}.concat.err"
            seqkit_cmd = ["seqkit", "concat"] + spacer_files + ["-o", output_spacer]
            print_command(seqkit_cmd)
            with open(seqkit_log, "w") as log:
                subprocess.run(seqkit_cmd, stderr=log)

        # run mmseqs2 to cluster concatenated spacer right after like this:
        mmseqs2_prefix = f"{output_dir}/spacer{k}/mmseqs2_clust/{sample}"
        mmseqs2_log = f"{output_dir}/spacer{k}/log/{sample}.clust.err"
        mmseqs2_out = f"{output_dir}/spacer{k}/log/{sample}.clust.out"
        os.makedirs(os.path.dirname(mmseqs2_prefix), exist_ok=True)
        with tempfile.TemporaryDirectory() as temp_dir:
            mmseqs_cmd = [
                "mmseqs",
                "easy-cluster",
                output_spacer,
                mmseqs2_prefix,
                temp_dir,
                "--min-seq-id",
                "0.95",
                "-c",
                "0.8",
                "--cov-mode",
                "1",
                "--remove-tmp-files",
            ]
            print_command(mmseqs_cmd)
            with open(mmseqs2_log, "w") as log, open(mmseqs2_out, "w") as out:
                subprocess.run(mmseqs_cmd, stderr=log, stdout=out)


def all_sequences_empty(fastq_file: str) -> bool:
    """Check if all sequences in a FASTQ file are empty, or if the file itself is empty."""
    with gzip.open(fastq_file, "rt") as handle:
        for record in SeqIO.parse(handle, "fastq"):
            if len(record.seq) > 0:
                return False
    return True


def collect_spacer_info(
    data_dir: str, sample: str, umi_length: int = 6
) -> pd.DataFrame | None:
    """Return a dataframe and a list of spacers.
    The dataframe contains information for each read. Read name being the index, and columns are:
        - umi_5: str
        - umi_3: str
        - num_spacers: int. Number of spacers in the read
        - spacer_idxs: list[int]. List of spacer index of the following list of spacers.
        - centroid: str. The index of the centroid sequence of the read.
    All reads that have either 5' UMI or 3' UMI will be kept, though following analysis will
    focus on reads with both.

    The list of spacers is a list of spacer sequences that's deduplicated. These sequences
    will be later clustered with MMSeqs2 and centroids will be BLASTed to obtain taxonomy
    information.
    """
    umi5_dir = os.path.join(data_dir, "umi5")
    umi3_dir = os.path.join(data_dir, "umi3")
    umi5_path = os.path.join(umi5_dir, f"umi/{sample}.fq.gz")
    umi3_path = os.path.join(umi3_dir, f"umi/{sample}.fq.gz")
    spacer_dirs = glob(os.path.join(data_dir, "spacer*"))
    spacer_orders = [int(i[-1]) for i in spacer_dirs]

    df_data = []
    spacers_set = set()
    spacers = []
    spacer_idx = -1
    # first add 5' umi and 3' umi
    with gzip.open(umi5_path, "rt") as f:
        seq2umi5 = {record.id: str(record.seq) for record in SeqIO.parse(f, "fastq")}
    umi5_df = pd.Series(seq2umi5, name="umi_5").to_frame()
    with gzip.open(umi3_path, "rt") as f:
        seq2umi3 = {record.id: str(record.seq) for record in SeqIO.parse(f, "fastq")}
    umi3_df = pd.Series(seq2umi3, name="umi_3").to_frame()
    df_data = pd.merge(
        umi5_df, umi3_df, how="outer", left_index=True, right_index=True
    ).fillna("")
    if df_data.empty:
        return None

    # remove reads that don't have both 5' and 3' UMI or have empty UMI
    # df_data = df_data.replace("", np.nan).dropna()
    dfs_clust = []
    seqs = {
        i: {"num_spacers": 0, "spacer_idxs": [], "spacer_lens": [], "spacer_qual": 0}
        for i in df_data.index
    }
    for spacer_order in sorted(spacer_orders):
        spacer_dir = os.path.join(data_dir, f"spacer{spacer_order}")
        spacer_path = os.path.join(spacer_dir, "spacer", f"{sample}.fq.gz")
        clust_path = os.path.join(spacer_dir, "mmseqs2_clust", f"{sample}_cluster.tsv")
        if not os.path.exists(spacer_path):
            continue
        with gzip.open(spacer_path, "rt") as f:
            for idx, record in enumerate(SeqIO.parse(f, "fastq")):
                seq = str(record.seq)
                qual = record.letter_annotations["phred_quality"]
                if not seq:  # skip empty sequences
                    continue
                if seq not in spacers_set:
                    spacers.append(seq)
                    spacers_set.add(seq)
                    spacer_idx += 1
                try:
                    seqs[record.id]["num_spacers"] = spacer_order
                    seqs[record.id]["spacer_idxs"].append(spacer_idx)
                    seqs[record.id]["spacer_lens"].append(len(seq))
                    seqs[record.id]["spacer_qual"] += sum(qual)
                except KeyError:
                    print(
                        f"Spacer {spacer_order} of read {record.id} not found in UMI files."
                    )
        if os.path.exists(clust_path):
            dfs_clust.append(pd.read_table(clust_path, names=["centroid", "seq"]))
    df_data = pd.concat([df_data, pd.DataFrame(seqs).transpose()], axis=1)

    for df in dfs_clust:
        df_data.loc[df.seq, "centroid"] = df.centroid.to_numpy()
    df_data["spacer_qual"] = df_data["spacer_qual"] / (
        df_data["spacer_lens"].map(sum) + 1e-8
    )
    df_data["umi"] = df_data.umi_5 + df_data.umi_3
    df_data["good_umi"] = (df_data.umi_5.str.len() == umi_length) & (
        df_data.umi_3.str.len() == umi_length
    )
    return df_data


def correct_umi(df_data: pd.DataFrame) -> dict[str, str]:
    """UMI correction and deduplication with UMI_tools"""
    clusterer = UMIClusterer(cluster_method="directional")
    read2umi_corrected = {}
    for g, df_g in df_data.query("good_umi").groupby("centroid", dropna=False):
        umis = list(map(lambda x: x.encode(), df_g.umi))
        if len(umis) < 2:
            read2umi_corrected[df_g.index[0]] = umis[0].decode()
        else:
            umis = Counter(umis)
            clustered_umis = clusterer(umis, threshold=1)
            umi2umi_corrected = {u: umis[0] for umis in clustered_umis for u in umis}
            for read, umi in zip(df_g.index, df_g.umi, strict=True):
                read2umi_corrected[read] = umi2umi_corrected[umi.encode()].decode()
    return read2umi_corrected


def add_target(df_data: pd.DataFrame, collapse_mode: str | None = None) -> pd.DataFrame:
    """Add target sequence to the dataframe"""
    df_data = df_data.copy()
    df_data["target"] = None
    df_data["size"] = -1
    if collapse_mode is None:
        pass
    else:
        if collapse_mode == "umi":
            groupby = ["num_spacers", "centroid", "umi_corrected"]
        elif collapse_mode == "centroid":
            groupby = ["centroid"]
        else:
            raise ValueError(f"Unknown collapse mode: {collapse_mode}")
        for _, df_g in df_data.groupby(groupby):
            target = df_g.sort_values("spacer_qual", ascending=False).iloc[0].name
            for i in df_g.index:
                df_data.loc[i, "target"] = target
            df_data.loc[target, "size"] = df_g.shape[0]
    return df_data


def agg_arrays(df_data: pd.DataFrame) -> pd.DataFrame:
    return df_data.loc[df_data.index == df_data.target].copy().sort_values("size")


def collect_spacers(
    df_data: pd.DataFrame,
    data_dir: str,
    output_path: str,
    sample: str,
) -> None:
    spacer_orders = [int(i[-1]) for i in glob(os.path.join(data_dir, "spacer*"))]
    # output_path = os.path.join(data_dir, "collect_spacers", "spacer", f"{sample}.fq.gz")
    # reads without spacers are not relevant
    # if collapse_mode == "umi":
    #     # For those that have the same num spacers and umi_corrected and centroid, take
    #     # the one with the highest spacer_qual.
    #     df_data = df_data.reset_index().groupby(["num_spacers", "centroid", "umi_corrected"]).apply(
    #         lambda x: x.sort_values("spacer_qual", ascending=False).iloc[0]
    #     ).set_index("index")
    # elif collapse_mode == "centroid":
    #     # For those that have the centroid, take the one with the highest spacer_qual.
    #     df_data = df_data.reset_index().groupby("centroid").apply(
    #         lambda x: x.sort_values("spacer_qual", ascending=False).iloc[0]
    #     ).set_index("index")
    # else:
    #     if collapse_mode is not None:
    #         raise ValueError(f"Unknown collapse mode: {collapse_mode}")

    good_reads = set(df_data.index)

    with gzip.open(output_path, "wt") as f_out:
        for spacer_order in sorted(spacer_orders):
            spacer_path = os.path.join(
                data_dir, f"spacer{spacer_order}", "spacer", f"{sample}.fq.gz"
            )
            if not os.path.exists(spacer_path):
                continue
            with gzip.open(spacer_path, "rt") as f_in:
                for record in SeqIO.parse(f_in, "fastq"):
                    if record.id in good_reads:
                        size = df_data.loc[record.id, "size"]
                        # add order to the read name as another :<order>
                        record.id = f"{record.id};order={spacer_order};size={size}"
                        SeqIO.write(record, f_out, "fastq")


def get_fastq_length(file_path: str) -> int:
    """
    Calculate the number of sequences in a gzipped FASTQ file.

    Args:
        file_path (str): Path to the gzipped FASTQ file.

    Returns:
        int: Number of sequences in the gzipped FASTQ file.
    """
    # Return 0 if file is empty
    # if os.path.getsize(file_path) == 0:
    #     return 0

    # Use subprocess to count the number of lines in the gzipped file
    if file_path.endswith(".fa.gz") or file_path.endswith(".fasta.gz"):
        prog = "zcat"
        factor = 2
    elif file_path.endswith(".fq.gz") or file_path.endswith(".fastq.gz"):
        prog = "zcat"
        factor = 4
    elif (
        file_path.endswith(".fa")
        or file_path.endswith(".fasta")
        or file_path.endswith(".fna")
    ):
        prog = "cat"
        factor = 2
    elif file_path.endswith(".fq") or file_path.endswith(".fastq"):
        prog = "cat"
        factor = 4
    else:
        raise ValueError(f"Unknown file extension: {file_path}")

    result = subprocess.run(
        # [prog, file_path, "|", "wc", "-l"],
        f"{prog} {file_path} | wc -l",
        capture_output=True,
        text=True,
        shell=True,
        check=True,
    )

    if result.returncode != 0:
        raise RuntimeError(f"Error counting lines in file: {result.stderr}")

    # Each sequence in a FASTQ file spans four lines
    num_lines = int(result.stdout.strip())
    num_sequences = num_lines // factor
    return num_sequences


# if __name__ == "__main__":
def main():
    # data_dir = "/mnt/c/aws_data/20240701_yuanyuan_recording/read_process"
    # fastq_dir = "/mnt/c/aws_data/20240701_yuanyuan_recording/fastq_raw"
    # ref_genome = "/mnt/c/aws_data/20240701_yuanyuan_recording/ref/ecoli_bl21_prec.fna"

    parser = argparse.ArgumentParser(description="Process some directories and files.")
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Path to the data directory"
    )
    parser.add_argument(
        "--fastq_dir", type=str, required=True, help="Path to the fastq directory"
    )
    parser.add_argument(
        "--collapse_mode",
        type=str,
        default=None,
        choices=["umi", "centroid"],
        help="Collapse mode for spacer sequences",
    )
    parser.add_argument(
        "--ref_genome",
        type=str,
        required=True,
        help="Path to the reference genome file",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress output")

    args = parser.parse_args()

    output_dir = args.output_dir
    fastq_dir = args.fastq_dir
    collapse_mode = args.collapse_mode
    ref_genome = args.ref_genome
    quiet = args.quiet

    if quiet:
        global print_command

        def print_command(*args, **kwargs):
            return None

    bar = tqdm(find_paired_end_files(fastq_dir), leave=False)
    for i, (read1, read2, sample) in enumerate(bar):
        if "Undetermined" in sample:
            continue
        # if i < 66:
        #     continue
        bar.set_description(sample)
        # ==================================
        rprint(f"[bold green]Extracting spacers from reads of {sample}..[/bold green]")
        with suppress_stdout():
            process_one_sample(
                sample,
                # read1=glob(f"{fastq_dir}/{sample}_S*_L001_R1_001.fastq.gz")[0],
                # read2=glob(f"{fastq_dir}/{sample}_S*_L001_R2_001.fastq.gz")[0],
                read1=read1,
                read2=read2,
                output_dir=output_dir,
                cpus=4,
            )

        # # ==================================
        rprint("[bold green]Correcting UMI..[/bold green]")
        spacer_info_path = f"{output_dir}/collect_spacers/spacer_info/{sample}_full.tsv"
        spacer_info_agg_path = f"{output_dir}/collect_spacers/spacer_info/{sample}.tsv"
        os.makedirs(os.path.dirname(spacer_info_path), exist_ok=True)
        df_data = collect_spacer_info(output_dir, sample)
        if df_data is None:
            print(f"WARNING: No arrays with UMI for {sample}, skipping mapping")
            continue
        if df_data.num_spacers.max() == 0:
            print(f"WARNING: No arryas with spacer found in {sample}, skipping mapping")
            df_data.to_csv(spacer_info_path, sep="\t")
            continue
        read2umi_corrected = correct_umi(df_data)
        df_data["umi_corrected"] = df_data.index.map(read2umi_corrected)
        assert df_data.query("good_umi").umi_corrected.notna().all()
        df_data_full = add_target(df_data, collapse_mode)
        df_data_agg = agg_arrays(df_data_full)
        # save spacer info
        df_data_full.to_csv(spacer_info_path, sep="\t")
        df_data_agg.to_csv(spacer_info_agg_path, sep="\t")

        # ==================================
        rprint("[bold green]Collecting spacer sequences..[/bold green]")
        df_data_full = pd.read_table(spacer_info_path, index_col=0)
        df_data_agg = pd.read_table(spacer_info_agg_path, index_col=0)
        output_full_path = f"{output_dir}/collect_spacers/spacer_full/{sample}.fq.gz"
        output_agg_path = f"{output_dir}/collect_spacers/spacer/{sample}.fq.gz"
        os.makedirs(os.path.dirname(output_agg_path), exist_ok=True)
        os.makedirs(os.path.dirname(output_full_path), exist_ok=True)
        collect_spacers(
            df_data=df_data_agg,
            data_dir=output_dir,
            output_path=output_agg_path,
            sample=sample,
        )
        collect_spacers(
            df_data=df_data_full,
            data_dir=output_dir,
            output_path=output_full_path,
            sample=sample,
        )

        # # ==================================
        rprint(
            "[bold green]Mapping reads to genome to distinguish exogenous and "
            "endogenous spacers..[/bold green]"
        )
        makedirs(
            *[
                f"{output_dir}/collect_spacers/{i}"
                for i in ["map", "self", "other", "log"]
            ]
        )
        map_se(
            f"{output_dir}/collect_spacers/spacer/{sample}.fq.gz",
            ref_genome,
            bam_file=f"{output_dir}/collect_spacers/map/{sample}.bam",
            bwa_log=f"{output_dir}/collect_spacers/log/{sample}.bwa.log",
            mapped_reads=f"{output_dir}/collect_spacers/self/{sample}.fq.gz",
            unmapped_reads=f"{output_dir}/collect_spacers/other/{sample}.fq.gz",
            args="relaxed",
        )

        # ==================================
        # ref_genome = (
        #     "/mnt/c/aws_data/20240701_yuanyuan_recording/ref/escherichia_others.fna"
        # )
        # makedirs(
        #     *[
        #         f"{output_dir}/collect_spacers/{i}"
        #         for i in ["map_e", "self_no_e", "other_no_e", "other_is_e", "log"]
        #     ]
        # )
        # map_se(
        #     f"{output_dir}/collect_spacers/other/{sample}.fq.gz",
        #     ref_genome,
        #     bam_file=f"{output_dir}/collect_spacers/map_e/{sample}.bam",
        #     bwa_log=f"{output_dir}/collect_spacers/log/{sample}.bwa_e.log",
        #     mapped_reads=f"{output_dir}/collect_spacers/other_no_e/{sample}.fq.gz",
        #     unmapped_reads=f"{output_dir}/collect_spacers/other_is_e/{sample}.fq.gz",
        # )

        # ==================================
        # rprint("[bold green]BLSATing non-self to plasmid database")
        # plasmid_db = (
        #     "/mnt/c/aws_data/20240701_yuanyuan_recording/db/PlasmidDatabaseJan18.db"
        # )
        # subprocess.run(
        #     [
        #         "seqkit",
        #         "fq2fa",
        #         f"{output_dir}/collect_spacers/other/{sample}.fq.gz",
        #         "-o",
        #         f"{output_dir}/collect_spacers/other/{sample}.fa",
        #     ]
        # )
        # os.makedirs(f"{output_dir}/collect_spacers/other_blastn_plasmid", exist_ok=True)
        # subprocess.run(
        #     [
        #         "blastn",
        #         "-query",
        #         f"{output_dir}/collect_spacers/other/{sample}.fa",
        #         "-db",
        #         plasmid_db,
        #         "-task",
        #         "megablast",
        #         "-word_size",
        #         "10",
        #         "-perc_identity",
        #         "95",
        #         "-dust",
        #         "yes",
        #         "-evalue",
        #         "1e-2",
        #         "-max_target_seqs",
        #         "10000",
        #         "-num_threads",
        #         "16",
        #         "-out",
        #         f"{output_dir}/collect_spacers/other_blastn_plasmid/{sample}.out",
        #         "-outfmt",
        #         "6 qseqid qlen sseqid pident length qstart qend sstart send evalue bitscore slen staxids",
        #     ]
        # )

        # ==================================
        # rprint(
        #     "[bold green]Mapping reads to genome to distinguish exogenous and "
        #     "endogenous spacers..[/bold green]"
        # )
        # ref_genome = (
        #     "/mnt/c/aws_data/20240701_yuanyuan_recording/selected_plasmids/agg.fna"
        # )
        # makedirs(
        #     *[
        #         f"{output_dir}/collect_spacers/{i}"
        #         for i in ["map_plasmid", "mapped_plasmid", "unmapped_plasmid"]
        #     ]
        # )
        # map_se(
        #     f"{output_dir}/collect_spacers/other/{sample}.fq.gz",
        #     ref_genome,
        #     bam_file=f"{output_dir}/collect_spacers/map_plasmid/{sample}.bam",
        #     # bwa_log=f"{output_dir}/collect_spacers/log/{sample}.bwa.log",
        #     mapped_reads=f"{output_dir}/collect_spacers/mapped_plasmid/{sample}.fq.gz",
        #     unmapped_reads=f"{output_dir}/collect_spacers/unmapped_plasmid/{sample}.fq.gz",
        #     args="relaxed",
        # )
