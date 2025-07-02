# -*- coding: utf-8 -*-
"""
Anonymize anything (medical)
Authors: Moritz Rempe & Lukas Heine
"""
import argparse
from mede.dicom_skullstrip_defacing import Inference
from mede.dicom_deidentification import DicomDeidentifier
from mede.text_detection import TextRemoval
from mede.wsi_deidentification import WSIDeidentifier
from mede.twix_deidentification import anonymize_twix
import torch
import logging

logging.basicConfig(level=logging.INFO)

def main():
    """Main function to run the deidentification process."""
    parser = argparse.ArgumentParser(
        prog="deidentify.py", epilog="If you find this useful, please cite our work."
    )
    parser.add_argument(
        "-v",
        "--verbose",
        required=False,
        default=True,
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "-t",
        "--text-removal",
        required=False,
        default=False,
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "-i",
        "--input",
        required=False,
        default=None,
        help="Path to the input data.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=False,
        default=None,
        help="Path to save the output data.",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU device number. (default 0)",
    )
    parser.add_argument(
        "-s",
        "--skull_strip",
        required=False,
        default=False,
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "-de",
        "--deface",
        required=False,
        default=False,
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "-tw",
        "--twix",
        required=False,
        default=False,
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "-w",
        "--wsi",
        required=False,
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Anonymize WSI label and overview file.",
    )
    parser.add_argument(
        "-p",
        "--processes",
        required=False,
        default=1,
        type=int,
        help="Number of processes to use for multiprocessing.",
    )
    parser.add_argument(
        "-d",
        "--deidentification-profile",
        choices=[
            "basicProfile",
            "cleanDescOpt",
            "cleanGraphOpt",
            "cleanStructContOpt",
            "rtnDevIdOpt",
            "rtnInstIdOpt",
            "rtnLongFullDatesOpt",
            "rtnLongModifDatesOpt",
            "rtnPatCharsOpt",
            "rtnSafePrivOpt",
            "rtnUIDsOpt",
        ],
        required=False,
        nargs="+",
        default=None,
        help="Which DICOM deidentification profile(s) to apply. (default None)",
    )

    parser.parse_args()
    args = parser.parse_args()
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.autograd.set_detect_anomaly(True)
    torch.set_num_threads(args.processes)

    _input = args.input
    _out = args.output

    if args.deidentification_profile is not None:
        dicom_deidentifier = DicomDeidentifier(
            args.deidentification_profile,
            processes=args.processes,
            out_path=args.output,
            verbose=args.verbose,
        )
        dicom_deidentifier(_input)
        _input = _out
    else:
        logging.info(
            "No DICOM deidentification profile specified. No Metadata anonymization will be performed!"
        )

    if args.wsi:
        wsi_deidentifier = WSIDeidentifier(verbose=args.verbose, out_path=args.output)
        wsi_deidentifier(_input)
        _input = _out

    if args.skull_strip:
        skull_strip = Inference(
            output_path=args.output, gpu=args.gpu, skullstrip=True, verbose=args.verbose
        )
        skull_strip(_input)
        _input = _out

    if args.deface:
        deface = Inference(
            output_path=args.output, gpu=args.gpu, deface=True, verbose=args.verbose
        )
        deface(_input)
        _input = _out

    if args.twix:
        anonymize_twix(_input, args.output)
        _input = _out

    if args.text_removal:
        txt_removal = TextRemoval(output_path=args.output, verbose=args.verbose)
        txt_removal(_input)
        
if __name__ == "__main__":
    main()