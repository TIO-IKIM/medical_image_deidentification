# -*- coding: utf-8 -*-
"""
De-Identification of Medical Imaging Data: A Comprehensive Tool for Ensuring Patient Privacy

Authors: Moritz Rempe & Lukas Heine
"""
import argparse
import logging

logging.basicConfig(level=logging.INFO)


def _configure_torch(processes: int) -> None:
    """Configure PyTorch only when a torch-backed feature is requested."""
    import torch

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.autograd.set_detect_anomaly(True)
    torch.set_num_threads(processes)


def main():
    """Main function to run the deidentification process."""
    parser = argparse.ArgumentParser(
        prog="deidentify.py", epilog="If you find this useful, please cite our work."
    )
    parser.add_argument(
        "-v",
        "--verbose",
        required=False,
        default=False,
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
        "--refine",
        required=False,
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Enable interactive refinement of text removal results.",
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
        "-r",
        "--rename",
        required=False,
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Rename files to UUIDs, preserving slice order for series.",
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

    args = parser.parse_args()

    if args.skull_strip or args.deface or args.text_removal:
        _configure_torch(args.processes)

    _input = args.input
    _out = args.output

    if args.deidentification_profile is not None:
        from mede.dicom_deidentification import DicomDeidentifier

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
        from mede.wsi_deidentification import WSIDeidentifier

        wsi_deidentifier = WSIDeidentifier(verbose=args.verbose, out_path=args.output)
        wsi_deidentifier(_input)
        _input = _out

    if args.skull_strip or args.deface:
        from mede.dicom_skullstrip_defacing import Inference

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
        from mede.twix_deidentification import anonymize_twix

        anonymize_twix(_input, args.output)
        _input = _out

    if args.text_removal:
        from mede.text_detection import TextRemoval

        txt_removal = TextRemoval(output_path=args.output, verbose=args.verbose, interactive=args.refine)
        txt_removal(_input)
        _input = _out

    if args.rename:
        from mede.rename import Rename

        rename = Rename(input_path=args.input, output_path=args.output)
        rename()

if __name__ == "__main__":
    main()
