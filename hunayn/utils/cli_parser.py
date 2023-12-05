import argparse


def get_default_parser() -> argparse.ArgumentParser:
    """creates a cli parser object

    Returns:
        argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--d_model', type=int, default=1024,
                        help='Dimension of the model (default: 1024)')
    parser.add_argument('--d_ff', type=int, default=512,
                        help='Dimension of the feedforward layer (default: 512)')
    parser.add_argument('--nhead', type=int, default=4,
                        help='Number of attention heads (default: 4)')
    parser.add_argument('--dropout', type=float, default=0.6,
                        help='Dropout probability (default: 0.6)')
    parser.add_argument('--num_encoder_layers', type=int,
                        default=4, help='Number of encoder layers (default: 4)')
    parser.add_argument('--num_decoder_layers', type=int,
                        default=4, help='Number of decoder layers (default: 4)')

    parser.add_argument('--batch_size', type=int, default=4,
                        help='Training batch size (default: 4)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers (default: 4)')

    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='Learning rate for optimization (default: 1e-3)')
    parser.add_argument('--beta1', type=float, default=0.9,
                        help='Beta1 parameter for Adam optimizer (default: 0.9)')
    parser.add_argument('--beta2', type=float, default=0.988,
                        help='Beta2 parameter for Adam optimizer (default: 0.988)')
    parser.add_argument('--warmup_steps', type=int, default=2000,
                        help='Number of warmup steps for learning rate scheduler (default: 2000)')

    parser.add_argument('--src_tokenizer_path', type=str,
                        required=True, help='Path to the source tokenizer file')
    parser.add_argument('--tgt_tokenizer_path', type=str,
                        required=True, help='Path to the target tokenizer file')

    parser.add_argument('--input_file', type=str,
                        required=True, help="Path to protein/function file")

    return parser
