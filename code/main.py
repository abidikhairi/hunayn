from utils.cli_parser import get_default_parser
from utils.logging import get_default_logger


def main(args):
    logger = get_default_logger(__name__, 'logs/main.log')
    

if __name__ == '__main__':
    parser = get_default_parser()

    args = parser.parse_args()
    
    main(args)
