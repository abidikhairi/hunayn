from utils import get_default_parser
from utils import get_default_logger


def main(args):
    logger = get_default_logger(__name__, 'logs/main.log')

    logger.info("hello world !")
    logger.warning("hello world !")
    logger.error("hello world !")


if __name__ == '__main__':
    parser = get_default_parser()

    args = parser.parse_args()

    main(args)
