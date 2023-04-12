from argparse import ArgumentParser

from .neural_compressor import INCCommand


def main():
    parser = ArgumentParser("Optimum Intel CLI tool", usage="optimum-intel-cli <command> [<args>]")
    commands_parser = parser.add_subparsers(help="optimum-intel-cli command helpers")

    # Register commands
    INCCommand.register_subcommand(commands_parser)

    args = parser.parse_args()

    if not hasattr(args, "func"):
        parser.print_help()
        exit(1)

    # Run
    service = args.func(args)
    service.run()


if __name__ == "__main__":
    main()
