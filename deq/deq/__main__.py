import arguably
from deq import cli


@arguably.command
def version() -> None:
    from deq import __version__
    print(__version__)


if __name__ == "__main__":
    cli.run()
