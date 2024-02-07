from __future__ import annotations


def wrap_kwargs(fn: callable) -> callable:
    """Wrap a function to accept keyword arguments from the command line."""
    from inspect import signature

    import click

    for param in signature(fn).parameters:
        fn = click.option("--" + param, type=str)(fn)
    return click.command()(fn)
