from argparse import Namespace
from typing import Generator, Tuple

from torchsr.esrgan.generator import Generator as ESRGANGen
from torchsr.esrgan.trainer import ESRGANTrainer
from torchsr.srgan.generator import Generator as SRGANGen
from torchsr.srgan.trainer import SRGANTrainer


MODELS = {
    'esrgan': ESRGANTrainer,
    'srgan': SRGANTrainer
}

CROP_SIZE = {
    'esrgan': 128,
    'srgan': 96
}

GENERATORS = {
    'esrgan': ESRGANGen,
    'srgan': SRGANGen
}


def select_trainer_model(args: Namespace) -> Tuple[object, int]:
    """
    Return the trainer class for the requested model.

    Check that the requested model is supported by the application and return
    the associated trainer class plus crop size.

    Parameters
    ----------
    args : Namespace
        A ``Namespace`` of all the arguments passed via the CLI.

    Returns
    -------
    tuple
        Returns a ``tuple`` of the model's class declaration and the crop size
        to be used for the specified model, respectively.

    Raises
    ------
    RuntimeError
        Raises a ``RuntimeError`` if the requested model is not supported.
    """
    if args.model.lower() in MODELS:
        return MODELS[args.model.lower()], CROP_SIZE[args.model.lower()]
    else:
        raise RuntimeError(f'{args.model} not supported. Please choose from: '
                           f'{MODELS.keys()}')


def select_test_model(args: Namespace) -> object:
    """
    Return the appropriate Generator for the requested model.

    Check that the requested model is supported by the application and return
    the associated Generator to be used for testing.

    Parameters
    ----------
    args : Namespace
        A ``Namespace`` of all the arguments passed via the CLI.

    Returns
    -------
    object
        Returns the Generator class declaration for the specified model.

    Raises
    ------
    RuntimeError
        Raises a ``RuntimeError`` if the requested model is not supported.
    """
    if args.model.lower() in GENERATORS:
        return GENERATORS[args.model.lower()]
    else:
        raise RuntimeError(f'{args.model} not supported. Please choose from: '
                           f'{GENERATORS.keys()}')
