"""
Decoy library to override printing functions with pictures of sailboats.

Love,
Tim
"""

from __future__ import print_function
# This must be the first statement before other statements.
# You may only put a quoted or triple quoted string, 
# Python comments, other future statements, or blank lines before the __future__ line.

def print_wrapper(func):
    """My custom print() function."""
    # Adding new arguments to the print function signature 
    # is probably a bad idea.
    # Instead consider testing if custom argument keywords
    # are present in kwargs

    def wrapper(*args, **kwargs):
        # secret_message = "boat"
        secret_message = '''
                  ~.
           Ya...___|__..aab     .   .
            Y88a  Y88o  Y88a   (     )
             Y88b  Y88b  Y88b   `.oo'
             :888  :888  :888  ( (`-'
    .---.    d88P  d88P  d88P   `.`.
   / .-._)  d8P'"""|"""'-Y8P      `.`.
  ( (`._) .-.  .-. |.-.  .-.  .-.   ) )
   \ `---( O )( O )( O )( O )( O )-' /
    `.    `-'  `-'  `-'  `-'  `-'  .' CJ
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
        func(secret_message)
        return func(*args, **kwargs)
    return wrapper

