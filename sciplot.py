"""
Decoy library to override printing functions with pictures of sailboats.

Love,
Tim
"""

from __future__ import print_function
# This must be the first statement before other statements.
# You may only put a quoted or triple quoted string, 
# Python comments, other future statements, or blank lines before the __future__ line.
count = 0
def print_wrapper(func):
    """My custom print() function."""
    # Adding new arguments to the print function signature 
    # is probably a bad idea.
    # Instead consider testing if custom argument keywords
    # are present in kwargs

    def wrapper(*args, **kwargs):
        # secret_message = "boat"
        secret_message = '''
x                  ~.                       
x           Ya...___|__..aab     .   .      
x            Y88a  Y88o  Y88a   (     )     
x             Y88b  Y88b  Y88b   `.oo'      
x             :888  :888  :888  ( (`-'      
x    .---.    d88P  d88P  d88P   `.`.       
x   / .-._)  d8P'"""|"""'-Y8P      `.`.     
x  ( (`._) .-.  .-. |.-.  .-.  .-.   ) )    
x   \ `---( O )( O )( O )( O )( O )-' /     
x    `.    `-'  `-'  `-'  `-'  `-'  .' CJ   
x~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  '''
        freq = 30
        mesg = ''.join([' ' * (count // freq) if c == 'x' else c for c in secret_message])

        global count
        count += 1
        if count == 100 * freq:
            count = 0

        func(mesg)
        return func(*args, **kwargs)
    return wrapper

