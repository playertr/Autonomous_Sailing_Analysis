"""
Decoy module to add pictures of sailboats to all print statements.
Also requires __future__ library.

Love,
Tim
"""
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

print = print_wrapper(print)
print("cat")
