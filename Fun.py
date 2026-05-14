# Define some global variables
x = 10
message = "Hello, world!"


# Define a global function
def greet():
    print(message)


if __name__ == "__main__":
    # Print the type of globals()
    print("Type of globals():", type(globals()))

    # Print the globals dictionary keys (names of global variables/functions)
    print("Globals keys:", list(globals().keys()))

    # Call the global function using globals()
    globals()['greet']()
