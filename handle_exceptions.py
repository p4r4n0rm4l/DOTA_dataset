CLASSES = ["small-vehicle", "large-vehicle", "ship", "plane"]
a = 10
try:
    x = CLASSES.index('small-=veasdf')
    a = 123
except ValueError:
    print("Oops!  That was no valid number.  Try again...")

print(a)