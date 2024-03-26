import copy
class ClassA:
    def __init__(self):
        self.var1 = [1, 2, 3]
        self.var2 = "hello"

class ClassB:
    pass

objA = ClassA()
objB = ClassB()

# Copying all attributes from objA to objB
for attr, value in objA.__dict__.items():
    setattr(objB, attr, copy.copy(value))

print(objB.var1)  # Outputs: [1, 2, 3]
objA.var2 = "heollo"
print(objB.var2)  # Outputs: "hello"

