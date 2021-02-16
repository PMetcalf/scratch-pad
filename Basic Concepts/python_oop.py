'''
Python OOP

An introduction to aspects of OOP as implemented in Python.
'''

'''
-- Classes --
'''

class Parrot:
    
    # Class attribute
    species = 'bird'

    # Instance attribute
    def __init__(self, name, age):
        self.name = name
        self.age = age

    # Instance methods
    def sing(self, song):
        return "{} sings {}".format(self.name, song)

    def dance(self):
        return "{} is now dancing".format(self.name)

# Instantiate the parrot class
blue = Parrot("Blue", 10)
Woo = Parrot("Woo", 15)

# Access class attributes
print("Blue is a {}".format(blue.__class__.species))
print("Woo is also a {}".format(Woo.__class__.species))

# Access the instance attributes
print("{} is {} years old".format( blue.name, blue.age))
print("{} is {} years old".format( Woo.name, Woo.age))

# Call instance methods
print(blue.sing("'Happy'"))
print(blue.dance())

'''
-- Inheritance --
'''

# Parent class
class Bird:

    def __init__(self):
        print("Bird is ready")

    def who_is_this(self):
        print("Bird")

    def swim(self):
        print("Swim faster")

# Child class
class Penguin(Bird):

    def __init__(self):
        # call super() function
        super().__init__()
        print("Penguin is ready")

    def who_is_this(self):
        print("Penguin")

    def run(self):
        print("Run faster")

peggy = Penguin()
peggy.who_is_this()
peggy.swim()
peggy.run()

'''
-- Encapsulation --
'''

class Computer:

    def __init__(self):
        # Private attribute (encapsulation)
        self.__maxprice = 900

    def sell(self):
        print("Selling price: {}".format(self.__maxprice))

    def set_max_price(self, price):
        self.__maxprice = price

computer = Computer()
computer.sell()

# Try to change the price directly
computer.__maxprice = 1000
computer.sell()

# Change the price using setter function
computer.set_max_price(1000)
computer.sell()

'''
Polymorphism
'''

class Robin:

    def fly(self):
        print("Robin can fly")

    def swim(self):
        print("Robin can't swim")

class Puffin:

    def fly(self):
        print("Puffin doesn't fly well")

    def swim(self):
        print("Puffin swims really well")

# Common interface
def flying_test(bird):
    bird.fly()

# Instantiate objects
robin = Robin()
puffin = Puffin()

# Use the interface in polymorphism
flying_test(robin)
flying_test(puffin)