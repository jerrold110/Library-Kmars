# Python program to
# demonstrate private members
 
# Creating a Base class
class Base:
    def __init__(self):
        self.a = "GeeksforGeeksa"
        self.__c = "GeeksforGeeksc"
 
# Creating a derived class
class Derived(Base):
    def __init__(self):
 
        # Calling constructor of
        # Base class
        Base.__init__(self)
        print("Calling private member of base class: ")
        print(self.a)
        print(self.__c)
 
 
# Driver code
Derived()
