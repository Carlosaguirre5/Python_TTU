## Exercise 1: Python Basics

# Sample solution
a = 10
b = 5

# Operations
print("Addition:", a + b)
print("Substraction:", a - b)
print("Multiplication", a * b)
print("Division:", a / b)
# Square function
def square(x):
        return x * x
#print("Square of 4:", square(4))

## Exercise 2: Lits and Loops
# Sample solutions 
numbers = [1, 2, 3, 4, 5]
for number in numbers:
    print("Square:", square(number))

## Exercise 3: Strings - Lenght and Uppercase

name = "Carlos"
print("Lenght of name:", len(name))
print("Name in uppercase:", name.upper())

## Exercise 4: Functions - Adding Two Numbers

def add_numbers(a, b):
      return a + b

#Calling the function
result = add_numbers(3, 5)
print("Sum:", result)


## Exercise 5: Dictionaries - Personal information

person = {"name": "Carlos", "age": 28, "city": "New York"}

print("name:", person["name"])
print("age:", person["age"])
print("city:", person["city"])

## Exercise 6: Conditional statements - Check Even or Odd

num = int(input("Enter a number: "))
if (num % 2) == 0:
    print("{0} is Even".format(num))
else:  
    print("{0} is odd".format(num))

## Exercise 7: File handling - Write and Read from file

#writing to a file
with open("example.txt", "w") as file:
    file.write("Hello, this is a simple file!")

# Reading from the file
with open("example.txt", "r") as file:
    content = file.read()
    print("File content:", content)

# Exercise 8: Basic Calculator

def calculator( a , b):
    return a * b

#Call the calc function

print("Sum of 10 and 20:", calculator(10, 20))

## Exercise 9: Simple Loop - Print Numbers

for i in range(1, 6):
    print(i)

## Exercise 10# Working with a list

fruits = ["Apple", "Bananas", "Oranges"]
print("First fruit:", fruits[0])
print("Last fruit:", fruits[-1])