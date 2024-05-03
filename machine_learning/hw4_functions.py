# default parameter - the one and only name parameter is set to a default string, "Guest"

def greet(name="Guest"):
    return f"Hello, {name}!"
    
# Calling the function without an argument
message =  greet()
print(message)

# multiple parameters - returns sum

def add(a, b):
    return a + b
    
# Calling the function
result = add(5, 3)
print(result)

# multiple returns - takes a list of numbers and returns both the minimum and maximum values

def find_min_max(numbers):
    if not numbers:
        return None, None
    return min(numbers), max(numbers)
    
# Calling the function
nums = [4, 2, 9, 1, 7]
min_value, max_value = find_min_max(nums)
print(f"Min: {min_value}, Max: {max_value}")

#recursive function calculates the factorial of a number.
# for 5, returns 120 because 5! = 5 * 4 * 3 * 2 * 1

def factorial(n):
    if n <= 1:
        return 1
    else:
        return n * factorial(n-1)
        
# Calling the function
result = factorial(5)
print(result)

# basic calculator program - perform basic arithmetic operations on two numbers.

def add(x, y):
    return x + y
    
def subtract(x, y):
    return x - y
    
def multiply(x, y):
    return x * y
    
def divide(x, y):
    if y == 0:
        return "Cannot divide by zero"
    return x / y
    
print("Select operation:")
print("1. Addition")
print("2. Subtraction")
print("3. Multiplication")
print("4. Division")

choice = input ("\n\nDo you want to add, subtract, multiply or divide (select choice): ")

num1 = float(input("Enter first number: "))
num2 = float(input("Enter second number: "))

print("\n")

if choice == '1':
    result = add(num1, num2)
    print(f"{num1} + {num2} = {result}")
elif choice == '2':
    result = subtract(num1, num2)
    print(f"{num1} - {num2} = {result}")
elif choice == '3':
    result = multiply(num1, num2)
    print(f"{num} * {num2} = {result}")
elif choice == '4':
    result = divide(num1, num2)
    print(f"{num1} / {num2} = {result}")
else:
    print("Invalid input")

