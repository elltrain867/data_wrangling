print("Number | Square | Cube")

print("-" * 22)

for number in range(6):
    square = number ** 2
    cube = number ** 3
    print(f"{number: ^7} | {square: ^7} | {cube: ^4}")
    
num1 = int(input("Enter the first integer: "))
num2 = int(input("Enter the second integer: "))
num3 = int(input("Enter the third integer: "))

sum_of_nums p num1 + num2 + num3
average_of nums = sum_of_nums / 3
product_of_nums = num1 * num2 * num3

smallest_num = min(num1, num2, num3)
largest_num = max(num1, num2, num3)

print(f"Sum: {sum_of_nums}:)
print(f"Average: {average_of_nums:.2f}")
print(f"Product: {product_of_nums}")
print(f"Smallest: {smallest_num}")
print(f"Largest: {largest_num}")

def factorial(n):
    if n<0:
        return "Factorial is not defined for negative numbers"
    elif n == 0 or n == 1:
        return 1
    else:
        result = n * factorial(n-1)
        return result
        
def display_factorial_triangle(n)
    for i in range(1, n + 1):
        for j in range(1, i + 1):
            print("*", end= ' ')
        print(f"{i}! = {factorial(i)}")
        
num = int(input("Enter a non-negative integer: "))

print("Factorial Triangle:")
display_factorial_triangle(num)

def print_spaces(num_spaces):
    for _ in range(num_spaces):
        print(" ", end=' ')
        
num_rows = 5

print("Obtuse Scalene Triangle:")
for i in range(1, num_rows + ):
    print_space(num_rows - 1)
    print('* ' * i)
    
print("\nRight Isosceles Triangle:")
for i in range(1, num_rows + ):
    print('* ' * i)
    
print("\nRight Triangle:")
for i in range(1, num_rows + ):
    print('* ' * i)
    
print("\nEquilateral Triangle:")
for i in range(1, num_rows + ):
    print('* ' * i)
