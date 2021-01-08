class Robot:
    def __init__(self, name):
        self.name = name

    def say_hi(self):
        print(f"Hi, I am {self.name}")



class PhysicianRobot(Robot):
    def say_hi(self):
        print("Everything will be ok!")
        print(self.name + " takes care of you!")

y = PhysicianRobot("Doc James")
y.say_hi()
print("and now the traditional way of saying hi")
Robot.say_hi(y)
