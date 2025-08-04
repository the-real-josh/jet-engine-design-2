import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# gas constants for air
class Gas:
    """Perfect gas"""
    def __init__(self,
                 gamma, 
                    R) -> None:
        self.gamma = gamma
        self.cp = (gamma*R/(gamma-1))
        self.cv = (self.cp-R)
# TODO: represent the air and combustion product mixture as a Gas() object
air = Gas(1.40, 287.05)

def main():
    pass

if __name__ == "__main__":
    main()