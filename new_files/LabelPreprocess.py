import numpy as np

if __name__ == "__main__":
    f = open("D:\phdmethods\MoDi-main\data\Labels-pre-processing.txt", "r")
    lines = f.readlines()

    for x in range(len(lines)):
        #remove jasper
        lines[x] = lines[x][7:]
        #take most relevent section
        lines[x] = lines[x].split('/')[0]
        #removing () sections
        if lines[x][(len(lines[x])-1)] == ')':
            lines[x] = lines[x][:len(lines[x]) - 4]
    lines = np.unique(lines)
    results = open("D:\phdmethods\MoDi-main\data\Labels-ProcessedUnique.txt","x")
    for x in range(len(lines)):
        results.write(lines[x] + '\n')