from multiprocessing import Process
from model import main
import utils as util


if __name__ == "__main__":
    multiple = 1
    process = []
    path = "."
    names = ['S' + str(i+1) for i in range(0, 16)]
    params = 1

    for param in params:
        for name in names:
            print(name)
            p = Process(target=main, args=(name, path, param))
            p.start()
            process.append(p)
            util.monitor(process, multiple, 60)
