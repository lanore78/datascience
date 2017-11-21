# import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
from numpyexample import numpy_examples as npe
from convolution import mnist as mn



def main():
    # csv_test = pd.read_csv('./dds_ch2_nyt/nyt1.csv')
    # print(csv_test)
    # python_numericType()
    # python_stringType()

    # example = npe.NumpyExample();
    # example.hello();
    run_mnist()

def run_mnist():
    mnistExample = mn.MnistExample()

    mnistExample.prepare_data()
    mnistExample.define_graph()
    mnistExample.start_train()



def python_stringType():
    s1 = 'test\n'
    s2 = '''
Hello
World\n'''
    s3 = 'Hello this is bgshim'

    s4 = s3[-4:]
    print(s4)

    print('{0}..{0}..{1}'.format(s4, s3))


    # print('%s, %s, %s' % (s1, s2, s3))


def python_numericType():
    a = 1
    b = 0x16
    c = 0.8
    e = 1 + 2j
    print(type(a))
    print(type(c))
    print(type(e))
    print(float(1))
    print(float.__doc__)


if __name__ == "__main__":
    main()