from scipy import weave
from scipy.weave import converters

def testweave():
    """Takes a time step using inlined C code -- this version uses
    blitz arrays."""
    code = """
           #line 8 "weave.py" (This is only useful for debugging)
           for (int i=0; i<10; ++i) {
               for (int j=0; j<10; ++j) {
                   printf("Hello world\\n");
               }
           }
           return_val = 666;
           """
    # compiler keyword only needed on windows with MSVC installed
    return = weave.inline(code,
                       [],
                       type_converters=converters.blitz,
                       compiler = 'msvc')


def mytest(n):
    code = r"""
           #line 25 "weave_example.py"
           int i=0;
           while (i<n)
               i++;
           return_val = i;
           """
    return weave.inline(code, ['n'],
                       type_converters=converters.blitz,
                       compiler='msvc')


if __name__ == '__main__':
    #testweave()
    print mytest(1000000000)


