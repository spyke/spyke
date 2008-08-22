from scipy import weave

def weave_pure_inc(N):
    code = r"""
           int i=0;
           while (i<N)
               i+=1;
           return_val = i;
           """
    return weave.inline(code, ['N'], compiler='gcc')



def weave_setmat(a, val):
    code = r"""
            int i, j;
            int nrows = Na[0];
            int ncols = Na[1];
            for (i=0; i<nrows; i++)
                for (j=0; j<ncols; j++)
                    a(i,j) = val;
           """
    return weave.inline(code, ['a', 'val'],
                        type_converters=weave.converters.blitz,
                        compiler='gcc')

def weave_pure_setmat(a, val):
    code = r"""
            int i, j;
            int nrows = Na[0];
            int ncols = Na[1];
            for (i=0; i<nrows; i++)
                for (j=0; j<ncols; j++)
                    a[i*ncols + j] = val;
           """
    return weave.inline(code, ['a', 'val'], compiler='gcc')


#timeit.Timer('weave_pure_setmat(a, 5)', 'from __main__ import weave_pure_setmat, a').timeit(100)
