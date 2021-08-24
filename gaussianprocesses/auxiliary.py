#! /usr/bin/env python3

import sympy as sy
from sympy.parsing import sympy_parser
from sympy.utilities.lambdify import lambdify
import kernels



if __name__ == '__main__':
    k1 = kernels.Linear([1,1])
    k2 = kernels.SquaredExponential([1,1])
    k3 = kernels.AnisotropicSquaredExponential([1,1])
    kk = k1 + k2
    kkk = kk * k3

    print(str(kkk))
    print('-'*20)
    k_str = ''.join(str(kkk).split(' '))
    k_sym = sympy_parser.parse_expr(k_str)
    k_names = [str(n)[:-6] for n in list(k_sym.free_symbols)]
    print(k_sym)
    print(k_names)
    k_obj = [getattr(kernels, n)([1,1]) for n in k_names]
    print(k_obj)

    test = lambdify(sy.Symbol(' '.join(k_names)), k_sym, 'numpy')
    bla = test(k_obj)
    print(str(bla))
    print(repr(bla))
    print(isinstance(bla, kernels.Combination))
    
    def test_func(x, y, z):
        return (x + y) * z

    bla_2 = test_func(*k_obj)
    print(str(bla_2))
    print(isinstance(bla_2, kernels.Combination))
    bla_2.parameters = [1, 1, 1, 1, 1, 1]
    

    print('-'*20)
    k_objects = {}
    for arg in sy.postorder_traversal(k_sym):
        print(arg)
        print(str(arg.func))
        if str(arg.func).strip("<>'").split('.')[-1] == 'Symbol':
            print('Variable')
            k_objects[str(arg)] = getattr(kernels, str(arg)[:-6])()
        elif str(arg.func).strip("<>'").split('.')[-1] == 'Add':
            print('Addition')
            print(arg.args)
            k_add = 0 
            for a in arg.args:
                if str(a) in k_objects:
                    k_add = k_add + k_objects[str(a)]
                else:
                    k_add = k_add + getattr(kernels, str(a)[:-6])()
            print(k_add)
            k_objects[str(arg)] = k_add 
        elif str(arg.func).strip("<>'").split('.')[-1] == 'Mul':
            print('Multiplication')
            print(arg.args)
            k_mul = 1
            for a in arg.args:
                if str(a) in k_objects:
                    k_mul = k_mul * k_objects[str(a)]
                else:
                    k_mul = k_mul * getattr(kernels, str(a)[:-6])()
            k_objects[str(arg)] = k_mul
        else:
            print('WARNING: unknown operation')
    print('-'*20)
    print(k_objects)
    print('-'*20)
    final_kernel = k_objects[str(k_sym)]
    print(str(final_kernel))
    print('Is final kernel a kernel?', isinstance(final_kernel, kernels.Combination))
