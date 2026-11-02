"""Williams crossover order: balance positions and adjacent backend pairs."""


def order_cycle_length(backends):
    n=len(backends)
    if not n or len(set(backends))!=n:
        raise ValueError('distinct, nonempty backend names required')
    return n if n==1 or n%2==0 else 2*n


def balanced_order(block,case_index,backends):
    """Return one row; each input shape needs a complete cycle of blocks.

    Odd treatment counts need the reversed square as well. A cyclic rotation
    alone leaves most backend pairs in the same relative execution order.
    """
    if block<0 or case_index<0:
        raise ValueError('nonnegative block and case index required')
    cycle=order_cycle_length(backends)
    n=len(backends)
    row=(block+case_index)%cycle
    base=[0]+[(i+1)//2 if i%2 else n-i//2 for i in range(1,n)]
    order=[backends[(value+row%n)%n] for value in base]
    if n%2 and row>=n:order.reverse()
    return tuple(order)
