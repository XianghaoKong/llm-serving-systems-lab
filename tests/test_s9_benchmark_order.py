from collections import Counter
from pathlib import Path
import sys
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from s9_benchmark_order import balanced_order,order_cycle_length


@pytest.mark.parametrize('n',[2,3,4,5,6,7])
@pytest.mark.parametrize('case_index',[0,1,7])
def test_balances_positions_and_directed_neighbors(n,case_index):
    backends=tuple(range(n));cycle=order_cycle_length(backends)
    orders=[balanced_order(block,case_index,backends) for block in range(cycle)]
    for order in orders:assert set(order)==set(backends) and len(order)==n
    for position in range(n):
        assert set(Counter(row[position] for row in orders).values())=={cycle//n}
    adjacent=Counter((a,b) for row in orders for a,b in zip(row,row[1:]))
    assert len(adjacent)==n*(n-1)
    assert set(adjacent.values())=={cycle//n}
    # All relative pair directions, not just adjacent ones, must be balanced.
    for a in backends:
        for b in backends:
            if a!=b:assert sum(row.index(a)<row.index(b) for row in orders)==cycle//2


def test_invalid_names_and_indices():
    for backends in ((),('a','a')):
        with pytest.raises(ValueError):order_cycle_length(backends)
    with pytest.raises(ValueError):balanced_order(-1,0,('a','b'))
    assert balanced_order(3,7,('a',))==('a',)
