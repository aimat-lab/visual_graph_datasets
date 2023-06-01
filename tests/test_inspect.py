"""
This test module is only for exploratory reasons to find out more about the python ``inspect`` module and
it's limits.
"""
import os
import inspect
import typing as t


def sum_elements(elements: t.List[t.Union[int, float]],
                 useless_callback: t.Callable = lambda: 'hello world',
                 ) -> float:
    """
    This method calculates the sum over the given elements but in a very bad manner.

    :param list elements: A list of elements which can be summed up

    :return: float
    """
    # We need to initialize this value which will later on hold the total sum
    value = 0
    for element in elements:
        # Here we add the current element to the sum value
        value += element

    return float(value)


def test_inspect_function():
    source_code = inspect.getsource(sum_elements)
    print(source_code)
    assert isinstance(source_code, str)

    source_lines = inspect.getsourcelines(sum_elements)[0]
    print(source_lines)
    assert isinstance(source_lines, list)

    doc = inspect.getdoc(sum_elements)
    print(doc)
    assert isinstance(doc, str)

    arg_spec = inspect.getfullargspec(sum_elements)
    print(arg_spec)


