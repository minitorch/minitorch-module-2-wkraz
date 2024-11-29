from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple

from typing_extensions import Protocol, runtime_checkable

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    
    vals_plus_epsilon = list(vals)
    vals_minus_epsilon = list(vals)
    
    # change the arg'th value by order of epsilon
    vals_plus_epsilon[arg] += epsilon
    vals_minus_epsilon[arg] -= epsilon
    
    return (f(*vals_plus_epsilon) - f(*vals_minus_epsilon)) / (2 * epsilon) # central difference formula
    


variable_count = 1


@runtime_checkable
class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    visited = set()
    sorted_variables = []
    
    def dfs(var):
        if var.unique_id in visited or var.is_constant():
            return
        visited.add(var.unique_id)
        for parent in var.parents:
            dfs(parent)            # traverse the parents of the current node
        sorted_variables.append(var) # now we can add it to the sorted list
        
    dfs(variable)                    # do dfs on rightmost variable
    return reversed(sorted_variables) # reverse to get topological order


def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    derivatives = {variable.unique_id: deriv}     # dictionary to store derivative for each variable
    
    # process variables in topological order
    for var in topological_sort(variable):
        if var.is_leaf():
            var.accumulate_derivative(derivatives[var.unique_id])   # accumulate derivatives for leaf nodes
        else:
            # if not leaf node, propogate derivatives to parents with chain rule
            for parent, partial_deriv in var.chain_rule(derivatives[var.unique_id]):
                # if we don't know its derivative, add it to dict and set it to 0
                if parent.unique_id not in derivatives:
                    derivatives[parent.unique_id] = 0
                derivatives[parent.unique_id] += partial_deriv 


@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
