import maia.pytree.maia as MT
import maia.pytree      as PT
from   maia.typing import *

__all__ = ['is_cgns_part_tree', 'is_cgns_dist_tree', 'is_cgns_full_tree',
           'check_cgns_dist_tree', 'check_cgns_part_tree', 'check_cgns_full_tree',
           'check_cgns_dist_part_tree']
    
def is_cgns_part_tree(tree: CGNSTree) -> bool:
    """Determine if the CGNS tree represents a partitioned tree.
    Args:
        tree (CGNSTree): The input CGNS tree.
    Returns:
        bool: True if the tree is a partitioned tree, False otherwise.
    """
    return all(MT.get_GlobalNumbering(zone) is not None for zone in PT.iter_all_Zone_t(tree))

def is_cgns_dist_tree(tree: CGNSTree) -> bool:
    """Determine if the CGNS tree represents a distributed tree.
    Args:
        tree (CGNSTree): The input CGNS tree.
    Returns:
        bool: True if the tree is a distributed tree, False otherwise.
    """
    return all(MT.get_Distribution(zone) is not None for zone in PT.iter_all_Zone_t(tree))

def is_cgns_full_tree(tree: CGNSTree) -> bool:
    """Determine if the CGNS tree is a full CGNS tree.
    Args:
        tree (CGNSTree): The input CGNS tree.
    Returns:
        bool: True if the tree is neither a partitioned nor a distributed tree.
    """
    return all(MT.get_Distribution(zone) is None and  MT.get_GlobalNumbering(zone) is None \
              for zone in PT.iter_all_Zone_t(tree))

def check_contain_zones(tree: CGNSTree) -> None:
    """Check if the given CGNS tree contains zones.
    Args:
        tree (CGNSTree): The input CGNS tree.
    Returns:
        List[CGNSTree]: A list of zones if found, otherwise an empty list.
    """
    if len(PT.get_all_Zone_t(tree)) == 0:
        raise ValueError("Invalid tree structure: missing Zone_t elements.")
    
def check_cgns_dist_tree(tree: CGNSTree) -> None:
    """Raise an error if the tree is not a distributed CGNS tree.
    Args:
        tree (CGNSTree): The input CGNS tree.
    Raises:
        ValueError: If the tree is not a distributed tree.
    """
    if not is_cgns_dist_tree(tree):
        raise ValueError("The provided CGNS tree is not a distributed tree.")

def check_cgns_part_tree(tree: CGNSTree) -> None:
    """Raise an error if the tree is not a partitioned CGNS tree.
    Args:
        tree (CGNSTree): The input CGNS tree.
    Raises:
        ValueError: If the tree is not a partitioned tree.
    """
    if not is_cgns_part_tree(tree):
        raise ValueError("The provided CGNS tree is not a partitioned tree.")
    

def check_cgns_full_tree(tree: CGNSTree) -> None:
    """Raise an error if the tree is not a full CGNS tree.
    Args:
        tree (CGNSTree): The input CGNS tree.
    Raises:
        ValueError: If the tree is not a full tree.
    """
    if not is_cgns_full_tree(tree):
        raise ValueError("The provided CGNS tree is not a full CGNS tree.")  
    
def check_cgns_dist_part_tree(tree: CGNSTree) -> None:
    """Raise an error if the tree is neither a distributed nor partitionned CGNS tree.
    Args:
        tree (CGNSTree): The input CGNS tree.
    Raises:
        ValueError: If the tree is neither a distributed nor partitionned CGNS tree.
    """
    if not (is_cgns_dist_tree(tree) or is_cgns_part_tree(tree)):
        raise ValueError("The provided CGNS tree is neither a distributed nor partitionned CGNS tree.") 
    