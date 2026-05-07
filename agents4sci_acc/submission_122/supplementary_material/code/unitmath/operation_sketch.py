"""
Operation Sketch Parser and Executor
Parses and executes symbolic operation sketches for table reasoning
"""

import re
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import json

from .unit_parser import Quantity, QuantityExtractor
from .symbolic_calculator import SymbolicCalculator, CalculationResult, OperationType, ConfidenceInterval


class SketchNodeType(Enum):
    """Types of nodes in operation sketch"""
    OPERATION = "operation"
    TABLE_REF = "table_ref"
    VALUE = "value"
    QUANTITY = "quantity"
    COLUMN = "column"
    ROW = "row"
    CELL = "cell"


@dataclass
class SketchNode:
    """Node in operation sketch tree"""
    node_type: SketchNodeType
    value: Any
    children: List['SketchNode'] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.children is None:
            self.children = []
        if self.metadata is None:
            self.metadata = {}
    
    def __repr__(self):
        if self.children:
            children_str = ", ".join(repr(c) for c in self.children)
            return f"{self.node_type.value}({self.value}, [{children_str}])"
        return f"{self.node_type.value}({self.value})"


class OperationSketch:
    """Represents a parsed operation sketch"""
    
    def __init__(self, sketch_text: str):
        self.sketch_text = sketch_text
        self.root = None
        self.parse()
    
    def parse(self):
        """Parse the sketch text into a tree structure"""
        self.root = self._parse_expression(self.sketch_text.strip())
    
    def _parse_expression(self, expr: str) -> SketchNode:
        """Parse a single expression recursively"""
        expr = expr.strip()
        
        # Check if it's a function call
        func_match = re.match(r'(\w+)\((.*)\)$', expr)
        if func_match:
            func_name = func_match.group(1)
            args_str = func_match.group(2)
            
            # Parse arguments
            args = self._parse_arguments(args_str)
            
            # Determine node type based on function name
            if func_name in ['add', 'subtract', 'multiply', 'divide', 
                            'compare', 'diff', 'ratio', 'percentage_change',
                            'fold_change', 'ci_overlap', 'mean', 'max', 'min']:
                node = SketchNode(
                    node_type=SketchNodeType.OPERATION,
                    value=func_name,
                    children=[self._parse_expression(arg) for arg in args]
                )
            elif func_name in ['col', 'column']:
                node = SketchNode(
                    node_type=SketchNodeType.COLUMN,
                    value=args[0] if args else ""
                )
            elif func_name in ['row']:
                node = SketchNode(
                    node_type=SketchNodeType.ROW,
                    value=args[0] if args else ""
                )
            elif func_name in ['cell']:
                # cell(row, col)
                node = SketchNode(
                    node_type=SketchNodeType.CELL,
                    value=(args[0], args[1]) if len(args) >= 2 else ("", "")
                )
            else:
                # Unknown function, treat as operation
                node = SketchNode(
                    node_type=SketchNodeType.OPERATION,
                    value=func_name,
                    children=[self._parse_expression(arg) for arg in args]
                )
            
            return node
        
        # Check if it's a table reference (e.g., "table.A.1")
        table_ref_match = re.match(r'table\.(\w+)\.(\w+)', expr)
        if table_ref_match:
            col = table_ref_match.group(1)
            row = table_ref_match.group(2)
            return SketchNode(
                node_type=SketchNodeType.TABLE_REF,
                value={'column': col, 'row': row}
            )
        
        # Check if it's a quantity (number with unit)
        quantity_match = re.match(r'([-+]?\d*\.?\d+)\s*([a-zA-Z%°μ]+)?', expr)
        if quantity_match and quantity_match.group(0) == expr:
            value = float(quantity_match.group(1))
            unit = quantity_match.group(2) if quantity_match.group(2) else ""
            return SketchNode(
                node_type=SketchNodeType.QUANTITY,
                value=Quantity(value=value, unit=unit, original_text=expr)
            )
        
        # Otherwise, treat as a value
        return SketchNode(
            node_type=SketchNodeType.VALUE,
            value=expr
        )
    
    def _parse_arguments(self, args_str: str) -> List[str]:
        """Parse comma-separated arguments, handling nested parentheses"""
        if not args_str.strip():
            return []
        
        args = []
        current_arg = ""
        paren_depth = 0
        
        for char in args_str:
            if char == ',' and paren_depth == 0:
                args.append(current_arg.strip())
                current_arg = ""
            else:
                if char == '(':
                    paren_depth += 1
                elif char == ')':
                    paren_depth -= 1
                current_arg += char
        
        if current_arg.strip():
            args.append(current_arg.strip())
        
        return args
    
    def to_dict(self) -> Dict:
        """Convert sketch to dictionary representation"""
        return self._node_to_dict(self.root)
    
    def _node_to_dict(self, node: SketchNode) -> Dict:
        """Convert a node and its children to dictionary"""
        result = {
            'type': node.node_type.value,
            'value': str(node.value) if not isinstance(node.value, dict) else node.value
        }
        
        if node.children:
            result['children'] = [self._node_to_dict(child) for child in node.children]
        
        if node.metadata:
            result['metadata'] = node.metadata
        
        return result


class SketchExecutor:
    """Executes operation sketches on table data"""
    
    def __init__(self):
        self.calculator = SymbolicCalculator()
        self.quantity_extractor = QuantityExtractor()
        
    def execute(self, sketch: OperationSketch, 
                table_data: Dict[str, Any],
                context: Dict[str, Any] = None) -> CalculationResult:
        """Execute an operation sketch on table data"""
        if context is None:
            context = {}
        
        return self._execute_node(sketch.root, table_data, context)
    
    def _execute_node(self, node: SketchNode,
                      table_data: Dict[str, Any],
                      context: Dict[str, Any]) -> Union[CalculationResult, Quantity, Any]:
        """Execute a single node in the sketch tree"""
        
        if node.node_type == SketchNodeType.OPERATION:
            return self._execute_operation(node, table_data, context)
        
        elif node.node_type == SketchNodeType.TABLE_REF:
            return self._resolve_table_ref(node.value, table_data)
        
        elif node.node_type == SketchNodeType.COLUMN:
            return self._resolve_column(node.value, table_data)
        
        elif node.node_type == SketchNodeType.ROW:
            return self._resolve_row(node.value, table_data)
        
        elif node.node_type == SketchNodeType.CELL:
            row, col = node.value
            return self._resolve_cell(row, col, table_data)
        
        elif node.node_type == SketchNodeType.QUANTITY:
            return node.value
        
        elif node.node_type == SketchNodeType.VALUE:
            # Try to parse as quantity
            quantities = self.quantity_extractor.parser.parse_text(str(node.value))
            if quantities:
                return quantities[0]
            return node.value
        
        else:
            raise ValueError(f"Unknown node type: {node.node_type}")
    
    def _execute_operation(self, node: SketchNode,
                          table_data: Dict[str, Any],
                          context: Dict[str, Any]) -> CalculationResult:
        """Execute an operation node"""
        operation = node.value
        
        # Execute children to get operands
        operands = []
        for child in node.children:
            result = self._execute_node(child, table_data, context)
            operands.append(result)
        
        # Map operation names to calculator methods
        if operation in ['add', '+']:
            if len(operands) >= 2:
                return self.calculator.add(operands[0], operands[1])
        
        elif operation in ['subtract', 'diff', '-']:
            if len(operands) >= 2:
                return self.calculator.subtract(operands[0], operands[1])
        
        elif operation in ['multiply', '*']:
            if len(operands) >= 2:
                return self.calculator.multiply(operands[0], operands[1])
        
        elif operation in ['divide', '/']:
            if len(operands) >= 2:
                return self.calculator.divide(operands[0], operands[1])
        
        elif operation == 'percentage_change':
            if len(operands) >= 2:
                return self.calculator.percentage_change(operands[0], operands[1])
        
        elif operation == 'fold_change':
            if len(operands) >= 2:
                return self.calculator.fold_change(operands[0], operands[1])
        
        elif operation == 'compare':
            if len(operands) >= 2:
                comparison_type = operands[2] if len(operands) > 2 else "eq"
                return self.calculator.compare(operands[0], operands[1], comparison_type)
        
        elif operation in ['mean', 'average']:
            if isinstance(operands[0], list):
                return self.calculator.aggregate(operands[0], "mean")
            else:
                return self.calculator.aggregate(operands, "mean")
        
        elif operation == 'max':
            if isinstance(operands[0], list):
                return self.calculator.aggregate(operands[0], "max")
            else:
                return self.calculator.aggregate(operands, "max")
        
        elif operation == 'min':
            if isinstance(operands[0], list):
                return self.calculator.aggregate(operands[0], "min")
            else:
                return self.calculator.aggregate(operands, "min")
        
        elif operation == 'ci_overlap':
            # Convert operands to ConfidenceInterval objects
            if len(operands) >= 2:
                ci1 = self._to_confidence_interval(operands[0])
                ci2 = self._to_confidence_interval(operands[1])
                return self.calculator.ci_overlap(ci1, ci2)
        
        else:
            raise ValueError(f"Unknown operation: {operation}")
        
        # Default error result
        return CalculationResult(
            value=0,
            unit="",
            operation=OperationType.ADD,
            inputs=[],
            confidence=0,
            error_message=f"Failed to execute operation: {operation}"
        )
    
    def _resolve_table_ref(self, ref: Dict[str, str], table_data: Dict[str, Any]) -> Quantity:
        """Resolve a table reference to a quantity"""
        col = ref['column']
        row = ref['row']
        return self._resolve_cell(row, col, table_data)
    
    def _resolve_column(self, col_name: str, table_data: Dict[str, Any]) -> List[Quantity]:
        """Resolve a column reference to list of quantities"""
        quantities = []
        
        # Find column index
        headers = table_data.get('headers', [])
        col_idx = None
        for i, header in enumerate(headers):
            if str(header).lower() == str(col_name).lower():
                col_idx = i
                break
        
        if col_idx is None:
            # Try to parse as index
            try:
                col_idx = int(col_name)
            except:
                return quantities
        
        # Extract quantities from column
        data = table_data.get('data', [])
        for row in data:
            if col_idx < len(row):
                cell_text = str(row[col_idx])
                cell_quantities = self.quantity_extractor.parser.parse_text(cell_text)
                if cell_quantities:
                    quantities.append(cell_quantities[0])
        
        return quantities
    
    def _resolve_row(self, row_name: str, table_data: Dict[str, Any]) -> List[Quantity]:
        """Resolve a row reference to list of quantities"""
        quantities = []
        
        # Try to parse as index
        try:
            row_idx = int(row_name)
        except:
            # Try to find row by first column value
            data = table_data.get('data', [])
            row_idx = None
            for i, row in enumerate(data):
                if row and str(row[0]).lower() == str(row_name).lower():
                    row_idx = i
                    break
            
            if row_idx is None:
                return quantities
        
        # Extract quantities from row
        data = table_data.get('data', [])
        if row_idx < len(data):
            row = data[row_idx]
            for cell in row:
                cell_text = str(cell)
                cell_quantities = self.quantity_extractor.parser.parse_text(cell_text)
                if cell_quantities:
                    quantities.append(cell_quantities[0])
        
        return quantities
    
    def _resolve_cell(self, row_name: str, col_name: str, table_data: Dict[str, Any]) -> Quantity:
        """Resolve a cell reference to a quantity"""
        # Find row index
        try:
            row_idx = int(row_name)
        except:
            data = table_data.get('data', [])
            row_idx = None
            for i, row in enumerate(data):
                if row and str(row[0]).lower() == str(row_name).lower():
                    row_idx = i
                    break
            
            if row_idx is None:
                return Quantity(value=0, unit="", original_text="", confidence=0)
        
        # Find column index
        headers = table_data.get('headers', [])
        col_idx = None
        for i, header in enumerate(headers):
            if str(header).lower() == str(col_name).lower():
                col_idx = i
                break
        
        if col_idx is None:
            try:
                col_idx = int(col_name)
            except:
                return Quantity(value=0, unit="", original_text="", confidence=0)
        
        # Get cell value
        data = table_data.get('data', [])
        if row_idx < len(data) and col_idx < len(data[row_idx]):
            cell_text = str(data[row_idx][col_idx])
            quantities = self.quantity_extractor.parser.parse_text(cell_text)
            if quantities:
                return quantities[0]
        
        return Quantity(value=0, unit="", original_text="", confidence=0)
    
    def _to_confidence_interval(self, data: Any) -> ConfidenceInterval:
        """Convert data to ConfidenceInterval object"""
        if isinstance(data, dict):
            return ConfidenceInterval(
                center=data.get('center', 0),
                lower=data.get('lower', 0),
                upper=data.get('upper', 0),
                confidence_level=data.get('confidence_level', 0.95),
                unit=data.get('unit', '')
            )
        elif isinstance(data, (list, tuple)) and len(data) >= 3:
            return ConfidenceInterval(
                center=float(data[0]),
                lower=float(data[1]),
                upper=float(data[2]),
                confidence_level=data[3] if len(data) > 3 else 0.95
            )
        else:
            # Try to parse as single value
            if isinstance(data, Quantity):
                value = data.value
                unit = data.unit
            else:
                value = float(data)
                unit = ""
            
            # Create a narrow interval around the value
            return ConfidenceInterval(
                center=value,
                lower=value * 0.95,
                upper=value * 1.05,
                unit=unit
            )