import pandas as pd
from typing import List, Optional
from tabulate import tabulate

class DomainInfoPrinter:
    """
    A class to print formatted domain information including subdomain details and parameters.
    """
    def __init__(self, domains: List[List[int]], 
                 xmin_list: List[float], xmax_list: List[float],
                 ymin_list: List[float], ymax_list: List[float],
                 x_mean_list: List[float], y_mean_list: List[float],
                 x_span_list: List[float], y_span_list: List[float]):
        """
        Initialize the DomainInfoPrinter with domain data.
        """
        self.domains = domains
        self.xmin_list = xmin_list
        self.xmax_list = xmax_list
        self.ymin_list = ymin_list
        self.ymax_list = ymax_list
        self.x_mean_list = x_mean_list
        self.y_mean_list = y_mean_list
        self.x_span_list = x_span_list
        self.y_span_list = y_span_list
        
        self._validate_inputs()

    def _validate_inputs(self) -> None:
        """Validate that all input lists have consistent lengths."""
        n_domains = len(self.domains)
        lists_to_check = [
            self.xmin_list, self.xmax_list, 
            self.ymin_list, self.ymax_list,
            self.x_mean_list, self.y_mean_list,
            self.x_span_list, self.y_span_list
        ]
        
        for lst in lists_to_check:
            if len(lst) != n_domains:
                raise ValueError(
                    f"All input lists must have length {n_domains} matching number of domains"
                )

    def print_basic_info(self) -> None:
        """Print basic domain information including number of subdomains."""
        print(f"\nNumber of subdomains = {len(self.domains)}")

    def print_subdomain_table(self) -> None:
        """Print table showing subdomain IDs and their cell IDs."""
        # Convert all values to strings explicitly
        data = []
        for i in range(len(self.domains)):
            cell_ids_str = ", ".join(map(str, self.domains[i]))  # Convert cell IDs to comma-separated string
            data.append([str(i), cell_ids_str])
        
        # Create table headers and data
        headers = ["Subdomain ID", "Cell IDs"]
        print("\nSubdomain Information:")
        print(tabulate(data, headers=headers, tablefmt='grid'))

    def print_parameter_table(self) -> None:
        """Print table showing all subdomain parameters."""
        data = []
        for i in range(len(self.domains)):
            row = [
                str(i),  # Convert ID to string
                f"{self.xmin_list[i]:.4f}",
                f"{self.xmax_list[i]:.4f}", 
                f"{self.ymin_list[i]:.4f}",
                f"{self.ymax_list[i]:.4f}",
                f"{self.x_mean_list[i]:.4f}",
                f"{self.y_mean_list[i]:.4f}", 
                f"{self.x_span_list[i]:.4f}",
                f"{self.y_span_list[i]:.4f}"
            ]
            data.append(row)
        
        headers = ["ID", "x_min", "x_max", "y_min", "y_max", 
                  "x_mean", "y_mean", "x_span", "y_span"]
        
        print("\nSubdomain Parameters:")
        print(tabulate(data, headers=headers, tablefmt='grid'))

    def print_domain_info(self) -> None:
        """Print all domain information in formatted tables."""
        try:
            self.print_basic_info()
            self.print_subdomain_table()
            self.print_parameter_table()
        except Exception as e:
            print(f"Error printing domain information: {str(e)}")
