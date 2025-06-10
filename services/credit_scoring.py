from typing import Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)

# Dependency to simulate a database (replace with actual DB later)
demo_invoices = {
    "demo-invoice-1": {
        "invoice_number": "INV-2025-001",
        "issue_date": "2025-06-01",
        "total_amount": 1000.00,
        "currency": "USD",
        "buyer": {"name": "Demo Buyer"},
        "seller": {"name": "Demo Seller"}
    }
}

def get_credit_score(invoice_data: Dict[str, Any]) -> Tuple[int, list]:
    """
    Calculate credit score and recommend lenders based on invoice data.
    """
    logger.info(f"Calculating credit score for invoice data: {invoice_data}")
    # Basic rule-based credit score (placeholder)
    if invoice_data["total_amount"] > 5000:
        credit_score = 600
    else:
        credit_score = 750
    
    # Placeholder lender matching
    lenders = ["Demo Lender A", "Demo Lender B"]  # Replace with actual lender matching
    
    return credit_score, lenders