from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import date

class PartyInfo(BaseModel):
    """Information about a party in the invoice (buyer or seller)"""
    name: Optional[str] = None
    address: Optional[str] = None
    tax_id: Optional[str] = None
    contact_email: Optional[str] = None
    contact_phone: Optional[str] = None

class LineItem(BaseModel):
    """Individual line item in the invoice"""
    description: Optional[str] = None
    quantity: Optional[float] = None
    unit_price: Optional[float] = None
    amount: Optional[float] = None

class InvoiceData(BaseModel):
    """Structured data extracted from an invoice"""
    invoice_number: Optional[str] = None
    issue_date: Optional[str] = None
    due_date: Optional[str] = None
    total_amount: Optional[float] = Field(None, description="Total amount due in the invoice")
    currency: Optional[str] = "USD"
    buyer: PartyInfo = Field(default_factory=PartyInfo)
    seller: PartyInfo = Field(default_factory=PartyInfo)
    line_items: List[LineItem] = Field(default_factory=list)
    payment_terms: Optional[str] = None
    confidence_score: float = Field(1.0, description="Confidence level of the extraction (0-1)")
    
    class Config:
        schema_extra = {
            "example": {
                "invoice_number": "INV-2025-001",
                "issue_date": "2025-05-01",
                "due_date": "2025-06-01",
                "total_amount": 1250.00,
                "currency": "USD",
                "buyer": {
                    "name": "Acme Corporation",
                    "address": "123 Main St, Anytown, USA",
                    "tax_id": "12-3456789"
                },
                "seller": {
                    "name": "Tech Solutions Inc",
                    "address": "456 Tech Ave, Innovation City, USA",
                    "tax_id": "98-7654321"
                },
                "line_items": [
                    {
                        "description": "Web Development Services",
                        "quantity": 10,
                        "unit_price": 125.00,
                        "amount": 1250.00
                    }
                ],
                "payment_terms": "Net 30",
                "confidence_score": 0.92
            }
        }