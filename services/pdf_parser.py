import io
import pdfplumber
import re
from typing import Dict, Any, Optional
from datetime import datetime

from models.invoice import InvoiceData, PartyInfo, LineItem

def extract_invoice_data(pdf_bytes: bytes) -> InvoiceData:
    """
    Extract structured data from an invoice PDF.
    
    Args:
        pdf_bytes: Raw PDF file bytes
        
    Returns:
        Structured invoice data
    """
    # Initialize with empty data structure
    invoice_data = InvoiceData()
    
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            # Extract text from all pages
            text = ""
            tables = []
            
            for page in pdf.pages:
                # Extract text
                page_text = page.extract_text() or ""
                text += page_text + "\n"
                
                # Extract tables
                page_tables = page.extract_tables()
                if page_tables:
                    tables.extend(page_tables)
            
            # Process text for invoice details
            invoice_data.invoice_number = extract_invoice_number(text)
            invoice_data.issue_date = extract_date(text, date_type="issue")
            invoice_data.due_date = extract_date(text, date_type="due")
            invoice_data.total_amount = extract_amount(text)
            invoice_data.currency = extract_currency(text)
            
            # Extract buyer and seller
            invoice_data.buyer = extract_party_info(text, party_type="buyer")
            invoice_data.seller = extract_party_info(text, party_type="seller")
            
            # Extract line items from tables if available
            if tables:
                invoice_data.line_items = extract_line_items(tables)
            
            # Calculate confidence score based on how many fields were successfully extracted
            invoice_data.confidence_score = calculate_confidence(invoice_data)
            
    except Exception as e:
        # Log the error
        print(f"Error extracting invoice data: {e}")
        # Still return the partial data we extracted
    
    return invoice_data

def extract_invoice_number(text: str) -> Optional[str]:
    """Extract invoice number from text."""
    patterns = [
        r"(?:Invoice|INV)(?:oice)?\s*(?:#|No|Number)?[:.\s]*([A-Za-z0-9-]+)",
        r"(?:Invoice|INV)(?:oice)?\s*(?:#|No|Number)?[:.\s]*([A-Za-z0-9-]+)",
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.I)
        if match:
            return match.group(1).strip()
    
    return None

def extract_date(text: str, date_type: str = "issue") -> Optional[str]:
    """Extract dates from text."""
    if date_type == "issue":
        patterns = [
            r"(?:Issue|Date|Invoice Date)[:.\s]*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",
            r"(?:Issue|Date|Invoice Date)[:.\s]*(\w+ \d{1,2},? \d{4})"
        ]
    else:  # due date
        patterns = [
            r"(?:Due|Payment) Date[:.\s]*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",
            r"(?:Due|Payment) Date[:.\s]*(\w+ \d{1,2},? \d{4})"
        ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.I)
        if match:
            # Try to standardize the date format
            date_str = match.group(1).strip()
            try:
                # This would be enhanced to handle various date formats
                return date_str
            except:
                return date_str
    
    return None

def extract_amount(text: str) -> Optional[float]:
    """Extract total amount from text."""
    patterns = [
        r"(?:Total|Amount Due|Balance Due)[:.\s]*[$€£]?([0-9,]+\.\d{2})",
        r"(?:Total|Amount Due|Balance Due)[:.\s]*[$€£]?([0-9,]+)"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.I)
        if match:
            # Remove commas and convert to float
            amount_str = match.group(1).replace(',', '')
            try:
                return float(amount_str)
            except:
                pass
    
    return None

def extract_currency(text: str) -> str:
    """Extract currency from text."""
    patterns = [
        r'[$€£¥]\s*[0-9,]+\.\d{2}'
    ]
    
    currency_map = {
        '$': 'USD',
        '€': 'EUR',
        '£': 'GBP',
        '¥': 'JPY'
    }
    
    for pattern in patterns:
        match = re.search(pattern, text)
        if match and match.group(0)[0] in currency_map:
            return currency_map[match.group(0)[0]]
    
    # Default to USD if no currency detected
    return "USD"

def extract_party_info(text: str, party_type: str) -> PartyInfo:
    """Extract buyer or seller information."""
    party = PartyInfo()
    
    # This is simplified - would need to be enhanced for real-world usage
    if party_type == "buyer":
        patterns = [
            r"(?:Bill To|Customer|Buyer|Client)[:.\s]*((?:[A-Za-z0-9,.]+ )+)",
        ]
    else:  # seller
        patterns = [
            r"(?:From|Seller|Vendor|Provider)[:.\s]*((?:[A-Za-z0-9,.]+ )+)",
        ]
    
    # Try to extract name
    for pattern in patterns:
        match = re.search(pattern, text, re.I)
        if match:
            party.name = match.group(1).strip()
            break
    
    # For a real implementation, we'd add more sophisticated extraction
    # of address, tax ID, etc.
    
    return party

def extract_line_items(tables) -> list:
    """Extract line items from table data."""
    line_items = []
    
    # This is a simplified implementation
    # In reality, we'd need more sophisticated table parsing
    
    for table in tables:
        # Skip tables that are too small
        if len(table) < 2:  # Need at least header and one data row
            continue
        
        # Try to identify if this looks like a line items table
        header_row = [str(cell).lower() if cell else "" for cell in table[0]]
        
        # Check if this looks like an items table
        keywords = ['description', 'item', 'quantity', 'qty', 'price', 'amount', 'total']
        if not any(keyword in " ".join(header_row) for keyword in keywords):
            continue
            
        # Find column indexes for item, quantity, price, amount
        desc_idx = next((i for i, h in enumerate(header_row) if 'desc' in h or 'item' in h), None)
        qty_idx = next((i for i, h in enumerate(header_row) if 'qty' in h or 'quant' in h), None)
        price_idx = next((i for i, h in enumerate(header_row) if 'price' in h or 'rate' in h), None)
        amount_idx = next((i for i, h in enumerate(header_row) if 'amount' in h or 'total' in h), None)
        
        # Process data rows
        for row_idx in range(1, len(table)):
            row = table[row_idx]
            
            # Skip empty rows
            if not any(cell for cell in row):
                continue
                
            item = LineItem()
            
            # Extract fields if column was identified
            if desc_idx is not None and desc_idx < len(row):
                item.description = row[desc_idx]
            
            if qty_idx is not None and qty_idx < len(row):
                try:
                    item.quantity = float(str(row[qty_idx]).replace(',', ''))
                except:
                    pass
                    
            if price_idx is not None and price_idx < len(row):
                try:
                    price_str = str(row[price_idx]).replace('$', '').replace(',', '')
                    item.unit_price = float(price_str)
                except:
                    pass
                    
            if amount_idx is not None and amount_idx < len(row):
                try:
                    amount_str = str(row[amount_idx]).replace('$', '').replace(',', '')
                    item.amount = float(amount_str)
                except:
                    pass
            
            # Add the item if we got any data
            if any([item.description, item.quantity, item.unit_price, item.amount]):
                line_items.append(item)
    
    return line_items

def calculate_confidence(invoice_data: InvoiceData) -> float:
    """Calculate confidence score based on completeness of extraction."""
    # Count how many essential fields were extracted
    essential_fields = [
        invoice_data.invoice_number,
        invoice_data.issue_date,
        invoice_data.total_amount,
        invoice_data.buyer.name, 
        invoice_data.seller.name
    ]
    
    extracted_count = sum(1 for field in essential_fields if field is not None)
    
    # Basic confidence calculation
    confidence = min(extracted_count / len(essential_fields), 1.0)
    
    # Boost confidence if we have line items
    if invoice_data.line_items:
        confidence = min(confidence + 0.1, 1.0)
        
    return confidence