import aiohttp
import json
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class GooseClient:
    """Client for interacting with the Goose Agent API."""
    
    def __init__(self, base_url: str = "http://51.15.197.135:8000"):
        self.base_url = base_url
        self.endpoint = f"{base_url}/run-goose/"
        
    async def run_goose_agent(
        self, 
        instructions: str, 
        session_name: str,
        data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Send a request to the Goose Agent API.
        
        Args:
            instructions: Instructions for the Goose Agent
            session_name: Name of the session
            data: Optional data to include in the request
            
        Returns:
            Response from the Goose Agent API
        """
        payload = {
            "instructions": instructions,
            "session_name": session_name
        }
        
        if data:
            payload["data"] = data
            
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.endpoint, json=payload) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        error_text = await response.text()
                        logger.error(f"Goose API error: {response.status} - {error_text}")
                        return {
                            "error": f"Failed to get response from Goose API: {response.status}",
                            "details": error_text
                        }
        except Exception as e:
            logger.exception(f"Exception when calling Goose API: {str(e)}")
            return {"error": f"Exception when calling Goose API: {str(e)}"}
            
    async def format_invoice_for_goose(
        self,
        invoice_data: Dict[str, Any],
        task: str,
        additional_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Format invoice data for Goose Agent processing based on task.
        
        Args:
            invoice_data: The invoice data to format
            task: The task to perform (credit_assessment, lender_matching, etc.)
            additional_context: Additional context to include
            
        Returns:
            Formatted data ready for Goose API
        """
        # Create a session name based on invoice number
        invoice_number = invoice_data.get("invoice_number", "unknown")
        session_name = f"invoice-{invoice_number}-{task}"
        
        # Default instructions
        instructions = f"Process this invoice for {task}"
        
        # Task-specific formatting
        if task == "credit_assessment":
            instructions = "Assess the credit risk of this invoice based on buyer, amount, and payment terms."
        elif task == "lender_matching":
            instructions = "Identify the best lenders for this invoice based on its characteristics."
        elif task == "funding_recommendation":
            instructions = "Provide a funding recommendation for this invoice including advance rate and terms."
        
        # Prepare the data payload
        goose_data = {
            "invoice": invoice_data
        }
        
        # Include additional context if provided
        if additional_context:
            goose_data.update(additional_context)
            
        # Return formatted request
        return {
            "instructions": instructions,
            "session_name": session_name,
            "data": goose_data
        }
    
    async def get_credit_assessment(self, invoice_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get credit assessment for an invoice using Goose Agent.
        
        Args:
            invoice_data: The invoice data to assess
            
        Returns:
            Credit assessment from Goose Agent
        """
        goose_request = await self.format_invoice_for_goose(
            invoice_data=invoice_data,
            task="credit_assessment"
        )
        
        response = await self.run_goose_agent(
            instructions=goose_request["instructions"],
            session_name=goose_request["session_name"],
            data=goose_request["data"]
        )
        
        return response
    
    async def get_lender_matches(self, invoice_data: Dict[str, Any], credit_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get lender matches for an invoice using Goose Agent.
        
        Args:
            invoice_data: The invoice data
            credit_data: Credit assessment data
            
        Returns:
            Lender matches from Goose Agent
        """
        goose_request = await self.format_invoice_for_goose(
            invoice_data=invoice_data,
            task="lender_matching",
            additional_context={"credit_assessment": credit_data}
        )
        
        response = await self.run_goose_agent(
            instructions=goose_request["instructions"],
            session_name=goose_request["session_name"],
            data=goose_request["data"]
        )
        
        return response

# Create a singleton instance
goose_client = GooseClient()

if __name__ == "__main__":
    import asyncio
    async def main():
        client = GooseClient()
        response = await client.run_goose_agent(
            instructions="say hello",
            session_name="test-session"
        )
        print(response)
    asyncio.run(main())
